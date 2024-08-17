
import pickle
import numpy                   as np
import pandas                  as pd

from scipy.optimize            import minimize



"""

This file contains functions that estimates the multivariate Hawkes model parameters by maxmimizing the log-likelihood function.
The log-likelihood function is computed according to equation (5) in the main text.
The parameters for each emotion (process) are estimated independently.

"""

# load data
with open('bootstrapped_index_fitting.pkl', 'rb') as file:
    indices = pickle.load(file)

with open('event_times.pkl', 'rb') as file:
    event_times_all = pickle.load(file)

with open('end_time.pkl', 'rb') as file:
    end_time_all = pickle.load(file)

with open('integrals_list_for_fitting.pkl', 'rb') as file:
    integrals_list_all = pickle.load(file)   

with open('mu_list_for_fitting.pkl', 'rb') as file:
    mu_list_all = pickle.load(file)



def fit_Hawkes_multi( timestamps_all = event_times_all[0], end_time_all = end_time_all[0], integral_sum_all = integrals_list_all[0], mu_1_sum_all = mu_list_all[0] , alpha = [0.1]*6 , gamma = 0.1, mu_constant= 0.5, nu = 0.5, row = 0):
    """
    This function estmated the hawkes process parameters by minimizing the (-log-likelihood function).
    The exponential decay component is calculated seperately in decay_multi.
    The log_likelihood function calculates the log-likelihood function.
    The log_likelihood_agg function calcualtes the aggreagted log-likelihood function across all videos.
    The objective function passes the function parameters to be estimate via scipy.minimize.
    
    params:
    
    timestamps_all: the timestamps at whcih an event takes place
    end_time_all: the endtime of the observation period
    integral_sum_all
    mu_func_all: a time-varying function of mu which is given
    alpha: parameter to estimate 
    gamma: parameter to estimate
    mu_constant: parameter to estimate
    nu: parameter to estimate
    row: the parameters are estimated seperately for each emotion; this indicates the corresponding index of emotion for estimation
    
    """
    

    def binary_search_max_smaller(arr, target):
        """
        Performs a binary search on a sorted array to find the maximum value that is smaller than a given target value.
        Returns the found value and its index in the array, if no such value exists, returns -1 for both.
        This function is efficient for finding the closest smaller value in a sorted list.

        params:
        arr: a list of elements to search for the maximum value smaller than the target
        target: the target value for which the function searches for the closest smaller element in 'arr'

        """
            
        left, right  = 0, len(arr) - 1            # initialize pointers for binary search
        index, value = -1, float('-1')            # initialize index and value to store results

        while left <= right:
            mid = left + (right - left) // 2      # calculate mid-point of current search range

            if arr[mid] < target:                
                value = arr[mid]                  # update value if current mid is smaller than target
                index = mid                       # update index to current mid
                left  = mid + 1                   # move left pointer to search in the right half
            else:
                right = mid - 1                   # move right pointer to search in the left half

        return value, index                       # return the found value and its index


    def find_max_smaller_values(list1, list2):
        """
        For each item in list1, finds the largest value in list2 that is smaller than the item.
        Returns a list of tuples, where each tuple contains the found value and its index in list2.
        The function sorts list2 to enable efficient binary search, making it faster for large datasets.

        params:

        list1: a list of target values for which to find the largest smaller value in list2
        list2: a list of values to be searched; the function will find the maximum value in this list that is smaller than each element in list1
        
        """
            
        list2.sort()      # sort the second list for binary search
        results = []      # initialize list to store results

        for item in list1:
            value, index = binary_search_max_smaller(list2, item) # find max smaller value for each item in list1
            results.append((value, index))                         

        return results


    def decay_binary(timestamps, gamma, t):
        """
        Computes the decay function values for a set of time points with decay parameter 'gamma'.
        This function computes the decay component recursively: R(i) = e^{-(1/gamma)(t_i - t_{i-1})}(1 + R(i-1)).

        params:
        timestamps: a sequence of time points representing when past events occurred.
        gamma: the decay rate for the exponential kernel, controlling how quickly the influence of past events decreases
        t: a sequence of time points at which to calculate the cumulative decay effects.
    
        """
            
        j_array    = np.zeros(len(timestamps))                # initialize an array to store cumulative decay effects up to each timestamp

        timestamps = np.array(timestamps)                     # convert the timestamps to a numpy array for easier manipulation

        past_array = find_max_smaller_values(t, timestamps )  # find the most recent timestamp smaller than each time point in 't'
            
        dt = np.zeros(len(timestamps))                        # initialize an array to store time differences between consecutive timestamps
        es = np.zeros(len(timestamps))                        # initialize an array to store exponential decay values for each time difference
        dt[1:] = np.diff(timestamps)                          # calculate the time differences between consecutive timestamps
        es[1:] = np.exp(-(1/gamma)* dt[1:] )                  # compute the exponential decay for each time difference

        for i in range(1, len(timestamps)):
            j_array[i] = es[i]*(1+j_array[i-1])               # calculate the decay for each timestamp recursively

        array = np.zeros(len(t))                              # initialize the result array to store decay values for each time point in 't'

        for j in range(0, len(t)):

            if past_array[j][0] != -1:                        # exclude cases where there are no past events for current time point 't'
                
                array[j] = np.exp(-(1/gamma)* (t[j] - past_array[j][0])) * (1 + j_array[int(past_array[j][1])]) # calculate the decay value at 't[j]' recursively by considering both the decay from the most recent event and the decay of all previous events
                                                                                                                # past_array[j][0] gives the timestamp of the most recent past event, past_array[j][1] gives the index of the most recent past event, j_array[int(past_array[j][1])] locates the cumulative influence up till the most recent past event                                                                                      
        
        return array


    def decay_multi(timestamps, gamma, alpha, t):
        """
        Extends the decay_binary function to a multivariate case, where multiple sequences of events are considered.
        The function iteratively computes the decay effects at each timepoint with respect to events in each process and takes the sum.
        Essentially, this function computes the cumulative influence of past events in all processes for each point in time.

        params:
        timestamps: a list where each element is a sequence of time points representing when events occurred for one process in a multivariate Hawkes process
        gamma: the decay parameter for the exponential decay function, controlling how quickly the influence of past events decreases
        alpha: a list of scaling factors, representing the relative influence from events in process f to events in process e
        t: a sequence of time points at which to calculate the cumulative decay effects
        """
                
        initial_array = []                           # initialize a list to store the decay effects from sequence of events in each process
        for i in range(len(timestamps)):             # iterate over each sequence of timestamps
            initial_array.append(1/gamma*alpha[i]*decay_binary(timestamps[i], gamma, t))  # calculate the decay array for each sequence using decay_binary, scaled by the corresponding alpha value

        array = np.array(initial_array)              # convert the list of decay arrays to a numpy array for easier manipulation
        
        max_length = max(len(arr) for arr in array)  # find the maximum length among the decay arrays
        sum_array  = np.sum([np.pad(arr, (0, max_length - len(arr)), 'constant') for arr in array], axis=0) # get column-wise sum for arrays of different length

        return sum_array

    def log_likelihood( alpha, gamma, nu,  mu_constant, timestamps, end_time, integral_sum, mu_1_sum, t):
        """
        This function computes the log-likelihood of a multivariate Hawkes process with exponential decay. 
        This function combines contributions from various sources and calculates the log-likelihood based on the model parameters.

        parameters:
        alpha: a list of scaling factors of the endogenous influence for each process
        gamma: the decay parameter for the exponential kernel, controlling how quickly the influence of past events decreases
        nu: a list of scaling factors of the video influence for each process
        mu_constant: a constant term for the baseline intensity, representing the baseline event rate independent of prior events
        timestamps: a list where each element is a sequence of time points representing when events occurred for different processes
        end_time: the end time of the observation period
        integral_sum: precomputed integrals over time for the function of video influence
        mu_1_sum: a matrix of precomputed values that represent the video influence at each time for each emotion
        t: a sequence of time points of event arrival with specified emotion 

        """
            
        r = decay_multi(timestamps, gamma, alpha, t) # compute the decay effects for all processes using the decay_multi function
        
        array_ =  []
        for i in range(len(timestamps)):
            array_.append(alpha[i]*(np.exp((-1/gamma)*(end_time - timestamps[i])) - 1))  # for each process, calculate the decay contribution from events up to the end of the observation period 
 
        # convert to numpy arrays for efficient numerical operations
        array = np.array(array_)
        integral_sum = np.array(integral_sum)
        nu = np.array(nu)

        # sum the decay contributions across all processes
        sums = [np.sum(arr) for arr in array]
        sums = sum(sums)
        
        # define the log-likelihood function by compiling differnet parts of the equation
        llh =  -np.sum(nu*integral_sum) -end_time * mu_constant + sums + \
               np.sum(np.log((nu.reshape(6,1)*mu_1_sum).sum(axis = 0) + mu_constant +  r))

        return llh
    
    def log_likelihood_agg(gamma, alpha, nu, mu_constant, timestamps_all, end_time_all, mu_1_sum_all, integral_sum_all):
        """
        This function computes the aggregated log-likelihood function across different videos. 
        It iterates through all videos in the sample and compute the log-likelihood values for each video with the log_likelihood function.
        Subsequently, the function sums up the log-likelihood values from all videos for parameter estimation.

        params:
        gamma (float): the decay parameter for the exponential kernel, controlling how quickly the influence of past events decreases
        alpha (list of floats): a list of scaling factors of the endogenous influence for each process
        nu (list or array): a list of scaling factors of the video influence for each process
        mu_constant (float):  a constant term for the baseline intensity, representing the baseline event rate independent of prior events
        timestamps_all (list of lists): a list where each element is a sequence of time points for different videos
        end_time_all (list of floats): a list of observation time corresponding to each video
        mu_1_sum_all (list of arrays): a list of arrays representing the video influence for each emotion in each video
        integral_sum_all (list of arrays): a list of precomputed integrals over time for the function of video influence for each video
        """
                
        log_likelihood_sum = 0                             # initialize the sum of log-likelihoods at zero
        for i in range(len(timestamps_all)):               # iterate through each video in the sample
   
            timestamps = timestamps_all[i]                 # get the sequence of time points for the current video
            mu_1_sum   = np.array(mu_1_sum_all[i][row])    # extract the video influence for the current video for given emotion
            t          = np.array(timestamps[row])         # extract the event timepoints for given emotion to estimate parameters for, convert to numpy array
            end_time        = end_time_all[i]              # get the end time for the current video
            integral_sum    = integral_sum_all[i]          # get the precomputed integrals for the current video
            # compute the log-likelihood for the current video
            log_likelihood_ = log_likelihood(alpha, gamma, nu,  mu_constant, timestamps, end_time, integral_sum, mu_1_sum, t)

            log_likelihood_sum += log_likelihood_          # aggregate the log-likelihood value across all videos

        return log_likelihood_sum
    
    def objective(params, *args):
        """
        This function computes the objective function used in the optimization process to estimate the parameters of a multivariate Hawkes model.
        The function computes the negative log-likelihood based on the given parameters, which the optimization routine seeks to minimize.

        parameters:
        params (array-like): a flat array containing all the model parameters to be optimized
                            the array is structured as [alpha_0, ..., alpha_5, nu_0, ..., nu_5, gamma, mu_constant]
        *args: additional arguments passed to the function, containing data and precomputed values needed for the log-likelihood calculation
            specifically, this includes timestamps_all, end_time_all, mu_1_sum_all, and integral_sum_all

        """
        # unpack the parameters from the flat array
        mu_constant    = params[-1]
        gamma = params[-2]
        nu    = params[6:12]
        alpha = params[0:6]

        timestamps_all, end_time_all, mu_1_sum_all, integral_sum_all = args # extract the data and precomputed values for all videos in the sample
        
        # compute the negative log-likelihood to be minimized using the log_likelihood_agg function
        llh = - log_likelihood_agg(gamma, alpha, nu, mu_constant, timestamps_all, end_time_all, mu_1_sum_all, integral_sum_all)
  
        return llh
    
    initial_guess = np.concatenate([alpha, nu, [gamma], [mu_constant]])     # define starting values for parameters
    bounds  = [(0, 50)] * len(alpha) + [(1e-6, 10)] * len(nu) + [(0.1, 20)] * len([gamma]) + [(0, 50)] * len([mu_constant])            # define the bound constraints for each parameter
    results = minimize(objective, initial_guess, args=(timestamps_all, end_time_all, mu_1_sum_all, integral_sum_all), bounds=bounds)   # pass to scipy.minimize for parameter estimation
    
    # extract estimated parameters
    mu_estimated    = results.x[-1]      
    gamma_estimated = results.x[-2]
    nu_estimated    = results.x[6:12]
    alpha_estimated = results.x[0:6]


    return alpha_estimated, gamma_estimated, nu_estimated, mu_estimated



# estimate multivariate Hawkes parameters
# adjust the "row" parameter as the index of emotion in the list below to estimate model parameters for each emotion 
# ['anger',     'disgust',     'fear',      'joy',    'sadness',     'surprise']

# collect estimated parameters from each bootstrapped sample
alpha_estimated_list = []
gamma_estimated_list = []
nu_estimated_list = []
mu_estimated_list = []

for i in range(len(indices)):
    # obtain corresponding data for the list of videos according to the lists of bootstrapped index
    end_time       = [end_time_all[j] for j in indices[i]]
    event_times    = [event_times_all[j] for j in indices[i]]
    integrals_list = [integrals_list_all[j] for j in indices[i]]
    mu_list        = [mu_list_all[j] for j in indices[i]]
    alpha_estimated, gamma_estimated, nu_estimated, mu_estimated = fit_Hawkes_multi( timestamps_all = event_times, end_time_all = end_time, integral_sum_all =integrals_list, mu_1_sum_all = mu_list , alpha = [0.1]*6 , gamma = 0.1, mu_constant= 0.1, nu = [0.1]*6, row = 3)
    alpha_estimated_list.append(alpha_estimated)
    gamma_estimated_list.append(gamma_estimated)
    nu_estimated_list.append(nu_estimated)
    mu_estimated_list.append(mu_estimated)

# return estimated parameters 
df_alpha = pd.DataFrame(alpha_estimated_list)
df_gamma = pd.DataFrame(gamma_estimated_list)
df_nu    = pd.DataFrame(nu_estimated_list)
df_mu    = pd.DataFrame(mu_estimated_list)

# store results
df_alpha.to_csv('...')
df_gamma.to_csv('...')
df_nu.to_csv('...')
df_mu.to_csv('...')






    


    
    
