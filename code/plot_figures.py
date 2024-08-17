
import pickle
import os

import numpy                   as np
import pandas                  as pd
import seaborn                 as sb
import matplotlib.pyplot       as plt
import matplotlib.pyplot       as plt
import matplotlib.gridspec     as gridspec
import matplotlib.gridspec     as gridspec
import matplotlib.lines        as mlines

from scipy                     import interpolate
from matplotlib.ticker         import FormatStrFormatter
from matplotlib.collections    import PolyCollection
from matplotlib.gridspec       import GridSpec

from matplotlib                import pyplot               as plt


"""

This file contains functions that generates the figures in the paper.

"""

def figure_1(df_transcript, df_livechat, df_s, df_times):
    """
    This function generates Figure 1 (b), (c), (d) in the main text.
    It plots the arrival of angry and sad events in the live chats, and time varying video influence of the emotion sadness.
    The sample video id is Sb179GLPNYE.

    params:
    df_transcript: a dataframe containing the transcript data for sample video
    df_livechat: a dataframe containing live chat data with for sample video
    df_s: a dataframe containing the time varying intensity of the emotion sadness in the video subtitles
    df_times: a dataframe containing the corresponding time points for the video sadness intensity values

    """

    df_s.columns     = ['intensity']                          # rename columns
    df_times.columns = ['times']
    df_figure_1      = pd.concat([df_s, df_times],  axis=1)   # concat dataframes horizontally

    df_transcript_sadness = pd.DataFrame(df_transcript[df_transcript['basic_sadness'] == 1]['start_minute'])  # restrict to transcripts with sadness
    df_livechat_sadness   = pd.DataFrame(df_livechat[df_livechat['basic_sadness'] == 1]['time_minute'])       # restrict to live chats with sadness
    df_livechat_anger     = pd.DataFrame(df_livechat[df_livechat['basic_anger'] == 1]['time_minute'])         # restrict to live chats with anger

    df_transcript_sadness['start_minute_'] = df_transcript_sadness['start_minute']+ 2/60                      # the lognormal function peaks at 2 seconds after transcript appearace, adjust time scale to match transcript arrival for visual convinience

    func_sadness             = interpolate.interp1d(df_figure_1['times'], df_figure_1['intensity'], kind= 'linear', bounds_error=True)    # generate interpolation function based on the time varying emotion intensity
    vals                     = func_sadness(df_transcript_sadness['start_minute_'].values)                                                # evaluate function at timepoints where transcript arrive
    vals[vals                <= 0] = 1e-10                                                                    # replace zero and negative values with 1e-6
    df_transcript_sadness['interpolated'] = vals                                                              # store interpolated values

    # get data within time span
    df_livechat_sadness   = df_livechat_sadness[(df_livechat_sadness['time_minute'] >= 80)&( df_livechat_sadness['time_minute'] <= 95)]  
    df_livechat_anger     = df_livechat_anger[(df_livechat_anger['time_minute'] >= 80)&( df_livechat_anger['time_minute'] <= 95)]
    df_transcript_sadness = df_transcript_sadness[(df_transcript_sadness['start_minute'] >= 80)& ( df_transcript_sadness['start_minute'] <= 95)]
    df_figure_1_ = df_figure_1[(df_figure_1['times'] >=80)& (df_figure_1['times'] <=95)]

    # plotting
    fs = 12                                                    # set the font size for the plot
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 3])      # define a 3-row grid for subplots with specified height ratios
    plt.figure(figsize=(12, 8))                                # set the figure size

    # plot the live chat sadness data as a rug plot and scatter plot
    ax0 = plt.subplot(gs[0])
    sb.rugplot(df_livechat_sadness['time_minute'],  label = 'live-chat events',  color='dimgrey', height = 0.3, linewidth=1.5, ax = ax0)    # rug plot of sadness events 
    sb.scatterplot(x = 'time_minute', y = [1]*len(df_livechat_sadness), data = df_livechat_sadness, marker='D', color='royalblue', s=50, edgecolor='none', ax = ax0)  # scatter plot of sadness events
    ax0.set_ylabel('sad', fontsize =16, labelpad=35)           # set the y-axis label for sadness
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # format y-axis labels to one decimal place
    ax0.set_ylim(0, 3)                    # set y-axis limits
    ax0.set_yticks([])                    # remove y-axis ticks
    dot_legend = mlines.Line2D([], [], color='royalblue', marker='D', linestyle='None',
                            markersize=5, label='sad comment') # create legend entry for sadness comments
    plt.legend(handles=[dot_legend])      # add legend to the plot
    ax0.set_xlim(79.8, 94.5)              # set x-axis limits

    # plot the live chat anger data as a rug plot and scatter plot
    ax1 = plt.subplot(gs[1], sharex=ax0)  # create the second subplot, sharing the x-axis with the first
    sb.rugplot(df_livechat_anger['time_minute'],  label = 'live-chat events',  color='dimgrey', height = 0.3, linewidth=1.5,ax = ax1) # rug plot of anger events
    sb.scatterplot(x= 'time_minute', y = [1]*len(df_livechat_anger), color='firebrick',  marker='D', data = df_livechat_anger, s = 50, edgecolor='none',  ax = ax1)   # scatter plot of anger events
    ax1.set_ylabel('angry', fontsize =16, labelpad=35) # set the y-axis label for anger
    ax1.set_xlabel('minute in video', fontsize = 16 )  # set the x-axis label
    ax1.set_ylim(0, 3)                    # set y-axis limits
    ax1.set_yticks([])                    # remove y-axis ticks
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))    # format y-axis labels to one decimal place
    dot_legend = mlines.Line2D([], [], color='firebrick', marker='D', linestyle='None',
                            markersize=5, label='angry comment') # create legend entry for anger comments
    plt.legend(handles=[dot_legend])      # add the legend to the plot
    ax1.set_xlim(79.8, 94.5)              # set x-axis limits

    # plot the interpolated sadness intensity and the baseline intensity
    ax2 = plt.subplot(gs[2], sharex=ax0)  # create the third subplot, sharing the x-axis with the first
    sb.scatterplot( x = 'start_minute_',  y = 'interpolated',data = df_transcript_sadness, color =  'royalblue', label = 'sad transcript', edgecolor='none', s=50, ax = ax2)   # plot arrival of transcripts with scatter plot
    sb.lineplot(   x = 'times',  y = 'intensity',  data = df_figure_1_ , color = 'dimgrey', label = r'$S^{sad}$', linestyle='--', ax = ax2)        # plot time varying intensity with line plot
    ax2.set_ylabel(r'$S^{sad}$', fontsize =16)

    # customize the legend
    handles, labels = ax2.get_legend_handles_labels() # get legend objects to customize the legend
    order   = [1, 0]                                  # reorder the handles and labels 
    handles = [handles[i] for i in order]
    labels  = [labels[i] for i in order]

    ax2.legend(handles, labels)                       # create legend with the specified order
    ax2.legend(handles, labels, loc='upper left', fontsize='large') # create and display the legend
    ax2.set_xlabel('minute in video', fontsize = 16)
    ax2.set_xlim(79.8, 94.5)                          # set x-axis limits
 
    # format figures
    for ax in [ax0, ax1]:
        ax.label_outer()                               

    ax0.set_xlabel('')
    ax0.tick_params(length=0) 
    ax1.set_xlabel('')
    ax1.tick_params(length=0) 
    ax2.tick_params(axis='x', labelsize=fs + 2) 
    ax0.tick_params(axis='y', labelsize=fs + 2) 
    ax1.tick_params(axis='y', labelsize=fs + 2) 
    ax2.tick_params(axis='y', labelsize=fs + 2) 

    plt.savefig('...', bbox_inches='tight', transparent=True) # store figure to output

# result visualization

basic_emotions = ['basic_anger', 'basic_disgust', 'basic_fear', 'basic_joy',  'basic_sadness', 'basic_surprise']
emotions       = ['anger', 'disgust', 'fear', 'joy',  'sadness', 'surprise']
emotions_plot  = ['joy',  'surprise', 'anger', 'disgust', 'fear',   'sadness']

def cal_spectral_radius(matrix):
    """
    Calcuates the spectral radius for a given matrix.

    params:
    matrix: the input matrix to calculate spectral radius

    """
    eigenvalues = np.linalg.eigvals(matrix)   # computes eigenvalues

    return max(abs(eigenvalues))              # return maximum eigenvalue


def get_param_vals(dir_ , emotions = emotions):
    """
    Returns the estimated values in correct shape for subsequent computations.

    params:
    dir_: the directory to load files with estimated values
    emotions: the list of emotions

    """
    
    alpha_coefs = {}
    for i in emotions:
        alpha_coefs[i] = pd.read_csv(dir_ + f'df_alpha_{i}.csv', index_col = 0)    # load alpha coefficients to dataframe
        alpha_coefs[i].columns = emotions

    gamma_coefs = {}
    for i in emotions:
        gamma_coefs[i] = pd.read_csv(dir_ + f'df_gamma_{i}.csv', index_col = 0)    # load gamma coefficients to dataframe
        gamma_coefs[i].columns = ['$\gamma$']

    mu_coefs = {}
    for i in emotions:
        mu_coefs[i] = pd.read_csv(dir_ + f'df_mu_{i}.csv', index_col = 0)          # load mu coefficients to dataframe
        mu_coefs[i].columns = ['$\mu$_0']

    nu_coefs = {}
    for i in emotions:
        nu_coefs[i] = pd.read_csv(dir_ + f'df_nu_{i}.csv', index_col = 0)          # load nu coefficients to dataframe
        nu_coefs[i].columns    = emotions

    # sort column and index orders
    alpha_mean = pd.DataFrame([alpha_coefs[e].mean() for e in emotions], index = emotions) 
    nu_mean    = pd.DataFrame([nu_coefs[e].mean() for e in emotions], index = emotions)
    gamma_mean = pd.DataFrame([gamma_coefs[e].mean() for e in emotions], index = emotions)
    mu_mean    = pd.DataFrame([mu_coefs[e].mean() for e in emotions], index = emotions)

    gamma_val  = gamma_mean.values
    mu_val     = mu_mean.values
    nu_val     = nu_mean.values
    alpha_val  = alpha_mean.values

    return gamma_val, mu_val, nu_val, alpha_val


def get_param_mean(dir_, emotions = emotions, emotions_plot = emotions_plot, num_samples = 1000):
    """
    Returns the estimated values in dataframes for the heatmap visualization plot.

    params:
    dir_: the directory to load files with estimated values
    emotions: the list of emotions
    emotions_plot: order of emotions for plotting
    num_samples: number of times the alpha matrix is sampled to obtain the scale of the spectral radius

    """

    # load estimated parameters in the correct shape and order
    alpha_coefs = {}
    for i in emotions:
        alpha_coefs[i] = pd.read_csv(dir_ + f'df_alpha_{i}.csv', index_col = 0)
        alpha_coefs[i].columns = emotions
        alpha_coefs[i] = alpha_coefs[i][emotions_plot]                           
        
    gamma_coefs = {}
    for i in emotions:
        gamma_coefs[i] = pd.read_csv(dir_ + f'df_gamma_{i}.csv', index_col = 0)
        gamma_coefs[i].columns = ['$\gamma$']
        
    mu_coefs = {}
    for i in emotions:
        mu_coefs[i] = pd.read_csv(dir_ + f'df_mu_{i}.csv', index_col = 0)
        mu_coefs[i].columns = ['$\mu$_0']
        
    nu_coefs = {}
    for i in emotions:
        nu_coefs[i] = pd.read_csv(dir_ + f'df_nu_{i}.csv', index_col = 0)
        nu_coefs[i].columns    = emotions
        nu_coefs[i] = nu_coefs[i][emotions_plot]
        
    # obtain the mean and standard deviation of estimated parameters in bootstrapped samples and return in annotated dataframe
    alpha_mean = pd.DataFrame([alpha_coefs[e].mean() for e in emotions_plot], index = emotions_plot)
    alpha_std  = pd.DataFrame([alpha_coefs[e].std() for e in emotions_plot], index = emotions_plot)
    alpha_heatmap = alpha_mean.round(3).astype(str) +  os.linesep + u" \u00B1 " + alpha_std.round(3).astype(str)
    
    nu_mean = pd.DataFrame([nu_coefs[e].mean() for e in emotions_plot], index = emotions_plot)
    nu_std  = pd.DataFrame([nu_coefs[e].std() for e in emotions_plot], index = emotions_plot)
    nu_heatmap = nu_mean.round(3).astype(str) +  os.linesep + u" \u00B1 " + nu_std.round(3).astype(str)
    
    gamma_mean = pd.DataFrame([gamma_coefs[e].mean() for e in emotions_plot], index = emotions_plot)
    gamma_std  = pd.DataFrame([gamma_coefs[e].std() for e in emotions_plot], index = emotions_plot)
    gamma_heatmap = gamma_mean.round(1).astype(str) +  os.linesep + u" \u00B1 " + gamma_std.round(2).astype(str)

    mu_mean = pd.DataFrame([mu_coefs[e].mean() for e in emotions_plot], index = emotions_plot)
    mu_std  = pd.DataFrame([mu_coefs[e].std() for e in emotions_plot], index = emotions_plot)
    mu_heatmap = mu_mean.round(3).astype(str) +  os.linesep + u" \u00B1 " + mu_std.round(3).astype(str)

    alpha_val       = alpha_mean.values
    eigenvalues     = np.linalg.eigvals(alpha_val)
    spectral_radius = max(abs(eigenvalues))
    
    # obtain interval for the spectral radius by sampling the alpha matrix with mean and standard deviation
    num_samples     = num_samples
    sample_matrices = []
    for _ in range(num_samples):
        sample_matrix = alpha_mean + np.random.randn(*alpha_mean.shape) * alpha_std  # generate 'num_samples' matrices by sampling from a normal distribution centered at 'alpha_mean' with standard deviation 'alpha_std'
        sample_matrices.append(sample_matrix)
    spectral_r = [cal_spectral_radius(matrix) for matrix in sample_matrices]         # obtain the spectral radius of sampled matrixes
    max_spectral_radius = np.max(spectral_r)
    min_spectral_radius = np.min(spectral_r)
    scale = [min_spectral_radius /(1-min_spectral_radius ), max_spectral_radius/ (1-max_spectral_radius)] # get the interval of the ratio of endogenous events with the minimal and maximum spectral radius
    

    return alpha_heatmap, nu_heatmap, gamma_heatmap, mu_heatmap,  alpha_mean, nu_mean, gamma_mean, mu_mean, spectral_radius, scale

    
def results_visualization(figure, 
                            df_alpha, 
                            alpha_annote, 
                            df_gamma, 
                            gamma_annote, 
                            df_nu, 
                            nu_annote, 
                            df_mu, 
                            mu_annote):
    
    """
    This function generates a horizontal figure with two sets of heatmaps visualizing the model parameters in a multivariate Hawkes model. 
    This function creates a layout with four heatmaps arranged in a row visualizing different model parameters.

    params:
    figure: the name of the figure file 
    df_alpha: dataframe containing the alpha parameter for endogenous events.
    alpha_annote: dataframe containing the annotations for the alpha heatmap.
    df_gamma: dataframe containing the gamma parameter for the model.
    gamma_annote: dataframe containing the annotations for the gamma heatmap.
    df_nu: dataframe containing the nu parameters
    nu_annote: dataframe containing the annotations for the nu heatmap
    df_mu: dataframe containing the baseline intensities (mu) for each emotion
    mu_annote: dataframe containing the annotations for the mu heatmap

    """
        
    fig = plt.figure(figsize=(20, 8))                           # set the figure size
    gs = gridspec.GridSpec(1, 5, width_ratios=[6,1, 0.5, 6,1])  # specify the grid layout with relative width ratios
    fs = 14

    # create the first 6x6 heatmap for alpha
    ax1 = plt.subplot(gs[0])
    sb.heatmap(df_alpha, annot=alpha_annote, fmt='', cmap='Blues', annot_kws={"size": fs}, ax=ax1, vmax = 0.5, cbar=False) # set vmax for visual convinience
    ax1.set_title(r'$Endo$ Excitation $\alpha^{e,f}$', fontsize=fs + 10)
    ax1.set_xlabel('Source Emotion $f$', fontsize=fs + 8)
    ax1.set_ylabel('Target Emotion $e$', fontsize=fs + 8)
    ax1.tick_params(axis='x', labelsize=fs + 6)
    ax1.tick_params(axis='y', labelsize=fs + 6)
    ax1.tick_params(length=0) 

    # create the second 1x6 heatmap for gamma
    ax2 = plt.subplot(gs[1])
    sb.heatmap(df_gamma, annot=gamma_annote, fmt='', annot_kws={"size": fs}, cmap='Blues', ax=ax2, cbar=False)
    ax2.set_title("$\gamma^{e}$", fontsize=fs + 10)
    ax2.set_xticklabels([])  # set axis labels empty
    ax2.set_yticklabels([])  
    ax2.set_xticks([])       # hide axis ticks
    ax2.set_yticks([]) 
    
    # create the third 6x6 heatmap for nu
    ax3 = plt.subplot(gs[3])
    sb.heatmap(df_nu, annot=nu_annote, fmt='', cmap='Blues', ax=ax3, annot_kws={"size": fs}, vmax  = 0.08, cbar=False)   # set vmax for visual convinience
    ax3.set_title(r'$Exo$ Influence $\nu^{e,f}$', fontsize=fs + 10)
    ax3.set_xlabel('Source Emotion $f$', fontsize=fs + 8)
    ax3.set_ylabel('Target Emotion $e$', fontsize=fs + 8)
    ax3.tick_params(axis='x', labelsize=fs + 6)
    ax3.tick_params(axis='y', labelsize=fs + 6)
    ax3.tick_params(length=0) 

    # create the fourth 1x6 heatmap for mu
    ax4 = plt.subplot(gs[4])
    sb.heatmap(df_mu, annot=mu_annote, fmt='', annot_kws={"size": fs}, cmap='Blues', ax=ax4, cbar=False)
    ax4.set_title("$\mu_0^{e}$", fontsize=fs + 10)
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])  
    ax4.set_xticks([]) 
    ax4.set_yticks([])  

    plt.tight_layout()
    fig.savefig(f'{figure}.pdf', bbox_inches='tight', transparent=True) # store figure to output
    plt.show()

# influence ratios visualization

def hawkes_endo_exo_percentage(num, 
                               timestamps_all,   
                               mu_1_sum_all, 
                               alpha, 
                               gamma, 
                               mu_constant, 
                               nu, 
                               row):
    
    """
    This function computes the values of exogenous and endogenous components according to equation (1) in the paper with the mean estimated parameter values. 
    It returns the distribution of the ratio of exogenous and endogenous influences across all videos.
    
    params:
    num: the number of dimension for the model (number of emotions)
    timestamps_all: a list where each element is a sequence of time points for different videos
    mu_1_sum_all: a list of arrays representing the video influence for each emotion in each video
    gamma: the average estimated gamma value
    alpha: the average estimated alpha values 
    mu_constant: the average estimated mu value 
    nu: the average estimated nu values 
    row: the index for the specified emotion 
    
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

    def endo_exo( alpha, gamma, nu,  mu_constant, timestamps, mu_1_sum, t):
        """
        For each video, this function calculate the ratio of exogenous and endogenous influences at each time an event takes place throughout the video.
        
        params:
        alpha: the average estimated alpha values 
        gamma: the average estimated gamma value
        nu: the average estimated nu values 
        mu_constant: the average estimated mu value 
        timestamps: a list where each element is a sequence of time points representing when events occurred for different processes
        mu_1_sum: a matrix of precomputed values that represent the video influence at each time for each emotion
        t: a sequence of time points of event arrival with specified emotion 

        """

        r = decay_multi(timestamps, gamma, alpha, t)                        # obtain the endogenous component

        nu = np.array(nu)
        
        exo  = (nu.reshape(num,1)*mu_1_sum).sum(axis = 0) + mu_constant     # obtain the exogenous component
        endo = r
        
        exo_ratio  = exo/(endo+exo)                                         # obtian the exogenous ratios
        endo_ratio = endo/(endo+exo)                                        # obtian the endogenous ratios

        endo_mean = np.mean(endo_ratio)                                     # average over all times an event takes place
        exo_mean  = np.mean(exo_ratio)
        
        return endo_mean, exo_mean
    
    
    def dist_endo_exo(gamma, alpha, nu, mu_constant, timestamps_all, mu_1_sum_all):
        """
        This function returns the distribution of ratios of exogenous and endogenous influences across all videos in the data sample.
        
        params:
        gamma: the average estimated gamma value
        alpha: the average estimated alpha values 
        nu: the average estimated nu values 
        mu_constant: the average estimated mu value 
        timestamps_all: a list where each element is a sequence of time points for different videos
        mu_1_sum_all: a list of arrays representing the video influence for each emotion in each video

        """
        endo_list = []
        exo_list = []
        
        for i in range(len(timestamps_all)):          # iterate through all videos in the sample
   
            # get the corresponding data
            timestamps = timestamps_all[i]
            mu_1_sum   = np.array(mu_1_sum_all[i][row])
            t          = np.array(timestamps[row])
            endo, exo = endo_exo(alpha, gamma, nu,  mu_constant, timestamps, mu_1_sum, t)  # obtain the average ratio for each video
            
            endo_list.append(endo)   
            exo_list.append(exo)

        return endo_list, exo_list
    
    # get correpsonding parameter for each emotion
    alpha = alpha[row]
    gamma = gamma[row]
    mu_constant = mu_constant[row]
    nu = nu[row]
    
    endo_list, exo_list = dist_endo_exo(gamma, alpha, nu, mu_constant, timestamps_all, mu_1_sum_all)


    return endo_list, exo_list


def hawkes_exo1_exo2(num, 
                     timestamps_all,   
                     mu_1_sum_all, 
                     mu_constant, 
                     nu, 
                     row):
    """
    This function computes the values of spontaneouss and video influence parts of the exogenous influence according to equation (1) in the paper with the mean estimated parameter values. 
    It returns the distribution of the ratio of spontaneous and video influence intensities across all videos.
    
    params:
    num: the number of dimension for the model (number of emotions)
    timestamps_all: a list where each element is a sequence of time points for different videos
    mu_1_sum_all: a list of arrays representing the video influence for each emotion in each video
    mu_constant: the average estimated mu value 
    nu: the average estimated nu values 
    row: the index for the specified emotion 
    
    """


    def exo1_exo2( nu,  mu_constant,  mu_1_sum):
        """
        For each video, this function calculate the ratio of spontaneous and video influence intensities at each time an event takes place throughout the video.
        
        params:
        nu: the average estimated nu values 
        mu_constant: the average estimated mu value 
        mu_1_sum: a matrix of precomputed values that represent the video influence at each time for each emotion

        """

        nu = np.array(nu)
        
        exo_video  = (nu.reshape(num,1)*mu_1_sum).sum(axis = 0) # obtain the video influence intensity
        
        exo_video_r  = exo_video /( mu_constant +  exo_video )    # obtain the ratios of video influence intensity
        exo_0_r = mu_constant/( mu_constant +  exo_video )        # obtain the ratios of spontaneous intensity

        exo_video_r_mean = np.mean(exo_video_r)                   # average over all times an event takes place
        exo_0_r_mean = np.mean(exo_0_r)
    
        
        return exo_0_r_mean, exo_video_r_mean
    

    
    def dist_exo1_exo2( nu, mu_constant, timestamps_all, mu_1_sum_all):
        """
        This function returns the distribution of ratios of spontaneous and video influence intensities across all videos in the data sample.
        
        params:
        nu: the average estimated nu values 
        mu_constant: the average estimated mu value 
        timestamps_all: a list where each element is a sequence of time points for different videos
        mu_1_sum_all: a list of arrays representing the video influence for each emotion in each video

        """
                
        exo_0_list = []
        exo_video_list = []
        
        for i in range(len(timestamps_all)):         # iterate through all videos in the sample
   
            # get the corresponding data
            mu_1_sum = np.array(mu_1_sum_all[i][row])
            exo_0, exo_video = exo1_exo2(nu,  mu_constant, mu_1_sum) # obtain the average ratio for each video
            
            exo_0_list.append(exo_0)
            exo_video_list.append(exo_video)

        return exo_0_list, exo_video_list
    
    # get correpsonding parameter for each emotion
    mu_constant = mu_constant[row]
    nu = nu[row]
    
    exo_0_list, exo_video_list = dist_exo1_exo2( nu, mu_constant, timestamps_all, mu_1_sum_all)

    return exo_0_list, exo_video_list



def plot_endo_exo(ax, endo, exo,  emotions = emotions_plot):
    """
    This function creates violin plots for distributions of influence ratios on given ax object.

    param:
    ax: the ax object to plot figure into
    endo: the distribution of ratios of endogenous influences
    exo: the distribution of ratios of exogenous influences
    emotions: the emotion labels

    """

    # load and prepare data for plotting
    df_endo = pd.DataFrame(endo)
    df_endo = df_endo[emotions]
    
    df_exo = pd.DataFrame(exo)
    df_exo = df_exo[emotions]
    
    df_endo['source'] = 'endo'
    df_exo['source']  = 'exo'
    
    df_combined = pd.concat([df_endo, df_exo], ignore_index=True)                            # concat data for plotting
    df_melted = df_combined.melt(id_vars='source', var_name='emotions', value_name='ratios') # melt dataframe for violin plot with "hue" parameter
    
    # specify color
    sb.set_style("whitegrid", {'axes.grid': True})
    custom_palette = {
    'endo':'#7C9895',
    'exo': '#D9EFD3'
    }
    
    fs = 12
    sb.violinplot(data=df_melted, x="emotions", y="ratios", hue="source",
                   split=True, palette=custom_palette, cut=0, ax = ax, linewidth=1.5)        # create violin plot for paired distributions for each emotion
    
    # remove the outter line for violin plots
    for violin in ax.collections:
        if isinstance(violin, PolyCollection):
            violin.set_edgecolor("face")
            
    # figure formatting
    ax.set_xlabel(' ', fontsize = fs + 6)
    ax.set_ylabel('Ratio', fontsize = fs + 8)

    ax.tick_params(axis='y', labelsize=fs + 8) 
    ax.tick_params(axis='x', labelsize=fs + 8)

    ax.legend(fontsize=fs + 6,loc='upper right', framealpha=0, ncol=2,  bbox_to_anchor=(1.02, 1.04)) 



def plot_exo1_exo2(ax, exo_0 , exo_video,  emotions = emotions_plot):
    """
    This function creates violin plots for distributions of influence ratios on given ax object.

    param:
    ax: the ax object to plot figure into
    exo_0: the distribution of ratios of spontaneous influences
    exo_video: the distribution of ratios of video influences
    emotions: the emotion labels

    """
        
    # load and prepare data for plotting

    df_exo_0 = pd.DataFrame(exo_0)
    df_exo_0 = df_exo_0[emotions]
    
    df_exo_video = pd.DataFrame(exo_video)
    df_exo_video = df_exo_video[emotions]

    df_exo_ratio = df_exo_0/df_exo_video
    print(df_exo_ratio.mean())
    print(np.mean(df_exo_ratio.mean()))
    
    df_exo_0['source']     = 'spontaneous'
    df_exo_video['source'] = 'video influence'
    
    df_exo_combined = pd.concat([df_exo_0, df_exo_video], ignore_index=True)                         # concat data for plotting
    df_exo_melted = df_exo_combined.melt(id_vars='source', var_name='emotions', value_name='ratios') # melt dataframe for violin plot with "hue" parameter

    fs = 12
    # specify color
    sb.set_style("whitegrid", {'axes.grid': True})
    custom_palette = {
    'spontaneous':'#9368A8',
    'video influence': '#E6CECF'
    }

    sb.violinplot(data=df_exo_melted, x="emotions", y="ratios", hue="source",
                   split=True, palette= custom_palette , cut=0, ax =ax , linewidth=1.5)               # create violin plot for paired distributions for each emotion

    # remove the outter line for violin plots
    for violin in ax.collections:
        if isinstance(violin, PolyCollection):
            violin.set_edgecolor("face")
        
    ax.set_xlabel(' ', fontsize = fs + 6)
    ax.set_ylabel(' ', fontsize = fs + 8)

    # figure formatting
    ax.tick_params(axis='y', labelsize=fs + 8) 
    ax.tick_params(axis='x', labelsize=fs + 8)

    ax.legend(fontsize=fs + 6,loc='upper right', framealpha=0, ncol=2,  bbox_to_anchor=(1.02, 1.04))
    

def violin_combined( endo_ratios, exo_ratios, exo_0_ratios, exo_video_ratios, figure = '...', emotions_plot = emotions_plot,):
    """
    This function creates combines the violin plots created by plot_endo_exo and plot_exo1_exo2.

    param:

    figure: the name to store figure as
    emotions_plot: the emotion labels for plotting
    endo_ratios: the distribution of ratios of endogenous influences
    exo_0_ratios: the distribution of ratios of spontaneous influences
    exo_video_ratios: the distribution of ratios of video influence

    """

    nrows = 1                                                          # specify number of rows for subplots
    ncols = 2                                                          # specify number of resulting
    figsize = (10 * ncols, 6* nrows)                                   # specify figure size for each subplot
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey = 'row') # create subplot objects
    axs = axs.flatten()                                                # create axis for subplots

    plot_endo_exo(axs[0], endo = endo_ratios, exo = exo_ratios, emotions = emotions_plot)
    plot_exo1_exo2(axs[1], exo_0 = exo_0_ratios, exo_video = exo_video_ratios, emotions = emotions_plot)
    plt.tight_layout()

    fig.savefig(f'{figure}.pdf', bbox_inches='tight', transparent=True) # store figure with given name

# descriptive plots


def keyword_count(df):
    """
    This function creates a barplot for the number of videos with each of the 27 keywords.

    param:
    df: dataframe with live chats with keyword labels
    """

    keyword_video_count = df.groupby('keyword')['video_id'].nunique().reset_index()    # create video counts for each keyword
    fig, ax = plt.subplots(figsize=(20, 8))                                            # set figure size
    fs = 12
    palette = sb.color_palette("flare")                                                # set palette
    sb.barplot(data=keyword_video_count,  x='keyword', y='video_id', ax=ax, palette=palette) # create barplot for each keyword
    # figure formatting
    ax.tick_params(axis='x', labelsize=fs + 6 , rotation=45, ) 
    ax.tick_params(axis='y', labelsize=fs + 6) 
    ax.set_ylabel('Number of Videos', fontsize = fs + 8)
    ax.set_xlabel(' ', fontsize = fs + 8)
    plt.xticks(rotation=45, ha='right')

    plt.savefig('...', bbox_inches='tight', transparent=True) # store figure to output

def num_events(df):
    """
    This function plots the distribution of the number of events in each video across all videos,
    as well as the number of events in each video for each of the six basic emotions across all videos.

    param:
    df: dataframe with live chats with timestamps
    """
        

    group_sizes    = df.groupby('video_id').size()                # get the number of events for each video
    emotions_count = df.groupby('video_id')[basic_emotions].sum() # get the number of events for each emotion in each video

    fs = 12
    palette  = sb.color_palette("icefire", 7)                     # set the color palette
    fig      = plt.figure(figsize=(12, 4))                        # set figure size
    outer_gs = GridSpec(1, 2, width_ratios=[1, 3], wspace=0.15)   # set figure ratios

    # create histogram plot for the distributio of number of events in each video
    ax1 = fig.add_subplot(outer_gs[0])
    sb.histplot(group_sizes, kde=True, ax=ax1, color=palette[6]) #'rosybrown')
    ax1.set_title('Number of Events per Video', fontsize = fs + 4)
    ax1.set_xlabel('Number of Events', fontsize = fs + 2)
    ax1.set_ylabel('Count', fontsize = fs + 2)
    ax1.tick_params(axis='y', labelsize=fs ) 
    ax1.tick_params(axis='x', labelsize=fs )

    # create nested GridSpec within the second main subplot with adjusted spacing
    inner_gs = outer_gs[1].subgridspec(2, 3, wspace=0.2, hspace=0.4)

    # create the six subplots for each emotion with shared y-axis and 4:3 aspect ratio
    for i in range(2):                                # outer loop iterating over two rows (i = 0, 1)
        for j in range(3):                            # inner loop iterating over three columns (j = 0, 1, 2)
            index = i * 3 + j                         # create index to enumerate through emotions
            if i == 0 and j == 0:
                ax = fig.add_subplot(inner_gs[i, j])  # create the first subplot at position (0, 0) in the grid
                first_ax = ax                         # save the first axis to share y-axis later
            else:
                ax = fig.add_subplot(inner_gs[i, j], sharey=first_ax) # create subsequent subplots and share the y-axis with the first subplot

            sb.histplot(emotions_count[emotions_count.columns[index]], kde=True, ax=ax, color=palette[index]) # create subplot for each emotion
            if j == 1 and i == 0:
                ax.set_title('Number of Events per Video by Emotions', fontsize = fs + 4)  # set title for 6 emotion subplots
            else:
                ax.set_title(' ')

            ax.set_xlabel(emotions_count.columns[index], fontsize = fs + 2)
            ax.tick_params(axis='y', labelsize=fs ) 
            ax.tick_params(axis='x', labelsize=fs )
            ax.set_ylabel('')
            if j != 0:
                ax.tick_params(labelleft=False)         # hide y-axis labels for other subplots

    plt.tight_layout()
    fig.savefig('...', bbox_inches='tight', transparent=True) # store figure
    plt.show()

def event_interval(df):
    """
    This function plots the distribution of the median time interval between subsequent events across all videos,
    as well as the median time interval between subsequent events for each of the six basic emotions across all videos.
   
     param:
    df: dataframe with live chats with timestamps
    """

    df['time_diff'] = df.groupby('video_id')['time_minute'].diff()    # obtain all time intervals between subsequent events in a video
    emotions_interval = df.groupby('video_id')['time_diff'].median()  # obtain the median time interval between subsequent events for each video

    # obtains the median time interval of subsequent events for each of the six basic emotions for each video and store to dictionary
    interval_emotion = {}
    for i in emotions_plot:
        df_e = df[df[f'basic_{i}'] == 1]                                # get live chats of emotion i
        df_e['diff'] = df_e.groupby('video_id')['time_minute'].diff()   # get time interval between subsequent emotions
        diff_median  = df_e.groupby('video_id')['diff'].median().values # get the median time interval for each video
        interval_emotion[i] = diff_median

    fs = 12
    palette = sb.color_palette("deep", 7)                         # set the color palette
    fig      = plt.figure(figsize=(12, 4))                        # set figure size
    outer_gs = GridSpec(1, 2, width_ratios=[1, 3], wspace=0.15)   # set figure ratios

    # create histogram plot for the distributio of number of events in each video
    ax1 = fig.add_subplot(outer_gs[0])
    sb.histplot(emotions_interval, kde=True, ax=ax1, color=palette[6]) 
    ax1.set_title('Event Time Interval per Video', fontsize = fs + 4)
    ax1.set_xlabel('Median Time Interval between Events', fontsize = fs + 2)
    ax1.set_ylabel('Count', fontsize = fs + 2)
    ax1.tick_params(axis='y', labelsize=fs ) 
    ax1.tick_params(axis='x', labelsize=fs )

    # create nested GridSpec within the second main subplot with adjusted spacing
    inner_gs = outer_gs[1].subgridspec(2, 3, wspace=0.2, hspace=0.4)

    # create the six subplots for each emotion with shared y-axis and 4:3 aspect ratio
    for i in range(2):                                                 # outer loop iterating over two rows (i = 0, 1)
        for j in range(3):                                             # inner loop iterating over three columns (j = 0, 1, 2)
            index = i * 3 + j                                          # create index to enumerate through emotions
            if i == 0 and j == 0:
                ax = fig.add_subplot(inner_gs[i, j])                   # create the first subplot at position (0, 0) in the grid
                first_ax = ax                                          # save the first axis to share y-axis later
            else:
                ax = fig.add_subplot(inner_gs[i, j], sharey=first_ax)  # create subsequent subplots and share the y-axis with the first subplot
    

            sb.histplot(interval_emotion[emotions_plot[index]], kde=True, ax=ax, color=palette[index]) # create subplot for each emotion
            
            #ax.set_aspect(4/3)  # Set aspect ratio to 4:3
            if j == 1 and i == 0:
                ax.set_title('Event Time Interval per Video by Emotions', fontsize = fs + 4) # set title for 6 emotion subplots
            else:
                ax.set_title(' ')
            ax.set_xlabel(emotions_plot[index], fontsize = fs + 2)

            ax.tick_params(axis='y', labelsize=fs ) 
            ax.tick_params(axis='x', labelsize=fs )
            ax.set_ylabel('')
            if j != 0:
                ax.tick_params(labelleft=False)                         # hide y-axis labels for other subplots

    plt.tight_layout()
    fig.savefig('...', bbox_inches='tight', transparent=True) # store figure
    plt.show()


def co_occurrence_plot(df):
    """
    This function creates a heatmap plot of the relative co_occurrence frequency of each pair of emotions.

    param:
    df: dataframe with live chats with timestamps
    """
        

    df_cooccur = df[basic_emotions]
    co_occurrence_emotions = df_cooccur.T.dot(df_cooccur)  # compute the co-occurrence matrix by performing a dot product between the transpose of 'df_cooccur' and 'df_cooccur' itself
    # each entry in 'co_occurrence_emotions' will represent the count of how many times two emotions co-occur across the dataset

    # initialize an empty frequency matrix with zeros, having the same dimensions as the co-occurrence matrix
    frequency_matrix = pd.DataFrame(np.zeros((df_cooccur.shape[1], df_cooccur.shape[1])), columns=df_cooccur.columns, index=df_cooccur.columns)
    # loop over each pair of emotions (col1, col2) to calculate the frequency of at least one of the two emotions being present
    for col1 in df_cooccur.columns:
        for col2 in df_cooccur.columns:
            # calculate the frequency where either 'col1' or 'col2' is present in the row
            frequency_matrix.loc[col1, col2] = ((df_cooccur[col1] == 1) | (df_cooccur[col2] == 1)).sum()

    co_occur_normalized = co_occurrence_emotions/frequency_matrix # normalize the co-occurrence matrix by dividing it by the frequency matrix
    # each entry in 'co_occur_normalized' represents the relative frequency of co-occurrence between two emotions

    plt.figure(figsize=(7, 4))                          # set figure size
    fs = 12
    def annot_func(data):
        return np.vectorize(lambda x: f'{x:.1%}')(data) # set figure annotation as percentages to 1 decimal place
    mask = np.triu(np.ones_like(co_occur_normalized, dtype=bool))  # mask upper triangle in the symmetric matrix
    cmap = sb.cubehelix_palette(as_cmap=True)
    sb.heatmap(co_occur_normalized, annot=annot_func(co_occur_normalized), fmt='', cmap=cmap, mask=mask, vmax = 0.2, cbar = False, annot_kws={"size": fs + 1}) # create heatmap plot to visualize the relative frequency of emotion co_occurrence
    plt.tick_params(axis='y', labelsize=fs + 2,  length=0) 
    plt.tick_params(axis='x', labelsize=fs + 2,  length=0)
    plt.tight_layout()
    plt.savefig('...', bbox_inches='tight', transparent=True)

    plt.show()