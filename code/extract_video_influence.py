import pickle
import numpy                   as np
import pandas                  as pd

from scipy                     import interpolate
from scipy.integrate           import quad
from scipy.stats               import lognorm
from tqdm                      import tqdm
from multiprocessing           import Pool


"""

This file contains functions that encodes the video influence of specified emotion for each video.
Re-run this file for each of the 6 basic emotions.

"""


emotions = ['...']                                # give columns for 6 basic emotions

event = '...'                                     # calculate the video influence of specified emotion
num_processes = 40                                # specify the number of cores for parallelization

sigma_value = np.sqrt(np.log(10) - np.log(2))     # define sigma and mu parameters for the lognormal function
mu_value    = np.log(10) 
s     = sigma_value                               # shape parameter (sigma)
scale = np.exp(mu_value)                          # scale parameter (exp(mu))

def calculate_contributions_lognormal(args):
    """
    This function calculates the influence of prior transcirpts given time.
    For each time point in the video, the total video influecen is the cumulative influence froma all prior live chats.

    params:
    df_t: dataframe with transcript texts and timestamps
    norm: normalization constant to make the maximum lognormal function value 1
    func: the lognormal function
    time_point: the time in video with milisecond precision in units of in minutes
    """
    
    df_t, norm, func, time_point = args              # take function arguments

    t_k       = df_t['start_minute'].values[:, np.newaxis] # obtain transcript arrival timepoints
    t_diff    = (time_point - t_k)*60                # convert minutes to seconds
    tolerance = 1e-10                                # set tolerance parameter for time differences 
    
    pdf_val = func.pdf(t_diff)                       # obtain the log-normal function value given time since transcript appeats
    pdf_val = pdf_val/norm                           # normalize function values 

    contributions = np.where((t_diff > 0) | np.isclose(t_diff, 0, atol=tolerance), pdf_val, 0) # only inlcude transcript influences that took place prior to each time point

    return np.sum(contributions, axis=0)             # sum up influences from all prior transcripts

def ems_func_parallel(df_t, s, scale, time_points, event, num_processes):
    """
    This function calculates the influence of prior transcript in parallel across the duration of the video in units of miliseconds.

    params:
    df_t: dataframe with transcript texts and timestamps
    s: the pre-defined shape parameter of lognormal function
    scale: the pre-defined scale parameter of lognormal function
    time_point: fine-grid timepoints from the start of the video to end with milisecond precision in units of in minutes
    event: calculate video influence of given emotion
    num_processes: the number of processes in multiprocess

    """
        
    start_time  = pd.Timestamp('1970-01-01')
    minute_time = start_time                        # define start time ith benchmark date
    end = minute_time + pd.Timedelta(seconds=100)   # for each transcript, evaluate influence up to 100 seconds after apperance

    time_ranges = pd.date_range(start=minute_time, end=end, freq='L') # time grid for the evaluation period in units of miliseconds
    df_test     = pd.DataFrame(time_ranges)                           # return in dataframe

    t_list = (df_test[0] - start_time).dt.total_seconds()             # return timepoints in units of seconds
    
    func = lognorm(s=s, scale=scale)                                  # define function with predefined parameters 

    pdf_values = func.pdf(t_list)                                     # obtain function values over evaluatio period
    norm = np.max(pdf_values)                                         # obtain normalization constant such that the maximum function value is 1

    id_ = df_t['video_id'].unique()                                   # obtain video id
    df_t = df_t[df_t[event] == 1]                                     # restrict to transcripts with the specific emotion

    pool = Pool(processes=num_processes)                              # initaize multiprocess instance
    args_list = [(df_t, norm, func, tp) for tp in time_points]        # define function arguments
    total_values = pool.map(calculate_contributions_lognormal, args_list) # calculate video influence in paralell across the duration of the video
    pool.close()
    pool.join()
    total_values_df = pd.DataFrame(total_values, columns=id_)         # return values in dataframe

    return  total_values_df



def calculate_video_influence(livechat_df, df_transcript, s, scale, event,  num_processes):
    """
    This function calculates the video influence at each point in time of given emotion for each video.
    It also extracts the live chat event times, the length of the video, and calculates the integral for video influence function for subsequent fitting.

    params:
    livechat_df: dataframe with live chat texts and timestamps
    df_transcript: dataframe with transcript texts and timestamps
    s: the pre-defined shape parameter of lognormal function
    scale: the pre-defined scale parameter of lognormal function
    event: calculate video influence of given emotion
    num_processes: the number of processes in multiprocess

    """

    event_all = []
    mu_all    = []
    end_time  = []
    integrals = []
    base_time = pd.Timestamp('1970-01-01')                                        # set benchmark date
    for i, df in tqdm(df_transcript.groupby('video_id')):                         # iterature through all videos

        mu            = {}
        event_times   = {}
        df_livechat   = livechat_df[livechat_df['video_id'].isin(df['video_id'])]    # obtain live chats for each video
        endtime       = df_livechat['duration_minute'].unique()[0]                   # get the video duration for each video in units of minutes
        end_time.append(endtime)                   

        df['DateTime'] = df['start_minute'].apply(lambda x: base_time + pd.Timedelta(minutes=x))     # convert to datatime objects
        date_range = pd.date_range(start=df['DateTime'].min(), end=df['DateTime'].max(), freq='L')   # obtain the observation period for each video between the first and last live chat 
        new_df     = pd.DataFrame(date_range, columns=['millisecond'])                               # obtian time points of video length in units milisecond precisions
        reference_time    = base_time  
        new_df['minutes'] = (new_df['millisecond'] - reference_time).dt.total_seconds() / 60         # convert time points to units of minutes
        time_vals_sample  =  new_df['minutes'].values
        
        result_df = ems_func_parallel(df, s, scale, time_vals_sample, event,  num_processes)         # calcualte the video influence in each point in time in parallel
        func_interpolate            = interpolate.interp1d( time_vals_sample , result_df[i], kind = 'linear', bounds_error = False, fill_value = 1e-10)  # generate interpolation function
        integral,err                = quad(func_interpolate, 0, endtime)                             # obtain integral of the interpolated function of video influence
        integrals.append(integral)  

        for emotion in emotions:                                                                  # given interpolatio function of video influence, obtain influence for events of different emotions
            livechats = df_livechat[[emotion, 'time_minute']]                                     # get relevant columns for live chats
            livechats = livechats[livechats[emotion] == 1]                                        # restrict to live chats with specified emotion 
            event_minutes        = livechats['time_minute'].tolist()                              # get livechat arrival times point in units of minutes
            event_times[emotion] = np.array(event_minutes)                                        # extract event arrival times
            mu_vals         = func_interpolate(event_minutes)                                     # obtain video influence of specified emotion on all events
            mu_vals[mu_vals <= 0] = 1e-10                                                         # set negative values, if any, to quasi zero
            mu[emotion]     = mu_vals                                                             # obtain video influence
        event_times         = list(event_times.values())                                          # collect event ariivla times in list
        mu                  = list(mu.values())                                                   # collect baseline intensities in list
        mu_all.append(mu)
        event_all.append(event_times)

    # store values
    with open('event_times.pkl', 'wb') as file:
        pickle.dump(event_all, file)

    with open('end_time.pkl', 'wb') as file:
        pickle.dump( end_time, file)

    with open('mu_joy.pkl', 'wb') as file:
        pickle.dump( mu_all, file)

    with open('integrals_joy.pkl', 'wb') as file:
        pickle.dump( integrals, file)








