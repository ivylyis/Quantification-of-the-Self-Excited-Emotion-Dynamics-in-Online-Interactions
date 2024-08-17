import numpy                   as np
import pandas                  as pd
import seaborn                 as sb



"""
This file contains functions that preprocesses the data before encoding video influence and fitting with Hawkes model.

"""


emotions       = ['anger',       'disgust',       'fear',        'joy',       'sadness',       'surprise']
basic_emotions = ['basic_anger', 'basic_disgust', 'basic_fear',  'basic_joy', 'basic_sadness', 'basic_surprise']
emotions_prob  = ['anger_prob',  'disgust_prob',  'fear_prob',   'joy_prob',  'sadness_prob',  'surprise_prob']
emotions_pred  = ['anger_pred',  'disgust_pred',  'fear_pred',   'joy_pred',  'sadness_pred',  'surprise_pred']


def get_pred_label(row):
    """
    This function converts predicted emotion probabilities to binary labels at the 0.5 cutoff.
    The 0.5 cutoff is tested to yeild optimal performance with the testing data.
    """
    prob = row[emotions_prob].values
    y_pred = np.zeros(prob.shape)
    y_pred[np.where(prob >= 0.5)] = 1      # assign emotion label if reaching the 0.5 cutoff
    y_pred = y_pred.astype(int)
    
    return y_pred


def convert_to_labels(df, df_prob, type = 'livechat'):
    """
    This function converts predicted emotion probabilities to binary labels for live chat/transcirpt dataframes.
    The functio takes as input the full dataframe and the dataframe of emotion probabilities.

    params:
    df: the full dataframe with text and additioanl variables
    df_prob: the dataframe of emotion probabilities
    type: indicate whether to process for live char or transcript data
    """ 

    df_prob['labels'] = df_prob.apply(get_pred_label, axis = 1)                    # call function to convert to binary labels
    df_emotion_labels = df_prob['labels'].apply(pd.Series)                         # seperate column of binary indicators for each emotion 
    df_emotion_labels.columns = basic_emotions                                     # assign column names 

    if type == 'livechat':
        df_emotions = df_prob[['Message_ID']].join(df_emotion_labels)              # merge with text identifiers
        df_emotions = df_emotions.merge(df, left_on = 'Message_ID', right_on = 'Message_ID') # merge binary emotion labels with the rest of the data
        df_emotions['time_minute']     = df_emotions['Time_Seconds']/60            # conver time unit to minutes
        df_emotions['duration_minute'] = df_emotions['Duration']/60                # conver time unit to minutes
    else:
        df_emotions = df_prob[['index']].join(df_emotion_labels)                   # merge with text identifiers
        df_emotions = df_emotions.merge(df, left_on = 'index', right_on = 'index') # merge binary emotion labels with the rest of the data
        df_emotions['start_minute'] = df_emotions['start']/60                      # conver time unit to minutes

    df_emotions   = df_emotions[~df_emotions[basic_emotions].eq(0).all(axis=1)]    # remove data with no emotion labels

    df_emotions.to_parquet(f'{type}_emotion.parquet')     

    return df_emotions

def avergae_by_time_bins(df , n_bin = 8):
    """
    This function parse each video into 8 bins of equal length, and calcualtes the rate of events in each bin.
    It appears that, on average, the event arrival rates are stiontionary over the course of a video.
    Subsequently, the coeficient of variation is calculated for each video as the standard deviation across 8 bins over the average.
    Vidoes that have a coefficient of variation over the 80th quantile of the distribution over all videos are removed.

    params:
    df: dataframe of live chats with timestamps
    n_bin: the number of bins to parse each video into

    """
    
    dfs =  df.groupby('video_id')                                            # groupby dataframe by videos 
    freq_video = {}
    
    # pase each video in bins of equal length
    def parse_bins(df):
        total_duration      = df['duration_minute'].unique()[0]              # get the total length for each video
        bin_size = total_duration / n_bin                                    # parse video length into n bins
        bins = [i * bin_size for i in range(n_bin + 1)]                      # get the time range limits for each
        bin_intervals = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]   # return in tuples
        return bin_intervals
    
    # calculate the frequency of events for each bin 
    def calculate_average(df):                                                                                                                       
        bins = parse_bins(df)
        freq_list = []
        for i, b in enumerate(bins):                                                      # iterate through each bin
            upper = b[1]
            lower = b[0]
            df_    = df[(df['time_minute'] >= lower) & (df['time_minute'] < upper )]      # get live chats for each bin
            if i     == len(bins) - 1:
                df_   = df[(df['time_minute'] >= lower) & (df['time_minute'] <= upper )]  # make adjustment for the last bin to include upper bound
            df_ = df_[~df_[basic_emotions].eq(0).all(axis=1)]                             # remove empty rows with no emotion labels
            event_counts = len(df_)                                                       # get the total number of events for each bin
            freq = event_counts/ (upper - lower)                                          # get the rate of events for each bin
            freq_list.append(freq)
        
        return freq_list   
     
    # filters videos by quantiles
    def filter_quantiles(data, lb = 0, ub = 0.8):
        lower_quantile = np.quantile(data, lb)
        upper_quantile = np.quantile(data, ub)
        filtered_data = data[(data >= lower_quantile) & (data <= upper_quantile)]          # get the videos that are below the lower quantiel and above the upper quantile
    
        return filtered_data

    for i, df in dfs:
        freq_video[i] = calculate_average(df)                                              # make calculation for each video
        
    df_freq_video = pd.DataFrame(freq_video).T                  
    cv = df_freq_video.std(axis = 1) / df_freq_video.mean(axis = 1)                        # obtain the coefficient of variation for each video
    cv_filtered = filter_quantiles(cv, 0, 0.80)                                            # get videos above the 80th quantile
    df_filtered = df[df['video_id'].isin(cv_filtered.index)]                               # remove videos above the 80th quantile
    
    return df_filtered



def livehcat_within_transcript(df_livechat, df_transcript ):
    """
    This function restricts livehctas of each video to the time span of transcripts. 
    Transcript start times are shifted forward for 5 seconds to account for reaction and typing time.
    
    params:
    
    df_livechat: the livechat dataframe
    df_transcript: the transcript dataframe
    
    """
    dfs = []
    for i, df in df_livechat.groupby('video_id'):                              # group live chats by videos 
        transcript_video = df_transcript[df_transcript['video_id'] == i]       # get the corresponding transcript for video id
        max_t = transcript_video['start_minute'].max()                         # get time of the last transcript 
        min_t = transcript_video['start_minute'].min()                         # get time of the first transcript 
        df = df[(df['time_minute'] >= min_t) & (df['time_minute'] <= max_t )]  # get live chats within first and last transcript time points
        dfs.append(df)                                                         # collect selected live chats within transcript time for each video
    livechat_within_t = pd.concat(dfs)                                         # concat live chats for all videos
    
    return livechat_within_t                                                   


def filter_videos_by_event_quantiles(df_livechat, lower_quantile = 0.1, upper_quantile = 0.9):
    """
    Filter videos based on the average events per minute for each event type, considering specified quantiles.
    Only includes videos where all event types are within their respective quantile range.

    :param video_data: List of dataframes, each representing data from a video.
    :param lower_quantile: The lower quantile threshold (default 10th percentile).
    :param upper_quantile: The upper quantile threshold (default 90th percentile).
    :return: A dataframe with filtered video averages.
    """

    def calculate_average(df):                                                                       # get the avergae number of videos per minute for each emotion
        total_duration      = df['duration_minute'].unique()[0]                                      # get the total length for each video
        event_counts        = df[basic_emotions].sum()                                               # get the total number of events for each emotion
        return event_counts / total_duration                                                         # get the average number of messages per minute 

    average                 = df_livechat.groupby('video_id').apply(calculate_average)               # get the average number of messages per emotion for each video                                                      

    quantiles = average.quantile([lower_quantile, upper_quantile])                                   # get quantile values
    print(quantiles)
    print(average.describe())

    def is_within_quantiles(row):
        for event_type in average.columns:                                                           # iterate through each emotion
            if not (quantiles.loc[lower_quantile, event_type] < row[event_type] < quantiles.loc[upper_quantile, event_type]): # if any of the emotion lies outside desired quantile range, return False
                return False
        return True                                                                                  # only return true when all emotions lie within desired quantile range

    filtered_videos = average[average.apply(is_within_quantiles, axis = 1)]                          # keep videos where all emotions lie within quantile range
    
    filtered = df_livechat[df_livechat['video_id'].isin(filtered_videos.index)]                      # return videos where number of events per minute lie within quantile for all emotions                                   

    return filtered


def shift_duplicated_timepoints(df_text, type = 'livechat'):
    """
    This function shifts duplicated timepoints for live chat and transcript by a small unit for computational needs.
    The resolution of the data is 1e-3/60, and the shifting unit is 1e-4/60.

    params:
    df_text: the dataframe with text and additioanl variables
    type: indicate whether to process for live char or transcript data
    """ 
        
    dfs = []
    for i, df in df_text.groupby('video_id'):                    # for each video, find duplicated time points
        shift_unit      = 1e-4/60                                # define small unit for shifting (a magnitude smaller than resolution the data)
        column = 'time_minute' if type == 'livechat' else 'start_minute' # specify different column names for live chat and transcript
        duplicates      = df.duplicated(column, keep = 'first')  # keep the fist duplicated timepoint unchanged
        cumcount        = df.groupby(column).cumcount()          # get the order count for multiple duplicated values
        shift_values    = cumcount * shift_unit                 
        df.loc[duplicates, column] += shift_values[duplicates]   # shift by a small unit
        dfs.append(df)
    livechat_shifted   = pd.concat(dfs)

    return livechat_shifted

def adjust_duration(df_text):
    """
    This function adjusts the duration of the video in cases where the duration can be mis-specified.
    The adjusted duration is the maximum between the video duration and the last live chat arrival time (within transcript).

    params:
    df_text: the dataframe with text and additioanl variables
    """ 
        
    dfs = []
    for i, df in df_text.groupby('video_id'):                                           # for each video, adjust the duration upper bound when it's mis-specified.
        df['duration_minute'] = max(df['time_minute'].max(), df['duration_minute'].unique()[0])  # take the maximum between video duration and last live chat arrival time (within transcript) as the adjusted duration
        dfs.append(df)                              
    livechat_shifted = pd.concat(dfs)

    return livechat_shifted
