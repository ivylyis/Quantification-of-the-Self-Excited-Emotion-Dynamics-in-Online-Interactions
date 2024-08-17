import isodate

import numpy                   as np
import pandas                  as pd

from datetime                  import timedelta
from googleapiclient.discovery import build
from youtube_transcript_api    import YouTubeTranscriptApi as yta
from chat_downloader           import ChatDownloader
from langdetect                import detect

"""
This file contains functions that are used to collect YouTube live chats, transcropts, and video statistics.
Data is fetech through the YouTube API

"""

youtube = build('youtube', 'v3', developerKey='key')     # input YouTube API credentials


def get_videos_info(search_keyword, max_total_results=1000):
    """
    This function fetches video IDs given keyword with the keyword search function.
    For each keyword, we collect all available videos with the next_page_token which goes through pages of the search result.
    Around 500 videos ids are collected for each keyword.

    params:
    search_keyword: the keyword for video search e.g. entertainment
    max_total_results: the maximum number of videos to fetch for each keyword

    """


    video_data = []                                     # create list to collect video information
    next_page_token = None                              # start with empty value and fetch the next_page_token for each iteration
    total_results = 0                                   # number of results counter
    max_results_per_request = 50                        # API restrction on the maxumum result fetched per request

    while total_results < max_total_results:            # keep searching for more videos if below the upper limit

        search_response = youtube.search().list(        # specify search criterias and restrictions
            q=search_keyword,                           # search with specified keyword
            part='snippet',                     
            videoCaption='closedCaption',               # restrict to videos with subtitles
            type='video',      
            eventType = 'completed',                    # restrict to completed live videos
            relevanceLanguage='en',                     # testrict to videos in the English language
            maxResults = min(max_results_per_request, max_total_results - total_results),
            pageToken  = next_page_token                # fetch the next page token for next iteration
        ).execute()

        video_ids = [item['id']['videoId'] for item in search_response.get('items', [])] # extract video IDs from returned information
        total_results += len(video_ids)                                                  # count the number of videos fetched

        videos_response = youtube.videos().list(         # get video detaisl for each fetched video ID
            part='contentDetails',
            id=','.join(video_ids)
        ).execute()

        # Append video data (ID and Duration)
        for item in videos_response.get('items', []):    # extract relevant information for each video (ID and Duration)
            video_data.append({                          
                'Video ID': item['id'],                  # extract video id
                'Duration': isodate.parse_duration(item['contentDetails']['duration']).total_seconds() # extract video duration
            })

        # Prepare for next iteration

        next_page_token = search_response.get('nextPageToken') # extrac the next page token to iterate through search results

        if not next_page_token:
            break

    df = pd.DataFrame(video_data)                        # return all video IDs, and duration in a DataFrame

    return df


def get_videos_info_all_keywords(keywords, max_total_results = 10000):
    """
    This function iterates through predefiend list of keywords and fetches video IDs for each keyword.
    For each keyword, we collect all available videos with the next_page_token which goes through pages of the search result.
    Around 500 videos IDs are collected for each keyword.

    params:
    search_keyword: the list of keywords for video search
    max_total_results: the maximum number of videos to fetch for each keyword

    """
    for i in keywords:    # iterate through the list of keywords
        df_videos_ids = get_videos_info( i, max_total_results = max_total_results) # get video IDs for each keyword
        df_videos_ids['keyword'] = i                                               # add column specifying the keyword
        df_videos_ids.to_parquet(f'df_videos_ids_{i}.parquet')                     # save dataframe
                    

def fetch_transcript(video_id):
    """
    This function iterates through a given dataframe with video IDs, and fetch the transcript for each video.
    The package yta.get_transcript[1] is used for trnacript fetching.
    params:

    video_id: fetch trancript for the given video ID

    [1] https://pypi.org/project/youtube-transcript-api/
    
    """
        

    transcript = yta.get_transcript(video_id)             # fetch the video transcript for given video ID
    transcript  = pd.DataFrame(transcript)                # return transcript in dataframe where each row contains the text and timestamp for each line of transcript
    transcript['video_id'] = video_id                     # specify video id
    
    return transcript


def fetch_livechat_transcript_filter(df, lb = 1, ub = 300):
    """
    This function iterates through a given dataframe with video ids and,
    for each video that has live chat replay function,
    fetches the timestampted live chat messages as well as the video transcript (subtitles).
    Transcripts are fetched with function fetch_transcript.
    Live chats are fetch with package ChatDownloader[1].
    To make sure that the live chats are arriving at a readible speed, we only keep videos that have a median live chat arrival time difference between lb and ub limits.
    The function uses try and except to skip videos that are missing either the live chat or the transcript.
    The function returns one dataframe for live chats across all videos and one for transcript across all videos, both with video IDs.
    The time column  in both dataframes corresponds to the time in the video a live chat/subtitle appears.
    
    params:

    df: dataframe containing youtube video IDs
    lb: the lower bound of inter-livechat duration for filtering
    ub: the upper bound of inter-livechat duration for filtering

    [1] https://github.com/xenova/chat-downloader
    
    """
    df        = df.drop_duplicates()
    video_ids = df['Video ID'].to_list()                                                                # get unique video IDs
    length    = df['Duration'].to_list()                                                                
    keyword   = df['keyword'].to_list()
    livechats = [] 
    transcripts = []
    
    for i, id_ in enumerate(video_ids):                                                                 # iterate through all videos 
        try:
            mid_point  = length[i]/2                                                                    # get video length mid-point
            start_time = str(timedelta(seconds=int(mid_point)))                                         # convert to hh:mm:ss format
            test_duation = min(600, 1/10*length[i])                                                     # get 10th of total length of video for filtering, set maximum at 10 minutes for computational efficiency
            end_time     = str(timedelta(seconds=int(mid_point + test_duation)))                        # get time point for end of test duration 

            url  = f'https://www.youtube.com/watch?v={id_}'                                             # input format of video url
            chat = ChatDownloader().get_chat(url = url, start_time= start_time , end_time= end_time)    # fetch chats for given interval

            timestamps = []


            for message in chat:
                time_seconds  = message.get('time_in_seconds')                                          # collect the timestamps for each live chat arrival for filtering
                timestamps.append(time_seconds)

            intervals_seconds = np.diff(timestamps)                                                     # get live chat arrival time intervals 

            median_interval_seconds = np.median(intervals_seconds)                                      # get median of chat intervals during test period

            
            if  lb <= median_interval_seconds <= ub:                                                    # only keep video if the chat median interval time is between lower and upper limits


                messages = []
                chat = ChatDownloader().get_chat(url)                                                    # create a generator to fetch live chats


                for message in chat:                                                                     # iterate over messages to get relevant information
                    timestamp = message['timestamp']                                                     # get timestamp for each live chat message
                    author = message.get('author', {}).get('name')
                    message_text = message.get('message')                                                # get the text of the live chat message
                    message_time = message.get('time_text')                                              # indicates time in video that the given message appears
                    message_time_seconds = message.get('time_in_seconds')
                    message_id = message.get('message_id')                                               # get unique identifier for each live chat message
                    message_author_id = message.get('author', {}).get('id')
                    message_author_name = message.get('author', {}).get('name')
                    messages.append([message_id, timestamp, author, message_text, message_time, message_time_seconds, message_author_id, message_author_name])

                livechat = pd.DataFrame(messages, columns=['Message_ID', 'Timestamp', 'Author', 'Message', 'Time', 'Time_Seconds', 'Message_Author_ID', 'Message_Author_Name'])    # returns results in dataframe
                livechat['video_id'] = id_
                livechat['chat_volumn'] = len(messages)                                                  # get the number of live chats for each video
                livechat['keyword'] = keyword[i]
                livechat['Duration'] = df[df['Video ID'] == id_]['Duration'].values[0]

                try:
                    transcript = fetch_transcript(id_)                                                    # for a given video with live chat replay, fetch the video transcript
                    transcripts.append(transcript)                                                        # collect all transcripts for given video
                    livechats.append(livechat)                                                            # collect all live chats for given video

                except Exception as e:                                                                    # return error if encountered
                    print(f"{e}: {id_}")
                    pass

        except Exception as e:                                                                            # return error if encountered
            print(f"{e}: {id_}")
            pass

    df_livechat = pd.concat(livechats)
    df_transcript = pd.concat(transcripts)
    
    df_livechat.to_parquet(f'test_livechat_{keyword[0]}.parquet')
    df_transcript.to_parquet(f'test_transcript_{keyword[0]}.parquet')
    
    return df_livechat, df_transcript

"""
Collect and store data for given list of keyword   

"""

# define a list of keywords for video search
keywords = ['entertainment', 'sports','news','politics','law', 'tech', 'education', 'gaming',
            'lifestyle', 'fitness','travel', 'food', 'trial', 'review', 'tutorial', 'podcast',
            'interview', 'music','comedy', 'vlog', 'film', 'documentary', 'auction', 'live',
            'Q&A', 'behind the scenes', 'unboxing', 'haul']


# collect and store live chat and transcript for all videos

def collect_and_store_data(path, keywords = keywords, max_total_results = 10000, lb = 1, ub = 300):
    """
    This wrapper function iterates through a list of keywords to fetch the video ids for each keyword and subsequently th elive chats and transcripts for each video.

    params:

    path: path for stored files
    keywords: the list of keywords to fetch data for
    max_total_results: the maximum number of videos to fetch for each keyword
    lb: the lower bound of inter-livechat duration for filtering
    ub: the upper bound of inter-livechat duration for filtering
    
    """
    get_videos_info_all_keywords(keywords, max_total_results = max_total_results)                           # collect and store video IDs for all keywords
    dfs = {}
    for i in keywords:
        dfs[i] = pd.read_parquet(path + f'df_videos_ids_{i}.parquet')
    df_concat  = pd.concat(dfs.values(), ignore_index=True)

    df_concat_ = df_concat.drop_duplicates(subset=['Video ID', 'Duration'], keep='first')                   # remove duplicated video ids

    grouped = df_concat_.groupby('keyword')
    df_by_keyword = {name: group for name, group in grouped}

    for i in keywords:
        df_livechat_, df_transcript_ = fetch_livechat_transcript_filter(df_by_keyword[i], lb = lb, ub = ub) # fetch and store live chat and transcript for all videos

    dfs_livechat   = {} 
    dfs_transcript = {}
    for i in keywords:                                                                                      # collect all live chat and transcript for each keyword
        dfs_livechat[i]   = pd.read_parquet(path + f'livechat_{i}.parquet')
        dfs_transcript[i] = pd.read_parquet(path + f'transcript_{i}.parquet')

    df_livechat   = pd.concat(dfs_livechat.values(), ignore_index=True)
    df_transcript = pd.concat(dfs_transcript.values(), ignore_index=True)


    columns = ['Message_ID', 'Timestamp', 'Author', 'Message', 'Time', 'Time_Seconds',
        'Message_Author_ID', 'Message_Author_Name', 'video_id', 'chat_volumn']
    df_livechat_ = df_livechat.drop_duplicates(subset=columns, keep='first')                                # drop duplicated entries in live chats    

    df_transcript_ = df_transcript.drop_duplicates()                                                        # drop duplicated entries in transcripts

    df_livechat_filtered   = df_livechat_[df_livechat_['video_id'].isin(df_transcript_['video_id'])]        # remove videos that don't have both transcript and live chat
    df_transcript_filtered = df_transcript_[df_transcript_['video_id'].isin(df_livechat_['video_id'])]

    df_livechat_filtered.to_parquet('...')
    df_transcript_filtered.to_parquet('...')
    df_concat.to_read_parquet('...')

    return df_livechat_filtered, df_transcript_filtered


"""
Remove non-English text.

"""

def is_english(text):
    """
    This function detects whether the input text is in English.

    params:
    text: input text

    """
    try:
        return detect(str(text)) == 'en'
    except:
        return False
    

    
def filter_non_english_livechat(df_livechat, df_transcript):
    """
    This function removes videos with a significant portion of non-English live chats.
    Videos with a share of non-English live chats tha exceeds the 90th percentile of the distribution across all videos are removed.
    For the remaining videos, non-English live chats are removed.

    params:
    df_livechat: dataframe of live chats to be filtered
    df_transcript: dataframe of transcripts to be filtered

    """

    df_livechat['is_english'] = df_livechat['Message'].apply(is_english)              # get true or false labels for each live chat text
    non_english = df_livechat[df_livechat['is_english'] == False]                     # get all non-english livechats
    df_livechat_non_english = df_livechat[df_livechat['video_id'].isin(non_english['video_id'])]  # get all videos that have non-English livechats
    total_count_per_id = df_livechat_non_english.groupby('video_id').size()           # get the number of total live chats per video
    false_count_per_id = non_english.groupby('video_id').size()                       # get the number of non-english live chats per video
    false_ratio_per_id = false_count_per_id / total_count_per_id                      # get the ratio of on-english live chats per video
    false_ratio_per_id = pd.DataFrame(false_ratio_per_id, columns = ['non_english_ratio']) # return in dataframe

    false_ratio_per_id_disgard = false_ratio_per_id[false_ratio_per_id['non_english_ratio'] >= false_ratio_per_id['non_english_ratio'].quantile(0.9)] # get videos with a non-english ratio higher than the 90th quantile of the distribution of all videos
    df_livechat_  = df_livechat[~df_livechat['video_id'].isin(false_ratio_per_id_disgard.index)]           # remove videos with high ratios of non-English live chats
    df_livechat_keep   = df_livechat_[df_livechat_['is_english']]                                          # for the remaining videos, remove non-English live chats
    df_transcript_keep = df_transcript[df_transcript['video_id'].isin(df_livechat_keep['video_id'])]       # remove the corresponding videos in the transcript dataset

    return df_livechat_keep, df_transcript_keep


