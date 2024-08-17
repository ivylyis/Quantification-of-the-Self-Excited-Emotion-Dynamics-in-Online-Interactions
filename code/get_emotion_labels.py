from transformers            import TextClassificationPipeline, RobertaForSequenceClassification, RobertaTokenizer

import os
os.environ['CURL_CA_BUNDLE'] = ''
import torch

import pandas as pd

assert torch.cuda.is_available(),                               'GPU not available. Can run on CPU.'

print(torch.cuda.is_available())
device = "cuda:0" if torch.cuda.is_available() else "cpu"

"""
This file contains functions that use pre-trained transformers to obtain 6 basic emotion labels in text.

"""


# load dataset for prediction 
fn  = '/../'

livechat = pd.read_parquet(fn)
livechat = livechat.reset_index()

transcript = pd.read_parquet(fn)
transcript = transcript.reset_index()
transcript['text_lower'] = transcript['text'].str.lower()   # convert to lower case because all-capitalized transcripts are biased towards high arousal emotions

basic_emotions = ['anger',     'disgust',     'fear',      'joy',    'sadness',     'surprise']
emotions_prob =  ['anger_prob','disgust_prob','fear_prob','joy_prob','sadness_prob','surprise_prob']
emotions_pred =  ['anger_pred','disgust_pred','fear_pred','joy_pred','sadness_pred','surprise_pred']


def predict_with_transformer_classifier( df, dir = '/.../', type = 'livechat'):
    """
    Load pre-trained transformers in given directory and predict emotion labels for each text.

    params:
    df:  the dataframe containing text for prediction
    dir: the directory for which pre-trained transformers are stored
    type: whether to predict for live chats for transcripts
    """

    # load data and pre-trained model 
    ####################################################################################################################
    if type == 'livechat':
        text = df['Message'].tolist()                                            # specify column name
    else:
        text = df['text_lower'].tolist()                                         # collect list of texts for prediction

    tokenizer  = RobertaTokenizer.from_pretrained(dir, local_files_only=True)    # load a pre-trained tokenizer from the specified directory
    model      = RobertaForSequenceClassification.from_pretrained(dir)           # load a finetuned sequence classification model from the specified directory

    pipe       = TextClassificationPipeline( model             =  model,         # create a text classification pipeline using the model and tokenizer
                                             tokenizer         =  tokenizer,  
                                             truncation        =  True,          # enable truncation to ensure input sequences fit within the model's maximum length
                                             padding           =  'max_length', 
                                             return_all_scores =  True,          # return all predicted scores for each label
                                             device = 0)
    
    # predict 6 basic emotions 
    ####################################################################################################################
    preds = pipe(text)             # pass the input text through the pipeline

    emotion_scores = []
    for i in preds:
        emotion_scores.append([dic['score'] for dic in i])  # iterate through the predictions and extract the scores for each emotion

    df_text = pd.DataFrame(emotion_scores, columns = emotions_prob)

    if type == 'livechat':
        df_text.index = df['Message_ID']                    # restore the original index of the dataframe
    else:
        df_text.index = df['index']

    df_text.to_parquet('...')

    return df_text


if __name__ == '__main__':

    predict_with_transformer_classifier( df = transcript)
    predict_with_transformer_classifier( df = livechat)




