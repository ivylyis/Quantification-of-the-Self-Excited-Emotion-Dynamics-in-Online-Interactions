from transformers import Trainer, TrainingArguments, EvalPrediction
from transformers import RobertaTokenizer,  RobertaForSequenceClassification

import os
os.environ['CURL_CA_BUNDLE'] = ''
import torch

import pandas as pd
import numpy  as np

from torch.utils.data import DataLoader
from sklearn.metrics  import f1_score, roc_auc_score, accuracy_score,jaccard_score

print(torch.cuda.is_available())
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('DEVICE: ', device)

"""
This file contains functions that finetunes a transformer for multilabel classification of 6 basic emotions.
The Roberta[1] model is used to handle emojis.
Training data is from the Semeval dataset[2]

[1] https://huggingface.co/docs/transformers/en/model_doc/roberta
[2] Mohammad, Saif, et al. "Semeval-2018 task 1: Affect in tweets." Proceedings of the 12th international workshop on semantic evaluation. 2018.
"""

basic_emotions = ['anger',     'disgust',     'fear',      'joy',    'sadness',     'surprise']
emotions_prob =  ['anger_prob','disgust_prob','fear_prob','joy_prob','sadness_prob','surprise_prob']
emotions_pred =  ['anger_pred','disgust_pred','fear_pred','joy_pred','sadness_pred','surprise_pred']

# define emotion mapping pairs

emotion_mapping = {
    'anger': 'anger',
    'anticipation': 'joy',
    'disgust': 'disgust',
    'fear': 'fear',
    'joy': 'joy',
    'love': 'joy',
    'optimism': 'joy',
    'pessimism': 'sadness',
    'sadness': 'sadness',
    'surprise': 'surprise',
    'trust': 'joy'
}


def map_emotions(row, emotion_mapping, additional_columns):
    """
    This function maps the original 11 emotions to 6 basic emotions accoding to the emotion_mapping dictionary.
    The mapped dataframe keeps the text content and identifiers.

    params:
    row: input text
    emotion_mapping: emotion mapping dictionary 
    additional_columns: columns to be kept after mapping
    """
        

    result = {col: row[col] for col in additional_columns}                     # start with the additional columns to keep
    
    basic_emotions = set(emotion_mapping.values())                             # get the set of basic emotions
    for emotion in basic_emotions:
        result[emotion] = 0                                                    # initialize the basic emotions in the result with 0

    for emotion, basic_emotion in emotion_mapping.items():                     # map specific emotions to basic emotions 
        if row[emotion] == 1:                                                  # indicate the presence of basic emotions if any sub-emotions are present
            result[basic_emotion] = 1
            
    return pd.Series(result)



dir  = '/.../'
fn1  = f'{dir}2018-E-c-En-train.txt'                                           # load training and evaluatio dataset from the Semeval dataset
fn2  = f'{dir}2018-E-c-En-dev.txt'
train_dataset = pd.read_csv(fn1, delimiter = "\t")                             # read as dataframe
eval_dataset  = pd.read_csv(fn2, delimiter = "\t")

rename = {'ID':'id', 'Tweet':'text'}           
train_dataset = train_dataset.rename(columns = rename)
eval_dataset  = eval_dataset.rename(columns = rename)

additional_columns = ['id','text']                                             # define columns to be kept after mapping

basic_emotions_train = train_dataset.apply(map_emotions, axis=1, args=(emotion_mapping, additional_columns)) # map training data
basic_emotions_eval  = eval_dataset.apply(map_emotions, axis=1, args=(emotion_mapping, additional_columns))  # map evaluation data

train_dataset['labels'] = basic_emotions_train[basic_emotions].values.tolist()  # get emotion labels in pre-defined order
eval_dataset['labels']  = basic_emotions_eval[basic_emotions].values.tolist()

train_dataset = train_dataset[['id','text', 'labels']].reset_index(drop=True)   # get relevanr columns
eval_dataset  = eval_dataset[['id','text', 'labels']].reset_index(drop=True)

# finetune transformer for multilable classification

class Data_Processing(object):
    def __init__(self, tokenizer, id_column, text_column, label_column):
        self.text_column  = text_column.tolist()            # define the text column from the dataframe
        self.label_column = label_column                    # define the label column and transform to list
        self.id_column    = id_column.tolist()              # define the id column and transform to list

    def __getitem__(self, index):                           # iteratively get text element and tokenize
        comment_text = str(self.text_column[index])
        comment_text = " ".join(comment_text.split())
        
        inputs = tokenizer.encode_plus(comment_text,               # encode the text sequence and add padding
                                       add_special_tokens=True,    # add special tokens (e.g., [CLS] and [SEP]) as required by the model
                                       max_length=512,             # set the maximum length for the sequence 
                                       padding='max_length',       # pad sequences to the maximum length
                                       return_attention_mask=True, # return an attention mask to differentiate between padding and actual tokens
                                       truncation=True,            # truncate sequences that exceed the maximum length
                                       return_tensors='pt')        # return the tokenized inputs as pytorch tensors
        input_ids = inputs['input_ids']                            # extract the input ids tensor from the tokenized inputs
        attention_mask = inputs['attention_mask']                  # extract the attention mask tensor from the tokenized inputs

        labels_ = torch.tensor(self.label_column[index], dtype=torch.float  )   # convert the label for this sample to a pytorch tensor with float dtype
        id_     = self.id_column[index]                                         # retrieve the id for this sample from the id column
        return {'input_ids': input_ids[0], 'attention_mask': attention_mask[0], # return a dictionary containing the input ids, attention mask, labels, and id for this sample
                'labels': labels_, 'id_': id_}

    def __len__(self):
        return len(self.text_column)                               # get length of data                    


# load roberta tokenizer
tokenizer = RobertaTokenizer.from_pretrained('/.../', local_files_only=True)

# toenize data
training_data = Data_Processing(tokenizer,
                                train_dataset['id'],
                                train_dataset['text'],
                                train_dataset['labels'])

eval_data = Data_Processing(tokenizer,
                            eval_dataset['id'],
                            eval_dataset['text'],
                            eval_dataset['labels'])

# use the dataloaders class to load the training and evaluation data into batches
# load training data in batches, shuffle data, use 4 workers
dataloaders_dict = {'train': DataLoader(training_data, batch_size=len(training_data), shuffle=True, num_workers=4),
                    'val': DataLoader(eval_data, batch_size=len(eval_data), shuffle=True, num_workers=4)
                    }

# get the size of the training and validation datasets
dataset_sizes = {'train': len(training_data),
                 'val': len(eval_data)
                 }

# use gpu if available, otherwise fall back to cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the pre-trained roberta model for sequence classification, the number of output labels = number of basic emotions, set device to gpu
model = RobertaForSequenceClassification.from_pretrained('/.../', local_files_only=True, num_labels = len(basic_emotions)).to("cuda")

# define a function to calculate the accuracy of predictions for each row 
def pairwise_accuracy(row):
    pred  = row['pred']                       # extract predicted labels
    label = row['labels']                     # extract true labels
    accuracy = accuracy_score(pred, label)    # calculate accuracy score
    return accuracy

# define a function to compute multi-label classification evaluation metrics
def multi_label_metrics(
    predictions,
    labels,
    ):
    sigmoid = torch.nn.Sigmoid()                   # initialize sigmoid activation function
    probs   = sigmoid(torch.Tensor(predictions))   # apply sigmoid to predictions to get probabilities 
    y_pred  = np.zeros(probs.shape)                # initialize prediction matrix with zeros
    y_true  = labels                               # assign true labels

    y_pred[np.where(probs >= 0.5)] = 1             # convert probabilities to binary labels at the 0.5 cutoff

    # convert predictions, probabilities, and labels into dataframes for evaluation
    df_prob   = pd.DataFrame(probs, columns = emotions_prob)
    df_pred   = pd.DataFrame(y_pred, columns = emotions_pred)
    df_labels = pd.DataFrame(y_true, columns = basic_emotions)

    # concatenate these dataframes along with the original id and text columns from the evaluation dataset
    df_eval =  pd.concat([df_prob, df_pred, df_labels], axis = 1)
    df_eval[['id', 'text']] = eval_dataset[['id', 'text']]
    df_eval['prob']         = df_eval[emotions_prob].values.tolist()
    df_eval['pred']         = df_eval[emotions_pred].values.tolist()
    df_eval['labels']       = df_eval[basic_emotions].values.tolist()
    df_eval['accuracy']     = df_eval.apply(pairwise_accuracy, axis = 1)  # apply the pairwise_accuracy function to calculate accuracy scores

    # calculate various performance metrics for multi-label classification
    f1_micro_average   = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average   = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_samples_average = f1_score(y_true=y_true, y_pred=y_pred, average='samples')
    roc_auc_micro = roc_auc_score(y_true, y_pred, average = 'micro')
    roc_auc_macro = roc_auc_score(y_true, y_pred, average = 'macro')

    jaccard_micro   = jaccard_score(y_true, y_pred, average = 'micro')
    jaccard_macro   = jaccard_score(y_true, y_pred, average = 'macro')
    jaccard_samples = jaccard_score(y_true, y_pred, average = 'samples')
    accuracy = accuracy_score(y_true, y_pred)

    # store and return all calculated metrics in a dictionary  
    metrics = {'f1_micro': f1_micro_average,
               'f1_macro': f1_macro_average,
               'f1_samples': f1_samples_average,
               'auc_micro': roc_auc_micro,
               'auc_macro':roc_auc_macro,
               'jaccard_micro': jaccard_micro,
               'jaccard_macro': jaccard_macro,
               'jaccard_samples': jaccard_samples,
               'accuracy': accuracy}
    return metrics



# compute evaluation metrics
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,  # extract predictions
            tuple) else p.predictions
    result = multi_label_metrics(                          # compute multi-label metrics using the predictions and true labels
        predictions=preds,
        labels=p.label_ids)
    return result

# define the training arguments and hyperparameters
dir = '/.../'
training_args  = TrainingArguments(
    output_dir = dir,                           # directory where the model checkpoints will be saved
    num_train_epochs = 5,                       # number of training epochs
    per_device_train_batch_size = 8,            # batch size for training
    gradient_accumulation_steps = 1,            # number of gradient accumulation steps
    per_device_eval_batch_size  = 8,            # batch size for evaluation
    evaluation_strategy   = "epoch",            # evaluate the model at the end of each epoch
    metric_for_best_model = 'jaccard_samples',  # metric to determine the best model during training; the Jaccard index is a commonly used evaluation metric in multi-label classification tasks
    save_strategy         = 'epoch',            # save model at the end of each epoch
    disable_tqdm = False,                       # enable progress bars
    load_best_model_at_end=True,                # load the best model at the end of training
    warmup_steps  = 1500,                       # number of warmup steps for learning rate scheduler
    learning_rate = 2e-5,                       # learning rate for training
    weight_decay  = 0.01,                       # weight decay for regularization
    logging_steps = 8,                          # log every 8 steps
    fp16 = False, 
    logging_dir=f'{dir}logs/', 
    dataloader_num_workers = 0, 
    run_name = 'multilabel_classification'
)
# instantiate the trainer class and check for available devices
trainer = Trainer(
    model = model,                      # laod the pre-trained roberta model
    args  = training_args,              # use training arguments defined above
    train_dataset   = training_data,    # training dataset
    eval_dataset    = eval_data,        # evaluation dataset
    compute_metrics = compute_metrics,  # call function to compute evaluation metrics
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'    

trainer.train()
trainer.save_model( dir )  # save the best model

