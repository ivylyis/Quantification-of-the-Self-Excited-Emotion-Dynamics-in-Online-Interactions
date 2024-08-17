Please find the description of each file in the order of execution:

data_fetching_youtube.py
- collects the data and performs basic processing

finetune_transformer.py
- fine-tunes a transformer with training data for multi-label classification of emotions

get_emotion_labels.py
- use the fine-tuned transformer model to obtain emotion labels for each live chat and transcript texts

data_preprocessing.py
- processes the data given emotion labels to compile the final sample for analysis

extract_video_influence.py
- encodes the video influence for each video and extracts the necessary data for calibrating the Hawkes process

fit_preprocess.py
- shapes the data for model fitting and constructs bootstrapped samples

fit_hawkes.py
- calibrate the Hawkes model to estimate parameters for each emotion

plot_figures.py
- generates figures in the paper including result visualizations and descriptive plots