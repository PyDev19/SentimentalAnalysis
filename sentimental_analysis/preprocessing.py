import pandas as pd
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import torch

print('Loading data...', end=' ')
data = pd.read_csv('data/text.csv').dropna().drop('Unnamed: 0', axis=1)
data['label'] = data['label'] + 1

reddit_data = pd.read_csv('data/Reddit_Data.csv').dropna()
twitter_data = pd.read_csv('data/Twitter_Data.csv').dropna()

reddit_data.columns = ['text', 'label']
twitter_data.columns = ['text', 'label']

combined_data = pd.concat([reddit_data, twitter_data], ignore_index=True)
combined_data = combined_data[combined_data['label'] == 0]

data = pd.concat([data, combined_data], ignore_index=True)
print('Data loaded')

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    return text

data['text'] = [clean_text(text) for text in tqdm(data['text'], desc='Cleaning text')]

def tokenize(text):
    return text.split()

data['tokens'] = [tokenize(text) for text in tqdm(data['text'], desc='Tokenizing text')]

if os.path.exists('models/vocab.pth'):
    print('Loading vocabulary...', end=' ')
    vocab = torch.load('models/vocab.pth')
    print('Vocabulary loaded')
else:
    import os
    if not os.path.exists('models'):
        os.makedirs('models')

    counter = Counter()
    for tokens in tqdm(data['tokens'], desc='Building vocabulary'):
        counter.update(tokens)

    vocab = {word: i+2 for i, (word, _) in enumerate(counter.items())}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    torch.save(vocab, 'models/vocab.pth')

def encode_tokens(tokens, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

data['input_ids'] = [encode_tokens(tokens, vocab) for tokens in tqdm(data['tokens'], desc='Encoding tokens')]

MAX_LEN = 128
def pad_sequence(seq, max_len):
    if len(seq) < max_len:
        seq += [vocab['<PAD>']] * (max_len - len(seq))
    return seq[:max_len]

data['input_ids'] = [pad_sequence(seq, MAX_LEN) for seq in tqdm(data['input_ids'], desc='Padding sequences')]

print('Encoding labels...', end=' ')

import os
if not os.path.exists('models'):
    os.makedirs('models')

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])
torch.save(label_encoder, 'models/label_encoder.pth')

print('Labels encoded')

print('Saving preprocessed data...', end=' ')
data.to_csv('data/preprocessed_data.csv', index=False)
print('Data saved')