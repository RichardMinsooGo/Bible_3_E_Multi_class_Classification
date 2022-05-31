!pip install -U torchtext==0.10.0

import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch.nn.functional as F

import random
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split    

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from IPython.display import clear_output 
clear_output()

# AG-News dataset download from Auther's Github repository

# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo/Bible_2_07_Multi_class_Classification.git
# ! git pull origin master
! git pull origin main

!unzip "/content/AG_News/AG_news.zip" -d "/content/"

import re
from sklearn.model_selection import train_test_split    
import unicodedata

# 1. Tokenizer Install & import
# Keras Tokenizer는 tensorflow 2.X 에서 기본으로 제공하는 tokenizer이며, word level tokenizer이다. 이는 별도의 설치가 필요 없다.

# 2. Copy or load raw data to Colab
BUFFER_SIZE = 20000

import shutil
import pandas as pd

pd.set_option('display.max_colwidth', 100)
# pd.set_option('display.max_colwidth', None)

train_df = pd.read_csv('/content/train.csv')
test_df = pd.read_csv('/content/test.csv')

print(len(train_df))
print(len(test_df))

train_df.head()

train_df["document"] = train_df["Title"] + ' ' + train_df["Description"]
test_df["document"] = test_df["Title"] + ' ' + test_df["Description"]

train_df.rename(columns = {'Class Index':'label'}, inplace = True)
test_df.rename(columns = {'Class Index':'label'}, inplace = True)

train_df = train_df.drop(columns=['Title', 'Description'])
test_df  = test_df.drop(columns=['Title', 'Description'])

# 5. Preprocess and build list

def preprocess_func(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()  
    return sentence

train_df['document'] = train_df['document'].apply(preprocess_func)
test_df['document']  = test_df['document'].apply(preprocess_func)

train_df["label"]= train_df["label"].apply(str)
train_df["label"] = train_df["label"].apply(lambda x: x.replace("1", "World").replace("2", "Sports").replace("3", "Business").replace("4", "Sci_Tech"))

test_df["label"]= test_df["label"].apply(str)
test_df["label"] = test_df["label"].apply(lambda x: x.replace("1", "World").replace("2", "Sports").replace("3", "Business").replace("4", "Sci_Tech"))

column_names = ["document", "label"]
train_df = train_df.reindex(columns=column_names)
test_df  = test_df.reindex(columns=column_names)

train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=32)

print(len(train_df))
print(len(valid_df))
print(len(test_df))
print(train_df.shape)
print(valid_df.shape)
print(test_df.shape)


TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm')

# TEXT = data.Field(sequential=True, use_vocab=True, tokenize=tokenizer.morphs, lower=False, batch_first=True, fix_length=20)
LABEL = data.LabelField()

def convert_dataset(input_data, text, label):
    list_of_example = [data.Example.fromlist(row.tolist(), fields=[('text', text), ('label', label)])  for _, row in input_data.iterrows()]
    dataset = data.Dataset(examples=list_of_example, fields=[('text', text), ('label', label)])
    return dataset

train_data = convert_dataset(train_df,TEXT,LABEL)
valid_data = convert_dataset(valid_df, TEXT, LABEL)
test_data = convert_dataset(test_df, TEXT, LABEL)

vars(train_data[-1])

print(f'Number of training examples   : {len(train_data)}')
print(f'Number of validation examples : {len(valid_data)}')
print(f'Number of testing examples    : {len(test_data)}')

MAX_VOCAB_SIZE = 20000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)

LABEL.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary : {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

print(TEXT.vocab.freqs.most_common(20))

print(TEXT.vocab.itos[:10])

print(LABEL.vocab.stoi)

BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = data.Iterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    sort = False,
    device = device)

print('Number of minibatch for training dataset   : {}'.format(len(train_iterator)))
print('Number of minibatch for validation dataset : {}'.format(len(valid_iterator)))
print('Number of minibatch for testing dataset    : {}'.format(len(test_iterator)))


class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, embedding_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        
        #text = [sent len, batch size]
        
        text = text.permute(1, 0)
                
        #text = [batch size, sent len]
        embedded = self.dropout(self.embedding(text))
        output, _ = self.rnn(embedded)
        output = self.linear(output[:, -1, :])
        return output
  
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
    
# Dimension shall be the same with Glove Vector
EMBEDDING_DIM = 100
OUTPUT_DIM = len(LABEL.vocab)
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = LSTM(len(TEXT.vocab), 128, OUTPUT_DIM, EMBEDDING_DIM, DROPOUT)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        # We initialize the gradient to 0 for every batch.
        optimizer.zero_grad()

        # batch of sentences인 batch.text를 model에 입력
        predictions = model(batch.text)
        
        loss = criterion(predictions, batch.label)
        
        acc = categorical_accuracy(predictions, batch.label)
        
        # backward()를 사용하여 역전파 수행
        loss.backward()

        # 최적화 알고리즘을 사용하여 parameter를 update
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    # "evaluation mode" : turn off "dropout" or "batch nomalizaation"
    model.eval()

    # Use less memory and speed up computation by preventing gradients from being computed in pytorch
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text)
            
            loss = criterion(predictions, batch.label)
            
            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut4-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

model.load_state_dict(torch.load('tut4-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

import torch
model.load_state_dict(torch.load('tut4-model.pt'))

import spacy
nlp = spacy.load('en_core_web_sm')

def predict_sentiment(model, sentence, min_len = 5):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = model(tensor)
    max_preds = prediction.argmax(dim = 1)
    return max_preds.item()

pred_class = predict_sentiment(model, "Beer Brewer Buying Mergers and acquisitions in the beer industry have been hot this year. Expect it to get hotter.")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')

pred_class = predict_sentiment(model, "Eyeing the next wave in RISC computing Some critics say RISC's time has passed. Sun Microsystems' David Yen has another idea.")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')

pred_class = predict_sentiment(model, "Entertainment to go Reviewer examines three categories of portable audio players, and adds video to the mix.")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')

pred_class = predict_sentiment(model, "Sony introduces pro high-definition video system Company will ship \$4,900 camcorder and \$3,700 digital recorder in February.")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')


