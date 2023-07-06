'''
Data Engineering
'''

'''
D01. Import Libraries for Data Engineering
'''
import re
import numpy as np
from sklearn.model_selection import train_test_split    
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata

print("Tensorflow version {}".format(tf.__version__))
import random
SEED = 1234
tf.random.set_seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE

'''
D02. Import AG News Dataset from Auther's gdrive
'''
# AG-News dataset download from Auther's Github repository

# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo/Bible_2_07_Multi_class_Classification.git
# ! git pull origin master
! git pull origin main

!unzip "/content/AG_News/AG_news.zip" -d "/content/"

from IPython.display import clear_output 
clear_output()


'''
D03. [PASS] Tokenizer Install & import
''' 
# Keras Tokenizer is a tokenizer provided by default in tensorflow 2.X and is a word level tokenizer. It does not require a separate installation.

'''
D04. Define Hyperparameters for Data Engineering
'''
max_len = 120  # cut texts after this number of words (among top vocab_size most common words)

'''
D05. Load and modifiy to pandas dataframe
'''

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

train_df.rename(columns = {'Class Index':'Label'}, inplace = True)
test_df.rename(columns = {'Class Index':'Label'}, inplace = True)

train_df = train_df.drop(columns=['Title', 'Description'])
test_df  = test_df.drop(columns=['Title', 'Description'])


'''
D06. [PASS] Delete duplicated data
'''

'''
D07. [PASS] Select samples
'''

'''
D08. Preprocess and build list
'''
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

'''
D17. Split Data
'''

train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=32)

train_df['Label'] = train_df['Label'].astype(int)
Y_train = train_df["Label"].to_numpy() - 1

valid_df['Label'] = valid_df['Label'].astype(int)
Y_valid = valid_df["Label"].to_numpy() - 1

test_df['Label'] = test_df['Label'].astype(int)
Y_test = test_df["Label"].to_numpy() - 1

'''
D09. Add <SOS>, <EOS> for source and target
'''

train_document = train_df['document']
valid_document = valid_df['document']
test_document  = test_df['document']

print(train_document[:10])

train_sentence  = train_document.apply(lambda x: "<SOS> " + str(x))
valid_sentence  = valid_document.apply(lambda x: "<SOS> " + str(x))

'''
D10. Define tokenizer
'''

filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'

SRC_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)

SRC_tokenizer.fit_on_texts(train_sentence)

vocab_size = len(SRC_tokenizer.word_index) + 1

print('Word set size of Encoder :',vocab_size)

'''
D11. Tokenizer test
'''

lines = [
  "It is winter and the weather is very cold.",
  "Will this Christmas be a white Christmas?",
  "Be careful not to catch a cold in winter and have a happy new year."
]
for line in lines:
    txt_2_ids = SRC_tokenizer.texts_to_sequences([line])
    ids_2_txt = SRC_tokenizer.sequences_to_texts(txt_2_ids)
    print("Input     :", line)
    print("txt_2_ids :", txt_2_ids)
    print("ids_2_txt :", ids_2_txt[0],"\n")

'''
D12. Tokenize
'''
# tokenize / encode integers / add start and end tokens / padding
train_tkn_inputs = SRC_tokenizer.texts_to_sequences(train_sentence)
valid_tkn_inputs = SRC_tokenizer.texts_to_sequences(valid_sentence)
test_tkn_inputs  = SRC_tokenizer.texts_to_sequences(test_document)

'''
D13. [EDA] Explore the tokenized datasets
'''

len_result = [len(s) for s in train_tkn_inputs]

print('Maximum length of review : {}'.format(np.max(len_result)))
print('Average length of review : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

'''
D14. Pad sequences
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train = pad_sequences(train_tkn_inputs,  maxlen=max_len, padding='post', truncating='post')
X_valid = pad_sequences(valid_tkn_inputs,  maxlen=max_len, padding='post', truncating='post')
X_test  = pad_sequences(test_tkn_inputs,   maxlen=max_len, padding='post', truncating='post')

'''
D15. Data type define
'''
X_train = tf.cast(X_train, dtype=tf.int64)
X_valid = tf.cast(X_valid, dtype=tf.int64)
X_test  = tf.cast(X_test, dtype=tf.int64)

'''
D16. [EDA] Explore the Tokenized datasets
'''
print('Training input(shape)   :', X_train.shape)
print('Training output(len)    :', len(Y_train))
print('Validation input(shape) :', X_valid.shape)
print('Validation output(len)  :', len(Y_valid))
print('Testing input(shape)    :', X_test.shape)
print('Testing output(len)     :', len(Y_test))

# 0번째 샘플을 임의로 출력
print(X_train[0])
print(X_valid[0])
print(X_test[0])

'''
D17. Split Data
'''
# Split was done at the previous step

'''
D18. [PASS] Build dataset
'''
# For eager mode, it is done at the "model.fit"

'''
D19. [PASS] Define some useful parameters for further use
'''

'''
Model Engineering
'''

'''
M01. Import Libraries for Model Engineering
'''

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import load_model

'''
M02. [PASS] TPU Initialization
'''

'''
M03. Define Hyperparameters for Model Engineering
'''
embedding_dim = 256
hidden_size = 128
output_dim = 4  # output layer dimensionality = num_classes
EPOCHS = 20
batch_size = 100
learning_rate = 5e-4

'''
M04. [PASS] Open "strategy.scope(  )"
'''

'''
M05. Build NN model
'''
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(GRU(hidden_size))
model.add(Dense(output_dim, activation='softmax'))

'''
M06. Optimizer
'''
optimizer = optimizers.Adam(learning_rate=learning_rate)

'''
M07. Model Compilation - model.compile
'''
model.compile(optimizer=optimizer, loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()

'''
M08. EarlyStopping
'''
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 8)

'''
M09. ModelCheckpoint
'''
mc = ModelCheckpoint('best_model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)

'''
M10. Train and Validation - `model.fit`
'''
history = model.fit(X_train, Y_train, epochs = EPOCHS,
                    batch_size=batch_size,
                    validation_data = (X_valid, Y_valid),
                    verbose=1,
                    callbacks=[es, mc])

'''
M11. Assess model performance
'''
loaded_model = load_model('best_model.h5')
print("\n Test Accuracy: %.4f" % (loaded_model.evaluate(X_test, Y_test)[1]))

'''
M12. [Opt] Plot Loss and Accuracy
'''
history_dict = history.history
history_dict.keys()

acc      = history_dict['accuracy']
val_acc  = history_dict['val_accuracy']
loss     = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'o', color='g', label='Training loss')   # 'bo'
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.plot(epochs, acc, 'o', color='g', label='Training acc')   # 'bo'
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

'''
M13. [Opt] Training result test for Code Engineering
'''

txt = ["Beer Brewer Buying	Mergers and acquisitions in the beer industry have been hot this year. Expect it to get hotter."]
seq = SRC_tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_len)
pred = model.predict(padded)
labels = ['World', 'Sports', 'Business', 'Sci_Tech']
# “World”, “Sports”, “Business”, “Sci_Tech”
print(pred, labels[np.argmax(pred)])
    
