
# coding: utf-8

# In[21]:

import numpy as np

from six.moves import cPickle as pickle
from keras.layers import Input, Embedding, LSTM, Dense, merge, Bidirectional
from keras.models import Model
from keras.preprocessing import sequence


# Load data from pickle files

#Load pickle files
training_objects = pickle.load(open('bilstmTraining.pickle', 'rb'))
test_objects = pickle.load(open('bilstmTest.pickle', 'rb'))
embed_objects = pickle.load(open('bilstmEmbed.pickle', 'rb'))

#Unpack training objects
train_X = training_objects['train_X']
train_Y = training_objects['train_Y']
w2i = training_objects['w2i']
c2i = training_objects['c2i']
task2t2i = training_objects['task2t2i']

#Unpack test objects
test_X = test_objects['test_X']
test_Y = test_objects['test_Y']
org_X = test_objects['org_X']
org_Y = test_objects['org_Y']
test_task_labels = test_objects['test_task_labels']

#Unpack Word Embeddings
word_embed = embed_objects['word_embed']


# Transform to nympy arrays and embed matrix
train_X = np.asarray(train_X)
train_Y = np.asarray(train_Y)
test_X = np.asarray(test_X)
test_Y = np.asarray(test_Y)


print('X_train shape:', train_X.shape)
print('X_test shape:', test_X.shape)


#Build embeddings matrix for all known words
embedding_matrix = np.zeros((len(w2i) + 1, 64))
for i, word in enumerate(w2i.keys()):
    embedding_vector = word_embed.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# Build Keras BiLSTM model

#Placeholder inputs
char_input = Input(shape=(128,64), dtype='float32')
word_input = Input(shape=(128,), dtype='int32')

#Embeddings
embed_layer = Embedding(len(w2i) + 1,
                    64,
                    weights=[embedding_matrix],
                    input_length=128,
                    trainable=False)(word_input)

#Apply LSTM on charachters
char_biLSTM = Bidirectional(LSTM(64))(char_input)

#Concatenate char and word embedings
merge_l1 = merge([embedded, char_biLSTM], mode='concat', concat_axis=0)

#Apply LSTM on concated inputs
biLSTM = Bidirectional(LSTM(64))(merge_l1)

after_dp = Dropout(0.5)(biLSTM)
output = Dense(17, activation='sigmoid')(after_dp)

model = Model([char_input,word_input], output)




