

import numpy as np
import os
import sys
import tarfile
import codecs

from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle



# Download UD1.3 data sets

url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1699/ud-treebanks-v1.3.tgz?sequence=1&isAllowed=y"

def maybe_download(filename, force=False):
  """Download a file if not present."""
  if force or not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  return filename

data = maybe_download('ud-treebanks-v1.3.tgz')


# Unzip .tar and load data


num_classes = 54
np.random.seed(133)

def maybe_extract(filename, num_expect, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  root = root + '.3'
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_expect:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
data_folders = maybe_extract(data, num_classes)



train_data_files = []
test_data_files = []
#val_data_files = []
for folder in data_folders:
    for file in os.listdir(folder):
        if file.endswith('ud-train.conllu'):
            train_data_files.append(folder + '/' + file)
        if file.endswith('ud-test.conllu'):
            test_data_files.append(folder + '/' + file)
        #if file.endswith('ud-dev.conllu'):
            #train_data_files.append(folder + '/' + file)

#override list for test purpose
#train_data_files = ['ud-treebanks-v1.3/UD_Danish/da-ud-train.conllu']
            
def load_data_from_file(file_name):
    print('trying to load ' + file_name)
    
    current_words = []
    current_tags = []
    skipped_lines = 0
    
    for i, line in enumerate(codecs.open(file_name, encoding='utf-8')):
        line = line.strip()
        
        if line:
            if len(line.split('\t'))<4:
                if line.startswith('#'): #skip comments
                    exit
                else:
                    #Error
                    if skipped_lines<2:
                        print("Skipping erroneous line: {} (line number: {}) ".format(line, i), file=sys.stderr)
                    skipped_lines +=1
                    exit
            else:
                word, tag = line.split('\t')[1],line.split('\t')[3]
                current_words.append(word)
                current_tags.append(tag)
        
        else:
            if current_words: #skip emtpy lines
                #print('current words ' + str(current_words))
                yield (current_words, current_tags)
            current_words = []
            current_tags = []
            
    # check for last one
    if current_tags != []:
        yield (current_words, current_tags)
    
    print('skipped ' + str(skipped_lines) + ' in ' + file_name)
        

print(train_data_files)



# Load training datasets into Python


task_ids = []

def get_train_data(list_folders_name):
        """
        :param list_folders_name: list of folders names
        :param lower: whether to lowercase tokens
        transform training data to features (word indices)
        map tags to integers
        """
        X = []
        Y = []
        task_labels = [] #keeps track of where instances come from "task1" or "task2"..
        tasks_ids = [] #record the id of the tasks

        #num_sentences=0
        #num_tokens=0

        # word 2 indices and tag 2 indices
        w2i = {} # word to index
        c2i = {} # char to index
        task2tag2idx = {} # id of the task -> tag2idx

        w2i["_UNK"] = 0  # unk word / OOV
        c2i["_UNK"] = 0  # unk char
        c2i["<w>"] = 1   # word start
        c2i["</w>"] = 2  # word end index
        
        
        for i, folder_name in enumerate( list_folders_name ):
            num_sentences=0
            num_tokens=0
            task_id = 'task'+str(i)
            tasks_ids.append( task_id )
            if task_id not in task2tag2idx:
                task2tag2idx[task_id] = {}
            for instance_idx, (words, tags) in enumerate(load_data_from_file(folder_name)):
                num_sentences += 1
                instance_word_indices = [] #sequence of word indices
                instance_char_indices = [] #sequence of char indices 
                instance_tags_indices = [] #sequence of tag indices

                for i, (word, tag) in enumerate(zip(words, tags)):
                    num_tokens += 1

                    # map words and tags to indices
                    if word not in w2i:
                        w2i[word] = len(w2i)
                    instance_word_indices.append(w2i[word])

                    chars_of_word = [c2i["<w>"]]
                    for char in word:
                        if char not in c2i:
                            c2i[char] = len(c2i)
                        chars_of_word.append(c2i[char])
                    chars_of_word.append(c2i["</w>"])
                    instance_char_indices.append(chars_of_word)
                            
                    if tag not in task2tag2idx[task_id]:
                        #tag2idx[tag]=len(tag2idx)
                        task2tag2idx[task_id][tag]=len(task2tag2idx[task_id])

                    instance_tags_indices.append(task2tag2idx[task_id].get(tag))

                X.append((instance_word_indices, instance_char_indices)) # list of word indices, for every word list of char indices
                Y.append(instance_tags_indices)
                task_labels.append(task_id)

            #self.num_labels[task_id] = len( task2tag2idx[task_id] )

            if num_sentences == 0 or num_tokens == 0:
                sys.exit( "No data read from: "+folder_name )
            print("TASK "+task_id+" "+folder_name, file=sys.stderr )
            print("%s sentences %s tokens" % (num_sentences, num_tokens), file=sys.stderr)
            print("%s w features, %s c features " % (len(w2i),len(c2i)), file=sys.stderr)

        assert(len(X)==len(Y))
        return X, Y, task_labels, w2i, c2i, task2tag2idx #sequence of features, sequence of labels, necessary mappings

train_X, train_Y, task_labels, w2i, c2i, task2t2i = get_train_data(train_data_files)


# Pickle training data


pickle_file = 'bilstmTraining.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_X': train_X,
    'train_Y': train_Y,
    'task_labels': task_labels,
    'w2i': w2i,
    'c2i': c2i,
    'task2t2i': task2t2i,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise


# Load test data


task2tag2idx = {}

def set_indices(task2t2i):
        for task_id in task2t2i:
            task2tag2idx[task_id] = task2t2i[task_id]
        
def get_features(words):
        """
        from a list of words, return the word and word char indices
        """
        word_indices = []
        word_char_indices = []
        for word in words:
            if word in w2i:
                word_indices.append(w2i[word])
            else:
                word_indices.append(w2i["_UNK"])
                
            chars_of_word = [c2i["<w>"]]
            for char in word:
                if char in c2i:
                    chars_of_word.append(c2i[char])
                else:
                    chars_of_word.append(c2i["_UNK"])
            chars_of_word.append(c2i["</w>"])
            word_char_indices.append(chars_of_word)
        return word_indices, word_char_indices

def get_data_as_indices(folder_name, task):
    """
    X = list of (word_indices, word_char_indices)
    Y = list of tag indices
    """
    X, Y = [],[]
    org_X, org_Y = [], []
    task_labels = []
    for (words, tags) in load_data_from_file(folder_name):
        word_indices, word_char_indices = get_features(words)
        tag_indices = [task2tag2idx[task].get(tag) for tag in tags]
        X.append((word_indices,word_char_indices))
        Y.append(tag_indices)
        org_X.append(words)
        org_Y.append(tags)
        task_labels.append( task )
    return X, Y, org_X, org_Y, task_labels

test_X, test_Y, org_X, org_Y, test_task_labels = [],[],[],[],[]

set_indices(task2t2i)

for i, file in enumerate(test_data_files):
    X, Y, oX, oY, tlabels = get_data_as_indices(file, 'task'+str(i))
    test_X.append([X])
    test_Y.append([Y])
    org_X.append([oX])
    org_Y.append([oY])
    test_task_labels.append([tlabels])
    



# Pickle test data


pickle_file = 'bilstmTest.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'test_X': test_X,
    'test_Y': test_Y,
    'org_X': org_X,
    'org_Y': org_Y,
    'test_task_labels': test_task_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise


# Load word embeddings


def load_embeddings_file(file_name, sep=" ",lower=False):
    """
    load embeddings file
    """
    emb={}
    for line in codecs.open(file_name, encoding='utf-8'):
        fields = line.split(sep)
        vec = [float(x) for x in fields[1:]]
        word = fields[0]
        if lower:
            word = word.lower()
        emb[word] = vec

    print("loaded pre-trained embeddings (word->emb_vec) size: {} (lower: {})".format(len(emb.keys()), lower))
    return emb, len(emb[word])

word_embed = {}
num = 0

os.listdir('/srv/embeds/poly_a')

for file in os.listdir('/srv/embeds/poly_a'):
    emb, i = load_embeddings_file('/srv/embeds/poly_a/' + file)
    word_embed.update(emb)
    num = num + i

# Pickle embeddings

pickle_file = 'bilstmEmbed.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'word_embed': word_embed,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise




