import os
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm



torch.cuda.is_available()

torch.cuda.device_count()
torch.cuda.get_device_name(torch.cuda.current_device())


PATH_TO_DATA = 'D:\\Documents\\ITMO\\Year1\\NLP\\Hw2\\'
data = pd.read_csv(os.path.join(PATH_TO_DATA, 'train.csv'))
data.head()

data = data[~data['question1'].isna()]
data = data[~data['question2'].isna()]

# non-duplicate example
questions = data[data['is_duplicate'] == 0].loc[0]
print(f'Q1: {questions.question1}\nQ2: {questions.question2}')


# duplicate example
questions = data[data['is_duplicate'] == 1].loc[7]
print(f'Q1: {questions.question1}\nQ2: {questions.question2}')


train, test = train_test_split(data, stratify=data['is_duplicate'], random_state=42)



# your code here

print('percentage of duplicates in train = ',np.mean(train['is_duplicate'])*100,'%')
print('percentage of duplicates in test = ',np.mean(test['is_duplicate'])*100,'%')
print('Number of duplicates and non-duplicates in train:',len(train[train['is_duplicate']==1]), 'and',len(train[train['is_duplicate']==0]))
print('Number of duplicates and non-duplicates in test:',len(test[test['is_duplicate']==1]), 'and',len(test[test['is_duplicate']==0])  )


train_idx = train[train['is_duplicate'] == 1].id.tolist()
print(f'Number of training examples: {len(train_idx)}')


q1_train_data = np.array(train.loc[train_idx, 'question1'])
q2_train_data = np.array(train.loc[train_idx, 'question2'])
q1_test_data = np.array(test['question1'])
q2_test_data = np.array(test['question2'])

q1_train = np.empty_like(q1_train_data)
q2_train = np.empty_like(q2_train_data)
q1_test = np.empty_like(q1_test_data)
q2_test = np.empty_like(q2_test_data)

y_test  = np.array(test['is_duplicate'])


q1_train_data[:5]



nltk.download('punkt')

vocab = defaultdict(lambda: 0)
vocab['<PAD>'] = 1

for idx in range(len(q1_train_data)):
    q1_train[idx] = nltk.word_tokenize(q1_train_data[idx])
    q2_train[idx] = nltk.word_tokenize(q2_train_data[idx])
    q = q1_train[idx] + q2_train[idx]
    for word in q:
        if word not in vocab:
            vocab[word] = len(vocab) + 1
print('Vocabulary size is: ', len(vocab))



# processing test

for idx in range(len(q1_test_data)): 
    q1_test[idx] = nltk.word_tokenize(q1_test_data[idx])
    q2_test[idx] = nltk.word_tokenize(q2_test_data[idx])


for i in range(len(q1_train)):
    q1_train[i] = [vocab[word] for word in q1_train[i]]
    q2_train[i] = [vocab[word] for word in q2_train[i]]

        
for i in range(len(q1_test)):
    q1_test[i] = [vocab[word] for word in q1_test[i]]
    q2_test[i] = [vocab[word] for word in q2_test[i]]



q1_train, q1_val, q2_train, q2_val = train_test_split(q1_train, q2_train)


class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, q1, q2):
        self.q1 = q1
        self.q2 = q2

    def __getitem__(self, idx):
        item = (self.q1[idx], self.q2[idx])
        return item

    def __len__(self):
        return len(self.q1)

def iterator(q1, q2, batch_size=128, shuffle=False):
    q1, q2 = data_generator(q1, q2, batch_size, shuffle=shuffle) # padding the sequences to the maximum length amongst the samples in batches
    dataset = PairsDataset(q1, q2)
    return dataset

def data_generator(q1, q2, batch_size, pad=1, shuffle=True):
    """Generator function that yields batches of data

    Args:
        q1 (list): List of transformed (to tensor) questions.
        q2 (list): List of transformed (to tensor) questions.
        batch_size (int): Number of elements per batch.
        pad (int, optional): Pad character defaults to 1.
        shuffle (bool, optional): If the batches should be randomnized or not. Defaults to True.
    Returns:
        tuple: Of the form (input1, input2) with types (numpy.ndarray, numpy.ndarray)
        NOTE: input1: inputs to your model [q1a, q2a, q3a, ...] i.e. (q1a,q1b) are duplicates
              input2: targets to your model [q1b, q2b,q3b, ...] i.e. (q1a,q2i) i!=a are not duplicates
    """
    q1_batch_all = []
    q2_batch_all = []
    idx = 0
    len_q = len(q1)
    question_indexes = [*range(len_q)]
    if shuffle:
        random.shuffle(question_indexes)
    ### START CODE HERE (Replace instances of 'None' with your code) ###  
      # get questions at the `question_indexes` position in q1 and q2

    q1 = q1[question_indexes]
    q2 = q2[question_indexes]

    #q1 = None
    #q2 = None
    batches_num = ceil(len_q / batch_size)
    
    q1_batch_all = [[q1[(i*batch_size):((i+1)*batch_size)]] for i in range(batches_num)]
    q2_batch_all = [[q2[(i*batch_size):((i+1)*batch_size)]] for i in range(batches_num)]
    for i in range(batches_num):
        max_len = np.max([len(arr) for single_batch in q1_batch_all + q2_batch_all for arr in single_batch[0]])
                          
        max_len = 2**int(np.ceil(np.log2(max_len)))
        q1_batch_all[i] = np.asarray([np.asarray(list(map( lambda x : np.asarray(x) if len(x)==max_len else np.asarray( x + [pad]*(max_len-len(x)) ), q1_batch_all[i][0] ))) for _ in range(1)]) # padding the sequences to max_len with pad symbols
        q2_batch_all[i] = np.asarray([np.asarray(list(map( lambda x : np.asarray(x) if len(x)==max_len else np.asarray( x + [pad]*(max_len-len(x)) ), q2_batch_all[i][0] ))) for _ in range(1)]) # padding the sequences to max_len with pad symbols
        q1_batch_all[i] = np.array(q1_batch_all[i])
        q2_batch_all[i] = np.array(q2_batch_all[i])
    return (q1_batch_all, q2_batch_all)


def data_generator(q1, q2, batch_size, pad=1, shuffle=True):
    """Generator function that yields batches of data

    Args:
        q1 (list): List of transformed (to tensor) questions.
        q2 (list): List of transformed (to tensor) questions.
        batch_size (int): Number of elements per batch.
        pad (int, optional): Pad character defaults to 1.
        shuffle (bool, optional): If the batches should be randomnized or not. Defaults to True.
    Returns:
        tuple: Of the form (input1, input2) with types (numpy.ndarray, numpy.ndarray)
         input1: inputs to your model [q1a, q2a, q3a, ...] i.e. (q1a,q1b) are duplicates
              input2: targets to your model [q1b, q2b,q3b, ...] i.e. (q1a,q2i) i!=a are not duplicates
    """
    q1_batch_all = []
    q2_batch_all = []
    idx = 0
    len_q = len(q1)
    question_indexes = [*range(len_q)]
    if shuffle:
        random.shuffle(question_indexes)
    ### START CODE HERE (Replace instances of 'None' with your code) ###  
      # get questions at the `question_indexes` position in q1 and q2

    q1 = q1[question_indexes]
    q2 = q2[question_indexes]

    #q1 = None
    #q2 = None
    batches_num = ceil(len_q / batch_size)
    #print(batches_num)
    q1_batch_all = [q1[(i*batch_size):((i+1)*batch_size)] for i in range(batches_num)]
    q2_batch_all = [q2[(i*batch_size):((i+1)*batch_size)] for i in range(batches_num)]
    #q1_batch_all = [[q1[(i*batch_size):((i+1)*batch_size)]] for i in range(batches_num)]
    #q2_batch_all = [[q2[(i*batch_size):((i+1)*batch_size)]] for i in range(batches_num)]

    q1_batch_all[0][0]

    for i in range(batches_num):
        max_len = max(max([len(q) for q in q1_batch_all[i]]),max([len(q) for q in q2_batch_all[i]]))                  
        max_len = 2**int(np.ceil(np.log2(max_len)))

        b1 = []
        b2 = []
        for q1, q2 in zip(q1_batch_all[i], q2_batch_all[i]):
                # add [pad] to q1 until it reaches max_len
                q1 = q1 + [pad]*(max_len-len(q1))
                # add [pad] to q2 until it reaches max_len
                q2 = q2 + [pad]*(max_len-len(q2))
                # append q1
                b1.append(q1)
                # append q2
                b2.append(q2)
             # padding the sequences to max_len with pad symbols
        q1_batch_all[i] = np.array(b1)
        q2_batch_all[i] = np.array(b2)
    return (q1_batch_all, q2_batch_all)

batch_size = 512
res1, res2 = data_generator(q1_train, q2_train, batch_size)


(res1[0])



class SiameseModel(nn.Module):
    """ Siamese model.

    Args:
        vocab_size (int, optional): Length of the vocabulary. Defaults to len(vocab).
        d_model (int, optional): Depth of the model. Defaults to 128.
        mode (str, optional): 'train', 'eval' or 'predict', predict mode is for fast inference. Defaults to 'train'.

    Returns:
        A PyTorch Siamese model. 
    """
    def __init__(self, vocab_size=len(vocab), d_model=128, hid_size=256, num_layers=2):
        super(SiameseModel, self).__init__()
        # you are free to modify the network and add additional elements
        self.emb =  nn.Embedding(vocab_size, d_model) # defining the embeddings of vocab and d_model size
        self.lstm = nn.LSTM(d_model, hid_size, num_layers = 1) # Defining an LSTM layer
        self.ll = nn.Linear(hid_size, hid_size) # Using dense layer 

    def forward_once(self, q):
        x = self.emb(q) # make all the transformations with input 
        x, _ = self.lstm(x)
        x = self.ll(x)
        x = F.relu(x)
        x = torch.mean(x,axis = 1) # get the mean accros 1 axis
        out = nn.functional.normalize(x,dim=0) # normalize the output
        return out

    def forward(self, q1, q2):
        o1 = self.forward_once(q1)
        o2 = self.forward_once(q2)
        return (o1, o2)



model = SiameseModel()

q1 = torch.tensor(res1[0])
q2 = torch.tensor(res2[0])
model.forward(q1,q2)


class TripletLoss(torch.nn.Module):
    """Custom Loss function.

    Args:
        v1 (torch.tensor): Array with dimension (batch_size, model_dimension) associated to Q1.
        v2 (torch.tensor): Array with dimension (batch_size, model_dimension) associated to Q2.
        margin (torch.tensor, optional): Desired margin. Defaults to 0.25.

    """

    # use torch functions to autograd the barward step
    def forward(self, v1, v2, margin=torch.tensor([0.25])):
        scores =  torch.mm(v1,v2.T)
         # pairwise cosine sim
        batch_size = len(scores) # calculate new batch size
        positive = torch.diagonal(scores) # the positive `diagonal` entries in `scores` (duplicates)
        negative_without_positive = (scores - torch.eye(batch_size)*2.0) # multiply `torch.eye(batch_size)` with 2.0 and subtract it out of `scores`
        closest_negative = (torch.max(negative_without_positive, dim=1)[0]) # take the row by row `max` of `negative_without_positive`
        
        negative_zero_on_duplicate = scores*(1.0 - torch.eye(batch_size)) # subtract `torch.eye(batch_size)` out of 1.0 and do element-wise multiplication with `scores`
        
        mean_negative = torch.sum(negative_zero_on_duplicate, axis=1)/(batch_size-1) # use `torch.sum` on `negative_zero_on_duplicate` for `axis=1` and normalize on batch_size - 1
        
        loss1 = torch.maximum(torch.tensor([0.0]), margin - positive + closest_negative) # subtract `positive` from `margin` and add `closest_negative` and get maximum from this value and 0
        loss2 = torch.maximum(torch.tensor([0.0]), margin - positive + mean_negative) # subtract `positive` from `margin` and add `mean_negative` and get maximum from this value and 0
        
        triplet_loss = torch.mean(torch.add(loss1, loss2))
        return triplet_loss


v1 = torch.tensor(np.array([[ 0.26726124,  0.53452248,  0.80178373],[-0.5178918 , -0.57543534, -0.63297887]]), requires_grad=True)
v2 = torch.tensor(np.array([[0.26726124, 0.53452248, 0.80178373],[0.5178918 , 0.57543534, 0.63297887]]), requires_grad=True)
loss = TripletLoss()
res = loss(v1, v2)
print("Triplet Loss:", res.item()) # expecting 0.5
res.backward()



# you can vary the hyperparams
BATCH_SIZE = 512

train_iter = iterator(
      q1_train,
      q2_train,
      BATCH_SIZE)


loss = TripletLoss()
model = SiameseModel()

# you can vary the hyperparams
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0005)
learning_rate = 0.01
momentum = 0.9
num_epoch = 20




for epoch in tqdm(range(num_epoch)):
    losses = []
    model.train()
    model.zero_grad()
    for i, data in enumerate(train_iter):
        #print(i)
        
        q1, q2 = data
        q1 = torch.tensor(q1)
        q2 = torch.tensor(q2)
        # calculate loss, step the optimizers, save losses
        # your code here
        l1,l2 = model.forward(q1,q2)
        iter_loss = loss(l1,l2)
        iter_loss.backward()
        optimizer.step()

        losses.append(iter_loss.item())
    print(f"Epoch {epoch}\n Current loss {np.mean(losses)}\n")
