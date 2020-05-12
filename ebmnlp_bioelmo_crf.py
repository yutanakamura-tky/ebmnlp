#!/usr/bin/env python
# coding: utf-8

# # 0. Preparation

# ### 0-1. Dependencies

# In[1]:


import logging
import os, random
import re
from pathlib import Path
from glob import glob
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, precision_recall_fscore_support

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.optim as optim

import subprocess
import shlex

from sklearn.model_selection import train_test_split
from seqeval.metrics import accuracy_score, f1_score, classification_report

from tqdm import tqdm_notebook as tqdm
import pytorch_lightning as pl


# In[2]:


from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG


def create_logger(exp_version):
    log_file = ("{}.log".format(exp_version))

    # logger
    logger_ = getLogger(exp_version)
    logger_.setLevel(DEBUG)

    # formatter
    fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    # file handler
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    fh.setFormatter(fmr)

    # stream handler
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger_.addHandler(fh)
    logger_.addHandler(ch)


def get_logger(exp_version):
    return getLogger(exp_version)


# In[3]:


VERSION = 'ver0.1' # 実験番号
create_logger(VERSION)
get_logger(VERSION).info('This is a message')


# ### 0-2. Prepare files

# In[4]:


DIR_DOC_1 = Path('ebm_nlp_1_00/documents')
DIR_LABEL_1 = Path('ebm_nlp_1_00/annotations')


# In[5]:


text_file_all = sorted(glob(str(DIR_DOC_1 / '*.text')))
token_file_all = sorted(glob(str(DIR_DOC_1 / '*.tokens')))


# In[6]:


p_file_train = sorted(glob(str(DIR_LABEL_1 / 'aggregated/starting_spans/participants/train/*.ann')))
i_file_train = sorted(glob(str(DIR_LABEL_1 / 'aggregated/starting_spans/interventions/train/*.ann')))
o_file_train = sorted(glob(str(DIR_LABEL_1 / 'aggregated/starting_spans/outcomes/train/*.ann')))


# In[7]:


idx_all = [re.compile(r'/([0-9]+)[^0-9]+').findall(path)[0] for path in text_file_all]
idx_train = [re.compile(r'/([0-9]+)[^0-9]+').findall(path)[0] for path in p_file_train]
idx_test = [idx for idx in idx_all if idx not in idx_train]


# In[8]:


text_file_train = sorted([str(DIR_DOC_1 / f'{idx}.text') for idx in idx_train])
text_file_test = sorted([str(DIR_DOC_1 / f'{idx}.text') for idx in idx_test])
token_file_train = sorted([str(DIR_DOC_1 / f'{idx}.tokens') for idx in idx_train])
token_file_test = sorted([str(DIR_DOC_1 / f'{idx}.tokens') for idx in idx_test])


# In[9]:


len(idx_train)


# In[10]:


len(idx_test)


# In[11]:


texts = []
tokens = []
p = []
i = []
o = []

for file in tqdm(text_file_train):
    with open(file) as f:
        texts.append(f.read())

for file in tqdm(token_file_train):
    with open(file) as f:
        tokens.append(f.read().split())

for file in tqdm(p_file_train):
    with open(file) as f:
        p.append(f.read())        

for file in tqdm(i_file_train):
    with open(file) as f:
        i.append(f.read())        

for file in tqdm(o_file_train):
    with open(file) as f:
        o.append(f.read())        
    


# ### 0-3. NER tag

# In[12]:


# 'BOS' and 'EOS' are not needed to be explicitly included
ID_TO_LABEL = {0: 'O', 1: 'I-P', 2: 'B-P', 3: 'I-I', 4: 'B-I', 5: 'I-O', 6: 'B-O'}
LABEL_TO_ID = {v:k for k, v in ID_TO_LABEL.items()}


# In[13]:


def integrate_pio(p, i, o):
    """
    str, str, str -> torch.tensor
        p (str): IO tagging for Patient entity (e.g., '0,0,1,1,0,0,...')
        i (str): IO tagging for Intervention entity (e.g., '0,0,1,1,0,0,...')
        o (str): IO tagging for Outcome entity (e.g., '0,0,1,1,0,0,...')
    """
    
    # sequence of 0 (O-tag) or 2 (B-P-tag) or 1 (I-P-tag)
    # e.g., '2,1,0,2,1,0,...'
    p = re.sub('1', f"{LABEL_TO_ID['I-P']}", p)
    p = re.sub(f"^{LABEL_TO_ID['I-P']}", f"{LABEL_TO_ID['B-P']}", p)
    p = re.sub(f"{LABEL_TO_ID['O']},{LABEL_TO_ID['I-P']}", f"{LABEL_TO_ID['O']},{LABEL_TO_ID['B-P']}", p)
        
    # sequence of 0 (O-tag) or 4 (B-I-tag) or 3 (I-I-tag)
    # e.g., '4,3,0,4,3,0,...'
    i = re.sub('1', f"{LABEL_TO_ID['I-I']}", i)
    i = re.sub(f"^{LABEL_TO_ID['I-I']}", f"{LABEL_TO_ID['B-I']}", i)
    i = re.sub(f"{LABEL_TO_ID['O']},{LABEL_TO_ID['I-I']}", f"{LABEL_TO_ID['O']},{LABEL_TO_ID['B-I']}", i)

    # sequence of 0 (O-tag) or 6 (B-O-tag) or 5 (I-O-tag)
    # e.g., '6,5,0,6,5,0,...'
    o = re.sub('1', f"{LABEL_TO_ID['I-O']}", o)
    o = re.sub(f"^{LABEL_TO_ID['I-O']}", f"{LABEL_TO_ID['B-O']}", o)
    o = re.sub(f"{LABEL_TO_ID['O']},{LABEL_TO_ID['I-O']}", f"{LABEL_TO_ID['O']},{LABEL_TO_ID['B-O']}", o)
       
    # integrated P, I and O tags        
    lp = [int(x) for x in p.split(',')]
    li = [int(x) for x in i.split(',')]
    lo = [int(x) for x in o.split(',')]
    tag = torch.tensor(np.max(np.vstack([lp,li,lo]), axis=0))
    
    return tag


# In[14]:


tags = [integrate_pio(p,i,o) for p,i,o in zip(p,i,o)]


# In[15]:




# ### 0-4. Check document lengths

# In[16]:


document_lengths = []

for file in tqdm(token_file_all):
    with open(file) as f:
        document_lengths.append(len(f.read().split()))


# In[17]:




# In[18]:


MAX_LENGTH = 1024


# # 1. BioELMo

# In[19]:


from allennlp.modules.elmo import Elmo, batch_to_ids


# In[20]:




# - options.json: CNN, BiLSTMモデルの形状を規定
# - bioelmo: hdf5ファイル. CNN, BiLSTMモデルの重みを格納している

# In[21]:


DIR_ELMo = Path('/home/nakamura/bioelmo')


# In[22]:


bioelmo = Elmo(DIR_ELMo / 'options.json', DIR_ELMo / 'bioelmo', 1, requires_grad=False, dropout=0)


# # 2. CRF

# In[ ]:


from allennlp.modules import conditional_random_field


# In[ ]:


TRANSITIONS = conditional_random_field.allowed_transitions(
    constraint_type='BIO', labels=ID_TO_LABEL
)


# In[ ]:


NUM_TAGS = 7


# In[ ]:


crf = conditional_random_field.ConditionalRandomField(
    # set to 7 because here "tags" means ['O', 'B-P', 'I-P', 'B-I', 'I-I', 'B-O', 'I-O']
    # no need to include 'BOS' and 'EOS' in "tags"
    num_tags=NUM_TAGS,
    constraints=TRANSITIONS,
    include_start_end_transitions=False
)


# ### 2-1. Log probability

# $x:$ `nn.Linear(MAX_LENGTH, NUM_TAGS)(torch.tensor(bioelmo_embedder.embed_sentence(tokens)[2]).unsqueeze(0))`
# 
# $\hat{y}:$ `tag.unsqueeze(0)`  
# 
# `crf.forward()` outputs the log probability: $\displaystyle \left( \log P(\hat{y}|x)=\log \frac{\exp{(score(x,\hat{y}))}}{\sum_{y\in\mathscr{y}}\exp{(score(x,y))}} \right)$

# #### 2-1-1. with BioELMo

# In[ ]:


affine = nn.Linear(MAX_LENGTH, NUM_TAGS)


# In[ ]:


"""
crf.forward(
    # [tokens] when batch_size == 1, tokens when batch_size > 1
    inputs=affine(bioelmo(batch_to_ids([tokens[0]]))['elmo_representations'][-1]),
    # .unsqueeze(0) is needed only when batch_size == 1
    tags=tags[0].unsqueeze(0)  
)
"""


# ### 2-2. Viterbi algorithm

# `crf.viterbi_tags()` outputs the most likely tag sequence: $\displaystyle \left( argmax_{y}P(y|x) \right)$

# #### 2-2-1. with BioELMo

# In[ ]:


"""
crf.viterbi_tags(
    # [tokens] when batch_size == 1, tokens when batch_size > 1
    logits=affine(bioelmo(batch_to_ids([tokens[0]]))['elmo_representations'][-1]),
    # .unsqueeze(0) is needed only when batch_size == 1
    mask=torch.tensor([1] * len(tokens[0])).to(bool).unsqueeze(0)
)
"""


# ### 2-3. Evaluation of NER prediction

# In[ ]:


"""
Y_id = crf.viterbi_tags(
    # [tokens] when batch_size == 1, tokens when batch_size > 1
    logits=affine(bioelmo(batch_to_ids([tokens[0]]))['elmo_representations'][-1]),
    # .unsqueeze(0) is needed only when batch_size == 1
    mask=torch.tensor([1] * len(tokens[0])).to(bool).unsqueeze(0)
)[0][0]
"""


# In[ ]:


"""
Y_tag = [ID_TO_LABEL[idx] for idx in Y_id]
Y_tag
"""


# In[ ]:


"""
T_id = tags[0].numpy().tolist()
"""


# In[ ]:


"""
T_id
"""


# In[ ]:


"""
T_tag = [ID_TO_LABEL[idx] for idx in T_id]
T_tag
"""


# In[ ]:


"""
print(classification_report(T_tag, Y_tag))
"""


# # 3. Dataset & Dataloader

# ### 3-1. Pad documents

# In[ ]:


# In ELMo token with ID 0 is used for padding
VOCAB_FILE_PATH = '/home/nakamura/bioelmo/vocab.txt'


# In[ ]:




# In[ ]:


command = shlex.split(f"head -n 1 {VOCAB_FILE_PATH}")
res = subprocess.Popen(command, stdout=subprocess.PIPE)
PAD_TOKEN = res.communicate()[0].decode('utf-8').strip()


# In[ ]:




# ### 3-2. Dataset

# In[ ]:


class EBMNLPDataset(torch.utils.data.Dataset):
    def __init__(self, text_files, token_files, p_files, i_files, o_files, max_length, pad_token):
        """
        text_files: list(str)
        token_files: list(str)
        p_files: list(str)
        i_files: list(str)
        o_files: list(str)
        """
        self.text_files = text_files
        self.token_files = token_files
        self.p_files = p_files
        self.i_files = i_files
        self.o_files = o_files
        assert len(text_files) == len(token_files) == len(p_files) == len(i_files) == len(o_files), 'All arguments must be lists of the same size'
        self.n = len(text_files)

        self.itol = {
            # 'BOS' and 'EOS' are not needed to be explicitly included
            0:'O', 1:'I-P', 2:'B-P', 3:'I-I', 4:'B-I', 5:'I-O', 6:'B-O'
        }
        self.ltoi = {
            self.itol[i]:i for i in range(len(self.itol))
        }
        
        self.max_length = max_length
        self.pad_token = pad_token

    def __len__(self):
        return self.n
        
    
    def __getitem__(self, idx):
        returns = {}
        
        with open(self.text_files[idx]) as f:
            # Raw document. Example:
            # [Triple therapy regimens involving H2 blockaders for therapy of Helicobacter pylori infections].
            # Comparison of ranitidine and lansoprazole in short-term low-dose triple therapy for Helicobacter pylori infection. 
            returns['text'] = f.read()
    
        with open(self.token_files[idx]) as f:
            # Tokenized document. Example:
            # ['[', 'Triple', 'therapy', 'regimens', 'involving', 'H2', 'blockaders',
            #  'for', 'therapy', 'of', 'Helicobacter', 'pylori', 'infections', ']', '.',
            #  'Comparison', 'of', 'ranitidine', 'and', 'lansoprazole', 'in',
            #  'short-term', 'low-dose', 'triple', 'therapy', 'for',
            #  'Helicobacter', 'pylori', 'infection', '.', ...
            tokens = f.read().split()
            tokens_padded = tokens + [self.pad_token] * (self.max_length - len(tokens))
            returns['tokens'] = tokens_padded
    
        with open(self.p_files[idx]) as f:
            # sequence of 0 (O-tag) or 1 (I-tag)
            # e.g., '1,1,0,1,1,0,...'
            p = f.read()
            
        with open(self.i_files[idx]) as f:
            # sequence of 0 (O-tag) or 1 (I-tag)
            # e.g., '1,1,0,1,1,0,...'
            i = f.read()  

        with open(self.o_files[idx]) as f:
            # sequence of 0 (O-tag) or 1 (I-tag)
            # e.g., '1,1,0,1,1,0,...'
            o = f.read()  


        # sequence of 0 (O-tag) or 2 (B-P-tag) or 1 (I-P-tag)
        # e.g., '2,1,0,2,1,0,...'
        p = re.sub('1', f"{self.ltoi['I-P']}", p)
        p = re.sub(f"^{self.ltoi['I-P']}", f"{self.ltoi['B-P']}", p)
        p = re.sub(f"{self.ltoi['O']},{self.ltoi['I-P']}", f"{self.ltoi['O']},{self.ltoi['B-P']}", p)
        
        # sequence of 0 (O-tag) or 4 (B-I-tag) or 3 (I-I-tag)
        # e.g., '4,3,0,4,3,0,...'
        i = re.sub('1', f"{self.ltoi['I-I']}", i)
        i = re.sub(f"^{self.ltoi['I-I']}", f"{self.ltoi['B-I']}", i)
        i = re.sub(f"{self.ltoi['O']},{self.ltoi['I-I']}", f"{self.ltoi['O']},{self.ltoi['B-I']}", i)

        # sequence of 0 (O-tag) or 6 (B-O-tag) or 5 (I-O-tag)
        # e.g., '6,5,0,6,5,0,...'
        o = re.sub('1', f"{self.ltoi['I-O']}", o)
        o = re.sub(f"^{self.ltoi['I-O']}", f"{self.ltoi['B-O']}", o)
        o = re.sub(f"{self.ltoi['O']},{self.ltoi['I-O']}", f"{self.ltoi['O']},{self.ltoi['B-O']}", o)
        

        # integrated P, I and O tags        
        lp = [int(x) for x in p.split(',')]
        li = [int(x) for x in i.split(',')]
        lo = [int(x) for x in o.split(',')]
        tag = torch.tensor(np.max(np.vstack([lp,li,lo]), axis=0))

        seq_len = tag.shape[0]
        padding = torch.tensor([int(self.ltoi['O'])] * (self.max_length - seq_len))
        tag_padded = torch.cat([tag, padding])
        returns['tag'] = tag_padded

        mask = torch.cat([torch.ones_like(tag), torch.zeros_like(padding)]).to(bool)
        returns['mask'] = mask

        return returns


# In[ ]:


ds = EBMNLPDataset(
    text_file_train, token_file_train, p_file_train, i_file_train, o_file_train,
    max_length=MAX_LENGTH,
    pad_token=PAD_TOKEN
)


# In[ ]:


ds_train, ds_val = train_test_split(ds, train_size=0.8, random_state=42)


# In[ ]:


ds_train[0]['tokens']


# In[ ]:


ds_train[0]['tag']


# In[ ]:


ds_train[0]['mask']


# ### 3-3. DataLoader

# In[ ]:


BATCH_SIZE = 16


# In[ ]:


dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
dl_val = torch.utils.data.DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)


# ### 3-4. DataLoader -> BioELMo -> CRF

# In[ ]:


class EBMNLPTagger(pl.LightningModule):
    def __init__(self, bioelmo, hidden_to_tag, crf, itol):
        """
        bioelmo_embedder: allennlp.commands.elmo.ElmoEmbedder
        crf: allennlp.modules.conditional_random_field.ConditionalRandomField
        """
        super().__init__()
        self.bioelmo = bioelmo
        self.hidden_to_tag = hidden_to_tag
        self.crf = crf
        self.itol = itol
        self.loss = []


    def get_device(self):
        return self.crf.state_dict()['transitions'].device
    
    
    def id_to_tag(self, T_padded, Y_packed):
        """
        T_padded: torch.tensor
        Y_packed: torch.nn.utils.rnn.PackedSequence
        """
        Y_padded, Y_len = rnn.pad_packed_sequence(Y_packed, batch_first=True, padding_value=-1)
        Y_padded = Y_padded[Y_packed.unsorted_indices].numpy().tolist()
        Y_len = Y_len[Y_packed.unsorted_indices].numpy().tolist()
        Y = [[self.itol[ix] for ix in ids[:length]] for ids, length in zip(Y_padded, Y_len)]

        T_padded = T_padded.numpy().tolist()
        T = [[self.itol[ix] for ix in ids[:length]] for ids, length in zip(T_padded, Y_len)]
        
        return T, Y
    

    def forward(self, character_ids, tags, masks):
        """
        inputs: dict ({'tokens':list(list(str)), 'tag':torch.Tensor, 'mask':torch.BoolTensor)
        """
        
        # characted_ids -> BioELMo hidden state of the last layer
        # Turn on gradient tracking
        out = self.bioelmo(character_ids)['elmo_representations'][-1]
        out.requires_grad_()
        
        # Affine transformation (Hidden_dim -> N_tag)
        out = self.hidden_to_tag(out)
        
        # Log probability
        log_prob = self.crf.forward(out, tags, masks)
        
        # top k=1 tagging
        Y = [torch.tensor(result[0]) for result in self.crf.viterbi_tags(logits=out, mask=masks)]
        Y = rnn.pack_sequence(Y, enforce_sorted=False)
        
        return log_prob, Y

    
    def training_step(self, batch, batch_nb):
        # Process on individual mini-batches
        """
        (batch) -> (dict or OrderedDict)
        # Caution: key for loss function must exactly be 'loss'.
        """
        
        # tokens: list(list(str))
        tokens = np.array(batch['tokens']).T
        n_batch = tokens.shape[0]
        tokens = tokens.tolist()
        
        # character_ids: torch.tensor(n_batch, max_len)
        cids = batch_to_ids(tokens)

        # character_ids, tags & masks
        cids = cids.to(self.get_device())
        tags = batch['tag'].to(self.get_device())
        masks = batch['mask'].to(self.get_device())
            
        # Negative Log Likelihood
        log_prob, Y = self.forward(cids, tags, masks)
        returns = {'loss':log_prob * (-1.0), 'T':tags, 'Y':Y}
        return returns
    

    def training_step_end(self, outputs):
        """
        outputs(dict) -> loss(dict or OrderedDict)
        # Caution: key must exactly be 'loss'.
        """
        loss = torch.mean(outputs['loss'])
        
        progress_bar = {'train_loss':loss}
        returns = {'loss':loss, 'T':outputs['T'], 'Y':outputs['Y'], 'progress_bar':progress_bar}
        return returns
    
    
    def training_epoch_end(self, outputs):
        """
        outputs(list of dict) -> loss(dict or OrderedDict)
        # Caution: key must exactly be 'loss'.
        """
        if len(outputs) > 1:
            loss = torch.mean(torch.tensor([output['loss'] for output in outputs]))
        else:
            loss = torch.mean(outputs[0]['loss'])
        
        Y = []
        T = []
        
        for output in outputs:
            T_batch, Y_batch = self.id_to_tag(output['T'].cpu(), output['Y'].cpu())
            T += T_batch
            Y += Y_batch

        get_logger(VERSION).info(f'Training Epoch {self.current_epoch} ==========')
        get_logger(VERSION).info(loss)
        get_logger(VERSION).info(classification_report(T, Y, 4))
        print(classification_report(T, Y, 4))

        self.loss.append(loss)
        progress_bar = {'train_loss':loss}
        returns = {'loss':loss, 'progress_bar':progress_bar}
        return returns
    
    

    def validation_step(self, batch, batch_nb):
        # Process on individual mini-batches
        """
        (batch) -> (dict or OrderedDict)
        """   
        
        # tokens: list(list(str))
        tokens = np.array(batch['tokens']).T
        n_batch = tokens.shape[0]
        tokens = tokens.tolist()
        
        # character_ids: torch.tensor(n_batch, max_len)
        cids = batch_to_ids(tokens)

        # character_ids, tags & masks
        cids = cids.to(self.get_device())
        tags = batch['tag'].to(self.get_device())
        masks = batch['mask'].to(self.get_device())
            
        # Negative Log Likelihood
        log_prob, Y = self.forward(cids, tags, masks)
        returns = {'loss':log_prob * (-1.0), 'T':tags, 'Y':Y}
        return returns
    
    
    def validation_end(self, outputs):
        """
        For single dataloader:
            outputs(list of dict) -> (dict or OrderedDict)
        For multiple dataloaders:
            outputs(list of (list of dict)) -> (dict or OrderedDict)
        """  

        if len(outputs) > 1:
            loss = torch.mean(torch.tensor([output['loss'] for output in outputs]))
        else:
            loss = torch.mean(outputs[0]['loss'])
        
        Y = []
        T = []
        
        for output in outputs:
            T_batch, Y_batch = self.id_to_tag(output['T'].cpu(), output['Y'].cpu())
            T += T_batch
            Y += Y_batch
            
        get_logger(VERSION).info(f'Validation Epoch {self.current_epoch} ==========')
        get_logger(VERSION).info(loss)
        get_logger(VERSION).info(classification_report(T, Y, 4))
        print(classification_report(T, Y, 4))

        self.loss.append(loss)
        progress_bar = {'val_loss':loss}
        returns = {'loss':loss, 'progress_bar':progress_bar}
        return returns        

    
    def configure_optimizers(self):
        LR=1e-4
        optimizer = optim.SGD(self.parameters(), LR)
        return optimizer
    
    def train_dataloader(self):
        return dl_train
    
    def val_dataloader(self):
        return dl_val


# In[ ]:


ebmnlp = EBMNLPTagger(bioelmo, nn.Linear(MAX_LENGTH, NUM_TAGS), crf, ID_TO_LABEL)


# In[ ]:


device = torch.device('cuda:2')


# In[ ]:


ebmnlp.to(device)


# In[ ]:


trainer = pl.Trainer(
    max_epochs=50,
    train_percent_check=1,
)


# In[ ]:


trainer.fit(ebmnlp)

