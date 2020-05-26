#!/usr/bin/env python
# coding: utf-8

# # 0. Preparation

# ### 0-1. Dependencies

import argparse
import logging
import os, random
import re
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.optim as optim

import subprocess
import shlex

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules import conditional_random_field

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from seqeval.metrics import accuracy_score as seq_accuracy_score
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import classification_report as seq_classification_report

from tqdm import tqdm_notebook as tqdm
import pytorch_lightning as pl

from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG


# ### 0-2. Prepare for logging

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


# ### 0-3. NER tag
# 'BOS' and 'EOS' are not needed to be explicitly included
ID_TO_LABEL = {0: 'O', 1: 'I-P', 2: 'B-P', 3: 'I-I', 4: 'B-I', 5: 'I-O', 6: 'B-O'}
LABEL_TO_ID = {v:k for k, v in ID_TO_LABEL.items()}


def integrate_pio(p, i, o, ltoi):
    """
    str, str, str, dict -> torch.tensor

    input:
        p (str): IO tagging for Patient entity (e.g., '0,0,1,1,0,0,...')
        i (str): IO tagging for Intervention entity (e.g., '0,0,1,1,0,0,...')
        o (str): IO tagging for Outcome entity (e.g., '0,0,1,1,0,0,...')
        itol (dict): LABEL-TO-ID Mapping
    output:
        torch.Tensor: IOB1 tagging for P/I/O (e.g., ([0,4,3,3,3,0,6,5,5,5,...]))
    """
    
    # sequence of 0 (O-tag) or 2 (B-P-tag) or 1 (I-P-tag)
    # e.g., '2,1,0,2,1,0,...'
    p = re.sub('1', f"{ltoi['I-P']}", p)
    p = re.sub(f"^{ltoi['I-P']}", f"{ltoi['B-P']}", p)
    p = re.sub(f"{ltoi['O']},{ltoi['I-P']}", f"{ltoi['O']},{ltoi['B-P']}", p)
        
    # sequence of 0 (O-tag) or 4 (B-I-tag) or 3 (I-I-tag)
    # e.g., '4,3,0,4,3,0,...'
    i = re.sub('1', f"{ltoi['I-I']}", i)
    i = re.sub(f"^{ltoi['I-I']}", f"{ltoi['B-I']}", i)
    i = re.sub(f"{ltoi['O']},{ltoi['I-I']}", f"{ltoi['O']},{ltoi['B-I']}", i)

    # sequence of 0 (O-tag) or 6 (B-O-tag) or 5 (I-O-tag)
    # e.g., '6,5,0,6,5,0,...'
    o = re.sub('1', f"{ltoi['I-O']}", o)
    o = re.sub(f"^{ltoi['I-O']}", f"{ltoi['B-O']}", o)
    o = re.sub(f"{ltoi['O']},{ltoi['I-O']}", f"{ltoi['O']},{ltoi['B-O']}", o)
       
    # integrated P, I and O tags        
    lp = [int(x) for x in p.split(',')]
    li = [int(x) for x in i.split(',')]
    lo = [int(x) for x in o.split(',')]
    tag = torch.tensor(np.max(np.vstack([lp,li,lo]), axis=0))
    
    return tag


# ### 0-4. Prepare File
def path_finder(top_dir):
    """
    str -> dict

    inputs:
        top_dir (str): The top directory name of the dataset (e.g., './ebm_nlp_1_00')
    outputs:
        paths (dict): Paths of text, token, P_annotation, I_annotation and O_annotation for train & test dataset.    
    """
    DIR_ROOT = Path(str(top_dir))
    DIR_DOC_1 = DIR_ROOT / 'documents'
    DIR_LABEL_1 = DIR_ROOT / 'annotations/aggregated/starting_spans'

    text_file_all = sorted(glob(str(DIR_DOC_1 / '*.text')))
    token_file_all = sorted(glob(str(DIR_DOC_1 / '*.tokens')))

    p_file_train = sorted(glob(str(DIR_LABEL_1 / 'participants/train/*.ann')))
    i_file_train = sorted(glob(str(DIR_LABEL_1 / 'interventions/train/*.ann')))
    o_file_train = sorted(glob(str(DIR_LABEL_1 / 'outcomes/train/*.ann')))

    idx_all = [re.compile(r'/([0-9]+)[^0-9]+').findall(path)[0] for path in text_file_all]
    idx_train = [re.compile(r'/([0-9]+)[^0-9]+').findall(path)[0] for path in p_file_train]
    idx_test = [idx for idx in idx_all if idx not in idx_train]

    text_file_train = sorted([str(DIR_DOC_1 / f'{idx}.text') for idx in idx_train])
    text_file_test = sorted([str(DIR_DOC_1 / f'{idx}.text') for idx in idx_test])
    token_file_train = sorted([str(DIR_DOC_1 / f'{idx}.tokens') for idx in idx_train])
    token_file_test = sorted([str(DIR_DOC_1 / f'{idx}.tokens') for idx in idx_test])

    p_file_test = sorted(glob(str(DIR_LABEL_1 / 'participants/test/gold/*.ann')))
    i_file_test = sorted(glob(str(DIR_LABEL_1 / 'interventions/test/gold/*.ann')))
    o_file_test = sorted(glob(str(DIR_LABEL_1 / 'outcomes/test/gold/*.ann')))
    
    paths = {
        'train' : (text_file_train, token_file_train, p_file_train, i_file_train, o_file_train),
        'test' : (text_file_test, token_file_test, p_file_test, i_file_test, o_file_test)
    }

    return paths


# ### 1. Dataset

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
        self.pmids = [int(re.compile(r'/([0-9]+)[^0-9]+').findall(path)[0]) for path in self.text_files]
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
        
        returns['pmid'] = self.pmids[idx] 

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

        # torch.tensor of IBO1 tag (e.g., ([0,6,5,5,5,0,0,4,3,3,3,0,0,...]))
        tag = integrate_pio(p, i, o, self.ltoi)

        seq_len = tag.shape[0]
        padding = torch.tensor([int(self.ltoi['O'])] * (self.max_length - seq_len))
        tag_padded = torch.cat([tag, padding])
        returns['tag'] = tag_padded

        mask = torch.cat([torch.ones_like(tag), torch.zeros_like(padding)]).to(bool)
        returns['mask'] = mask

        return returns



# 2. LightningModule

class EBMNLPTagger(pl.LightningModule):
    def __init__(self, hparams): 
        """
        input:
            hparams: dict
               {'config' : config
                'bioelmo' : allennlp.module.elmo.Elmo
                'hidden_to_tag' : torch.nn.Linear
                'crf': allennlp.modules.conditional_random_field.ConditionalRandomField
                'itol': dict
                'dl_train': torch.utils.data.DataLoader 
                'dl_val': torch.utils.data.DataLoader
                'dl_test': torch.utils.data.DataLoader
               }
        """
        super().__init__()
        self.hparams = hparams
        self.itol = ID_TO_LABEL

        # Load Pretrained BioELMo
        DIR_ELMo = Path(str(self.hparams.bioelmo_dir))
        self.bioelmo = Elmo(DIR_ELMo / 'biomed_elmo_options.json', DIR_ELMo / 'biomed_elmo_weights.hdf5', 1, requires_grad=bool(self.hparams.fine_tune_bioelmo), dropout=0)

        # ELMo Padding token (In ELMo token with ID 0 is used for padding)
        VOCAB_FILE_PATH = DIR_ELMo / 'vocab.txt'
        command = shlex.split(f"head -n 1 {VOCAB_FILE_PATH}")
        res = subprocess.Popen(command, stdout=subprocess.PIPE)
        self.bioelmo_pad_token = res.communicate()[0].decode('utf-8').strip()

        # Initialize Intermediate Affine Layer 
        self.hidden_to_tag = nn.Linear(int(self.hparams.max_length), len(self.itol))

        # Initialize CRF
        TRANSITIONS = conditional_random_field.allowed_transitions(
            constraint_type='BIO', labels=self.itol
        )
        self.crf = conditional_random_field.ConditionalRandomField(
            # set to 7 because here "tags" means ['O', 'B-P', 'I-P', 'B-I', 'I-I', 'B-O', 'I-O']
            # no need to include 'BOS' and 'EOS' in "tags"
            num_tags=len(self.itol),
            constraints=TRANSITIONS,
            include_start_end_transitions=False
        )
        self.crf.reset_parameters()
 

    def get_device(self):
        return self.crf.state_dict()['transitions'].device
   
   
    def unpack_pred_tags(self, Y_packed):
        """
        input:
            Y_packed: torch.nn.utils.rnn.PackedSequence
        output:
            Y: list(list(str))
                Predicted NER tagging sequence.
        """
        Y_padded, Y_len = rnn.pad_packed_sequence(Y_packed, batch_first=True, padding_value=-1)
        Y_padded = Y_padded.numpy().tolist()
        Y_len = Y_len.numpy().tolist()

        # Replace B- tag with I- tag because the original paper defines the NER task as identification of spans, not entities
        Y = [[self.itol[ix].replace('B-', 'I-') for ix in ids[:length]] for ids, length in zip(Y_padded, Y_len)]

        return Y


    def unpack_gold_and_pred_tags(self, T_padded, Y_packed):
        """
        input:
            T_padded: torch.tensor
            Y_packed: torch.nn.utils.rnn.PackedSequence
        output:
            T: list(list(str))
                Gold NER tagging sequence.
            Y: list(list(str))
                Predicted NER tagging sequence.
        """
        Y = self.unpack_pred_tags(Y_packed)
        Y_len = [len(seq) for seq in Y]

        T_padded = T_padded.numpy().tolist()

        # Replace B- tag with I- tag because the original paper defines the NER task as identification of spans, not entities
        T = [[self.itol[ix].replace('B-', 'I-') for ix in ids[:length]] for ids, length in zip(T_padded, Y_len)]
        
        return T, Y



    def forward(self, tokens, tags=None, masks=None):
        """
        inputs:
            tokens: list(list(str))
            tags: torch.Tensor in size (n_batch, max_len)
            masks: torch.Booltensor in size (n_batch, max_len)
                Masks indicating the original sequences with True and padded sections with False.
        outputs:
            log_prob: torch.Tensor in size (1)
                Log probability of the gold standard NER tagging calculated with sum-product algorithm.
            Y: torch.Tensor in size(n_batch, max_len)
                The most probable NER tagging sequence predicted with Viterbi algorithm.
        """
        # tokens: list(list(str))
        # # check if tokens have the same lengths
        lengths = [len(seq) for seq in tokens]
        len_max = max(lengths)
        len_min = min(lengths)

        # # if tokens have different lengths, pad with self.bioelmo_pad_token
        if len_max > len_min:
            tokens = [seq + [self.bioelmo_pad_token] * (length - len_max) for seq, length in zip(tokens, lengths)]

        if masks is None:
            masks = torch.stack([torch.cat([torch.ones(length), torch.zeros(length - len_max)]).to(bool) for length in lengths])

        
        # character_ids: torch.tensor(n_batch, max_len)
        character_ids = batch_to_ids(tokens)
        character_ids = character_ids.to(self.get_device())

        # characted_ids -> BioELMo hidden state of the last layer
        # Turn on gradient tracking
        out = self.bioelmo(character_ids)['elmo_representations'][-1]
        out.requires_grad_()
        
        # Affine transformation (Hidden_dim -> N_tag)
        out = self.hidden_to_tag(out)
        

        if tags is not None:
            tags = tags.to(self.get_device())
            masks = masks.to(self.get_device())
        
            # Log probability
            log_prob = self.crf.forward(out, tags, masks)
        
            # top k=1 tagging
            Y = [torch.tensor(result[0]) for result in self.crf.viterbi_tags(logits=out, mask=masks)]
            Y = rnn.pack_sequence(Y, enforce_sorted=False)
        
            return log_prob, Y

        else:
            masks = masks.to(self.get_device())
        
            # top k=1 tagging
            Y = [torch.tensor(result[0]) for result in self.crf.viterbi_tags(logits=out, mask=masks)]
            Y = rnn.pack_sequence(Y, enforce_sorted=False)
        
            return Y

    
    def training_step(self, batch, batch_nb, *optimizer_idx):
        # Process on individual mini-batches
        """
        (batch) -> (dict or OrderedDict)
        # Caution: key for loss function must exactly be 'loss'.
        """

        # DataLoader transposes tokens (max_len, n_batch), so transpose it again (n_batch, max_len)        
        tokens = batch['tokens']
        tokens = np.array(tokens).T
        n_batch = tokens.shape[0]
        tokens = tokens.tolist()

        tags = batch['tag'].to(self.get_device())
        masks = batch['mask'].to(self.get_device())
            
        # Negative Log Likelihood
        log_prob, Y = self.forward(tokens, tags, masks)
        returns = {'loss':log_prob * (-1.0), 'T':tags, 'Y':Y, 'I':batch['pmid']}
        return returns
    

    def training_step_end(self, outputs):
        """
        outputs(dict) -> loss(dict or OrderedDict)
        # Caution: key must exactly be 'loss'.
        """
        loss = torch.mean(outputs['loss'])
        
        progress_bar = {'train_loss':loss}
        returns = {'loss':loss, 'T':outputs['T'], 'Y':outputs['Y'], 'I':outputs['I'], 'progress_bar':progress_bar}
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
        
        I = []
        Y = []
        T = []
        
        for output in outputs:
            T_batch, Y_batch = self.unpack_gold_and_pred_tags(output['T'].cpu(), output['Y'].cpu())
            T += T_batch
            Y += Y_batch
            I += output['I'].cpu().numpy().tolist()

        get_logger(self.hparams.version).info(f'========== Training Epoch {self.current_epoch} ==========')
        get_logger(self.hparams.version).info(f'Loss: {loss.item()}')
        get_logger(self.hparams.version).info(f'Entity-wise classification report\n{seq_classification_report(T, Y, 4)}')
        get_logger(self.hparams.version).info(f'Token-wise classification report\n{span_classification_report(T, Y, 4)}')

        progress_bar = {'train_loss':loss}
        returns = {'loss':loss, 'progress_bar':progress_bar}
        return returns
    
    

    def validation_step(self, batch, batch_nb):
        # Process on individual mini-batches
        """
        (batch) -> (dict or OrderedDict)
        """
   
        # DataLoader transposes tokens (max_len, n_batch), so transpose it again (n_batch, max_len)        
        tokens = batch['tokens']
        tokens = np.array(tokens).T
        n_batch = tokens.shape[0]
        tokens = tokens.tolist()
        
        tags = batch['tag'].to(self.get_device())
        masks = batch['mask'].to(self.get_device())

        # Negative Log Likelihood
        log_prob, Y = self.forward(tokens, tags, masks)
        returns = {'loss':log_prob * (-1.0), 'T':tags, 'Y':Y, 'I':batch['pmid']}
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
        
        I = []
        Y = []
        T = []
        
        for output in outputs:
            T_batch, Y_batch = self.unpack_gold_and_pred_tags(output['T'].cpu(), output['Y'].cpu())
            T += T_batch
            Y += Y_batch
            I += output['I'].cpu().numpy().tolist()
            
        get_logger(self.hparams.version).info(f'========== Validation Epoch {self.current_epoch} ==========')
        get_logger(self.hparams.version).info(f'Loss: {loss.item()}')
        get_logger(self.hparams.version).info(f'Entity-wise classification report\n{seq_classification_report(T, Y, 4)}')
        get_logger(self.hparams.version).info(f'Token-wise classification report\n{span_classification_report(T, Y, 4)}')

        progress_bar = {'val_loss':loss}
        returns = {'loss':loss, 'progress_bar':progress_bar}
        return returns        


    def test_step(self, batch, batch_nb):
        # Process on individual mini-batches
        """
        (batch) -> (dict or OrderedDict)
        """
        # DataLoader transposes tokens (max_len, n_batch), so transpose it again (n_batch, max_len)        
        tokens = batch['tokens']
        tokens = np.array(tokens).T
        n_batch = tokens.shape[0]
        tokens = tokens.tolist()

        tags = batch['tag'].to(self.get_device())
        masks = batch['mask'].to(self.get_device())

        # Negative Log Likelihood
        log_prob, Y = self.forward(tokens, tags, masks)
        returns = {'loss':log_prob * (-1.0), 'T':tags, 'Y':Y, 'I':batch['pmid']}
        return returns


    def test_epoch_end(self, outputs):
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

        I = []
        Y = []
        T = []

        for output in outputs:
            T_batch, Y_batch = self.unpack_gold_and_pred_tags(output['T'].cpu(), output['Y'].cpu())
            T += T_batch
            Y += Y_batch
            I += output['I'].cpu().numpy().tolist()

        get_logger(self.hparams.version).info(f'========== Test ==========')
        get_logger(self.hparams.version).info(f'Loss: {loss.item()}')
        get_logger(self.hparams.version).info(f'Entity-wise classification report\n{seq_classification_report(T, Y, 4)}')
        get_logger(self.hparams.version).info(f'Token-wise classification report\n{span_classification_report(T, Y, 4)}')

        progress_bar = {'test_loss':loss}
        returns = {'loss':loss, 'progress_bar':progress_bar}
        return returns


    def configure_optimizers(self):
        if self.hparams.fine_tune_bioelmo:
            optimizer_bioelmo_1 = optim.Adam(self.bioelmo.parameters(), lr=float(self.harapms.lr_bioelmo))
            optimizer_bioelmo_2 = optim.Adam(self.hidden_to_tag.parameters(), lr=float(self.hparams.lr_bioelmo))
            optimizer_crf = optim.Adam(self.crf.parameters(), lr=float(self.hparams.lr))
            return [optimizer_bioelmo_1, optimizer_bioelmo_2, optimizer_crf]
        else:        
            optimizer = optim.Adam(self.parameters(), lr=float(self.hparams.lr))
            return optimizer


    def train_dataloader(self):
        ds_train_val = EBMNLPDataset(
            *path_finder(self.hparams.data_dir)['train'],
            self.hparams.max_length,
            self.bioelmo_pad_token
        )

        ds_train, _ = train_test_split(ds_train_val, train_size=0.8, random_state=self.hparams.random_state)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=self.hparams.batch_size, shuffle=True)
        return dl_train


    def val_dataloader(self):
        ds_train_val = EBMNLPDataset(
            *path_finder(self.hparams.data_dir)['train'],
            self.hparams.max_length,
            self.bioelmo_pad_token
        )

        _, ds_val = train_test_split(ds_train_val, train_size=0.8, random_state=self.hparams.random_state)
        dl_val = torch.utils.data.DataLoader(ds_val, batch_size=self.hparams.batch_size, shuffle=False)
        return dl_val


    def test_dataloader(self):
        ds_test = EBMNLPDataset(
            *path_finder(self.hparams.data_dir)['test'],
            self.hparams.max_length,
            self.bioelmo_pad_token
        )
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=self.hparams.batch_size, shuffle=False)
        return dl_test



# 3. Evaluation function

def span_classification_report(T, Y, digits=4):
    """
    Token-wise metrics of NER IOE1 tagging task.
    T: list(list(str)) True labels 
    Y: list(list(str)) Pred labels
    """
    T_flatten = []
    Y_flatten = []
    n_sample = len(T)

    for i in range(n_sample):
        T_flatten += [label.replace('B-', 'I-') for label in T[i]]
        Y_flatten += [label.replace('B-', 'I-') for label in Y[i]]

    label_types = [label_type for label_type in set(T_flatten) if label_type != 'O']

    return classification_report(T_flatten, Y_flatten, labels=label_types, digits=digits)



# 4. MAIN
def main(config):
    # ### 4-0. Print config
    create_logger(config.version)
    get_logger(config.version).info(config)
    
    
    # ### 4-1. DataLoader -> BioELMo -> CRF
    
    ebmnlp = EBMNLPTagger(config)


    # ### 4-2. Training
    if config.cuda is None:
        device = torch.device('cuda')
    else:
        device = torch.device(f'cuda:{config.cuda}')

    ebmnlp.to(device)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath='./models/ebmnlp_bioelmo_crf/ebmnlp_bioelmo_crf.ckpt'
    )

    trainer = pl.Trainer(
        max_epochs=int(config.max_epochs),
        fast_dev_run=bool(config.debug_mode),
        checkpoint_callback=checkpoint_callback
    )
    trainer.fit(ebmnlp)

    # ### 4-3. Test
    trainer.test()


if __name__=='__main__':
    def get_args():
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--debug', '--debug-mode', action='store_true', dest='debug_mode', help='Set this option for debug mode')
        parser.add_argument('-d', '--dir', '--data-dir', dest='data_dir', type=str, default='./official/ebm_nlp_1_00', help='Data Directory')
        parser.add_argument('--bioelmo-dir', dest='bioelmo_dir', type=str, default='./models/bioelmo', help='BioELMo Directory')
        parser.add_argument('-v', '--version', dest='version', type=str, help='Experiment Name')
        parser.add_argument('-e', '--max-epochs', dest='max_epochs', type=int, default='15', help='Max Epochs (Default: 15)')
        parser.add_argument('--max-length', dest='max_length', type=int, default='1024', help='Max Length (Default: 1024)')
        parser.add_argument('-l', '--lr', dest='lr', type=float, default='1e-2', help='Learning Rate (Default: 1e-2)')
        parser.add_argument('--fine-tune-bioelmo', action='store_true', dest='fine_tune_bioelmo', help='Whether to Fine Tune BioELMo')
        parser.add_argument('--lr-bioelmo', dest='lr_bioelmo', type=float, default='1e-4', help='Learning Rate in BioELMo Fine-tuning')
        parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default='16', help='Batch size (Default: 16)')
        parser.add_argument('-c', '--cuda', dest='cuda', default=None, help='CUDA Device Number')
        parser.add_argument('-r', '--random-state', dest='random_state', type=int, default='42', help='Random state (Default: 42)')
        namespace = parser.parse_args()
        return namespace

    config = get_args()
    print(config)
    main(config)
