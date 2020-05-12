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

    
def main(config):    
    VERSION = str(config.version)  # 実験番号
    create_logger(VERSION)
    get_logger(VERSION).info(config)
    
    
    # ### 0-2. Prepare files
    
    DIR_DOC_1 = Path('ebm_nlp_1_00/documents')
    DIR_LABEL_1 = Path('ebm_nlp_1_00/annotations')
    
    
    text_file_all = sorted(glob(str(DIR_DOC_1 / '*.text')))
    token_file_all = sorted(glob(str(DIR_DOC_1 / '*.tokens')))
    
    p_file_train = sorted(glob(str(DIR_LABEL_1 / 'aggregated/starting_spans/participants/train/*.ann')))
    i_file_train = sorted(glob(str(DIR_LABEL_1 / 'aggregated/starting_spans/interventions/train/*.ann')))
    o_file_train = sorted(glob(str(DIR_LABEL_1 / 'aggregated/starting_spans/outcomes/train/*.ann')))
    
    idx_all = [re.compile(r'/([0-9]+)[^0-9]+').findall(path)[0] for path in text_file_all]
    idx_train = [re.compile(r'/([0-9]+)[^0-9]+').findall(path)[0] for path in p_file_train]
    idx_test = [idx for idx in idx_all if idx not in idx_train]
    
    text_file_train = sorted([str(DIR_DOC_1 / f'{idx}.text') for idx in idx_train])
    text_file_test = sorted([str(DIR_DOC_1 / f'{idx}.text') for idx in idx_test])
    token_file_train = sorted([str(DIR_DOC_1 / f'{idx}.tokens') for idx in idx_train])
    token_file_test = sorted([str(DIR_DOC_1 / f'{idx}.tokens') for idx in idx_test])
    
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
    
    # 'BOS' and 'EOS' are not needed to be explicitly included
    ID_TO_LABEL = {0: 'O', 1: 'I-P', 2: 'B-P', 3: 'I-I', 4: 'B-I', 5: 'I-O', 6: 'B-O'}
    LABEL_TO_ID = {v:k for k, v in ID_TO_LABEL.items()}
    
    
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
    
    
    tags = [integrate_pio(p,i,o) for p,i,o in zip(p,i,o)]
    
    
    
    
    # ### 0-4. Check document lengths
    
    document_lengths = []
    
    for file in tqdm(token_file_all):
        with open(file) as f:
            document_lengths.append(len(f.read().split()))
    
    MAX_LENGTH = 1024
    
    
    # # 1. BioELMo
    
    from allennlp.modules.elmo import Elmo, batch_to_ids
    
    # - options.json: CNN, BiLSTMモデルの形状を規定
    # - bioelmo: hdf5ファイル. CNN, BiLSTMモデルの重みを格納している
    
    DIR_ELMo = Path('/home/nakamura/bioelmo')
    bioelmo = Elmo(DIR_ELMo / 'options.json', DIR_ELMo / 'bioelmo', 1, requires_grad=False, dropout=0)
    
    
    # # 2. CRF
    
    from allennlp.modules import conditional_random_field
    
    TRANSITIONS = conditional_random_field.allowed_transitions(
        constraint_type='BIO', labels=ID_TO_LABEL
    )
    
    NUM_TAGS = 7
    
    crf = conditional_random_field.ConditionalRandomField(
        # set to 7 because here "tags" means ['O', 'B-P', 'I-P', 'B-I', 'I-I', 'B-O', 'I-O']
        # no need to include 'BOS' and 'EOS' in "tags"
        num_tags=NUM_TAGS,
        constraints=TRANSITIONS,
        include_start_end_transitions=False
    )
    crf.reset_parameters() 
    affine = nn.Linear(MAX_LENGTH, NUM_TAGS)
    
    
    # # 3. Dataset & Dataloader
    
    # ### 3-1. Pad documents
    
    # In ELMo token with ID 0 is used for padding
    VOCAB_FILE_PATH = '/home/nakamura/bioelmo/vocab.txt'
    
    command = shlex.split(f"head -n 1 {VOCAB_FILE_PATH}")
    res = subprocess.Popen(command, stdout=subprocess.PIPE)
    PAD_TOKEN = res.communicate()[0].decode('utf-8').strip()
    
    
    # ### 3-2. Dataset
    
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
    
    
    ds = EBMNLPDataset(
        text_file_train, token_file_train, p_file_train, i_file_train, o_file_train,
        max_length=MAX_LENGTH,
        pad_token=PAD_TOKEN
    )
    
    ds_train, ds_val = train_test_split(ds, train_size=0.8, random_state=42)
    
    
    # ### 3-3. DataLoader
    BATCH_SIZE = 16
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    
    
    # ### 3-4. DataLoader -> BioELMo -> CRF
    
    class EBMNLPTagger(pl.LightningModule):
        def __init__(self, config, bioelmo, hidden_to_tag, crf, itol):
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
            self.config = config
    
    
        def get_device(self):
            return self.crf.state_dict()['transitions'].device
        
        
        def id_to_tag(self, T_padded, Y_packed):
            """
            T_padded: torch.tensor
            Y_packed: torch.nn.utils.rnn.PackedSequence
            """
            Y_padded, Y_len = rnn.pad_packed_sequence(Y_packed, batch_first=True, padding_value=-1)
            Y_padded = Y_padded.numpy().tolist()
            Y_len = Y_len.numpy().tolist()
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
    
            self.loss.append(loss)
            progress_bar = {'val_loss':loss}
            returns = {'loss':loss, 'progress_bar':progress_bar}
            return returns        
    
        
        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=float(self.config.lr))
            return optimizer
        
        def train_dataloader(self):
            return dl_train
        
        def val_dataloader(self):
            return dl_val
    
    
    ebmnlp = EBMNLPTagger(config, bioelmo, nn.Linear(MAX_LENGTH, NUM_TAGS), crf, ID_TO_LABEL)
    device = torch.device(f'cuda:{config.cuda}')
    ebmnlp.to(device)
    trainer = pl.Trainer(
        max_epochs=50,
        train_percent_check=1,
    )
    trainer.fit(ebmnlp)




if __name__=='__main__':
    def get_args():
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('-v', '--version', dest='version', help='Experiment Name')
        parser.add_argument('-l', '--lr', dest='lr', help='Learning Rate')
        parser.add_argument('-w', '--wd', '--weight-decay', dest='weight_decay', help='Weight Decay')
        parser.add_argument('-c', '--cuda', dest='cuda', help='CUDA Device Number')
        args = parser.parse_args()
        return args

    config = get_args()
    main(config)
