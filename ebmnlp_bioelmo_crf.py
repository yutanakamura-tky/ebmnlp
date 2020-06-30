#!/usr/bin/env python
# coding: utf-8

# # 0. Preparation

# ### 0-1. Dependencies

import argparse
import itertools
import logging
import math
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

from transformers import BertConfig, BertForPreTraining, BertTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from seqeval.metrics import accuracy_score as seq_accuracy_score
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import classification_report as seq_classification_report

from tqdm import tqdm_notebook as tqdm
import pytorch_lightning as pl

from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG


# ### 0-1. Hyperparameters
def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--debug', '--debug-mode', action='store_true', dest='debug_mode', help='Set this option for debug mode')
    parser.add_argument('-d', '--dir', '--data-dir', dest='data_dir', type=str, default='./official/ebm_nlp_1_00', help='Data Directory')
    parser.add_argument('--model', default='bioelmo', dest='model', type=str, help='bioelmo or biobert')
    parser.add_argument('--bioelmo-dir', dest='bioelmo_dir', type=str, default='./models/bioelmo', help='BioELMo Directory')
    parser.add_argument('--biobert-dir', dest='biobert_dir', type=str, default='./models/bioelmo/biobert/biobert_v1.0_pubmed_pmc', help='BioBERT Directory')
    parser.add_argument('-v', '--version', dest='version', type=str, help='Experiment Name')
    parser.add_argument('-e', '--max-epochs', dest='max_epochs', type=int, default='15', help='Max Epochs (Default: 15)')
    parser.add_argument('--max-length', dest='max_length', type=int, default='1024', help='Max Length (Default: 1024)')
    parser.add_argument('-l', '--lr', dest='lr', type=float, default='1e-2', help='Learning Rate (Default: 1e-2)')
    parser.add_argument('--fine-tune-bioelmo', action='store_true', dest='fine_tune_bioelmo', help='Whether to Fine Tune BioELMo')
    parser.add_argument('--lr-bioelmo', dest='lr_bioelmo', type=float, default='1e-4', help='Learning Rate in BioELMo Fine-tuning')
    parser.add_argument('--fine-tune-biobert', action='store_true', dest='fine_tune_biobert', help='Whether to Fine Tune BioELMo')
    parser.add_argument('--lr-biobert', dest='lr_biobert', type=float, default='2e-5', help='Learning Rate in BioELMo Fine-tuning')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default='16', help='Batch size (Default: 16)')
    parser.add_argument('-c', '--cuda', dest='cuda', default=None, help='CUDA Device Number')
    parser.add_argument('-r', '--random-state', dest='random_state', type=int, default='42', help='Random state (Default: 42)')
    namespace = parser.parse_args()
    return namespace


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


# ### 1. Dataset, DataLoader

class EBMNLPDataset(torch.utils.data.Dataset):
    def __init__(self, text_files, token_files, p_files, i_files, o_files):
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
            returns['tokens'] = tokens
    
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
        returns['tags'] = tag.numpy().tolist()

        return returns


class EBMNLPDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        """
        text_files: list(str)
        token_files: list(str)
        p_files: list(str)
        i_files: list(str)
        o_files: list(str)
        """
        kwargs['collate_fn'] = lambda batch: {
            'pmid' : torch.tensor([sample['pmid'] for sample in batch]),
            'text' : [sample['text'] for sample in batch],
            'tokens' : [sample['tokens'] for sample in batch],
            'tags' : [sample['tags'] for sample in batch]
        }
        super().__init__(dataset, **kwargs)



# 2. LightningModule

class EBMNLPTagger(pl.LightningModule):
    def __init__(self, hparams): 
        """
        input:
            hparams: namespace with the following items:
                'data_dir' (str): Data Directory. default: './official/ebm_nlp_1_00'
                'bioelmo_dir' (str): BioELMo Directory. default: './models/bioelmo', help='BioELMo Directory')
                'max_length' (int): Max Length. default: 1024
                'lr' (float): Learning Rate. default: 1e-2
                'fine_tune_bioelmo' (bool): Whether to Fine Tune BioELMo. default: False
                'lr_bioelmo' (float): Learning Rate in BioELMo Fine-tuning. default: 1e-4
        """
        super().__init__()
        self.hparams = hparams
        self.itol = ID_TO_LABEL
        self.ltoi = {v:k for k,v in self.itol.items()}

        # Load Pretrained BioELMo
        DIR_ELMo = Path(str(self.hparams.bioelmo_dir))
        self.bioelmo = Elmo(DIR_ELMo / 'biomed_elmo_options.json', DIR_ELMo / 'biomed_elmo_weights.hdf5', 1, requires_grad=bool(self.hparams.fine_tune_bioelmo), dropout=0)
        self.bioelmo_output_dim = self.bioelmo.get_output_dim()

        # ELMo Padding token (In ELMo token with ID 0 is used for padding)
        VOCAB_FILE_PATH = DIR_ELMo / 'vocab.txt'
        command = shlex.split(f"head -n 1 {VOCAB_FILE_PATH}")
        res = subprocess.Popen(command, stdout=subprocess.PIPE)
        self.bioelmo_pad_token = res.communicate()[0].decode('utf-8').strip()

        # Initialize Intermediate Affine Layer 
        self.hidden_to_tag = nn.Linear(int(self.bioelmo_output_dim), len(self.itol))

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
   

    def _forward_crf(self, hidden, gold_tags_padded, crf_mask):
        """
        input:
            hidden (torch.tensor) (n_batch, seq_length, hidden_dim)
            gold_tags_padded (torch.tensor) (n_batch, seq_length)
            crf_mask (torch.bool) (n_batch, seq_length)
        output:
            result (dict)
                'log_likelihood' : torch.tensor
                'pred_tags_packed' : torch.nn.utils.rnn.PackedSequence
                'gold_tags_padded' : torch.tensor
        """
        result = {}

        if gold_tags_padded is not None:
            # Log likelihood
            log_prob = self.crf.forward(hidden, gold_tags_padded, crf_mask)
        
            # top k=1 tagging
            Y = [torch.tensor(result[0]) for result in self.crf.viterbi_tags(logits=hidden, mask=crf_mask)]
            Y = rnn.pack_sequence(Y, enforce_sorted=False)
        
            result['log_likelihood'] = log_prob
            result['pred_tags_packed'] = Y
            result['gold_tags_padded'] = gold_tags_padded
            return result

        else:
            # top k=1 tagging
            Y = [torch.tensor(result[0]) for result in self.crf.viterbi_tags(logits=hidden, mask=crf_mask)]
            Y = rnn.pack_sequence(Y, enforce_sorted=False)
            result['pred_tags_packed'] = Y
            return result



    def forward(self, tokens, gold_tags=None):
        """
        input:
            hidden (torch.tensor) (n_batch, seq_length, hidden_dim)
            gold_tags_padded (torch.tensor) (n_batch, seq_length)
            crf_mask (torch.bool) (n_batch, seq_length)
        output:
            result (dict)
                'log_likelihood' : torch.tensor
                'pred_tags_packed' : torch.nn.utils.rnn.PackedSequence
                'gold_tags_padded' : torch.tensor
        """
        # character_ids: torch.tensor(n_batch, len_max)
        character_ids = batch_to_ids(tokens)
        character_ids = character_ids[:, :self.hparams.max_length, :]
        character_ids = character_ids.to(self.get_device())

        # characted_ids -> BioELMo hidden state of the last layer & mask
        out = self.bioelmo(character_ids)

        # Turn on gradient tracking
        # Affine transformation (Hidden_dim -> N_tag)
        hidden = out['elmo_representations'][-1]
        hidden.requires_grad_()
        hidden = self.hidden_to_tag(hidden)

        crf_mask = out['mask'].to(torch.bool).to(self.get_device())

        if gold_tags is not None:
            gold_tags = [torch.tensor(seq) for seq in gold_tags]
            gold_tags_padded = rnn.pad_sequence(gold_tags, batch_first=True, padding_value=self.ltoi['O'])
            gold_tags_padded = gold_tags_padded[:, :self.hparams.max_length]
            gold_tags_padded = gold_tags_padded.to(self.get_device())
        else:
            gold_tags_padded = None

        result = self._forward_crf(hidden, gold_tags_padded, crf_mask)
        return result


    def step(self, batch, batch_nb, *optimizer_idx):
        tokens_nopad = batch['tokens']
        tags_nopad = batch['tags']

        # Negative Log Likelihood
        result = self.forward(tokens_nopad, tags_nopad)
        returns = {
            'loss':result['log_likelihood'] * (-1.0),
            'T':result['gold_tags_padded'],
            'Y':result['pred_tags_packed'],
            'I':batch['pmid']
        }
        return returns 


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


    def gather_outputs(self, outputs):
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

        returns = {
            'loss':loss,
            'T':T,
            'Y':Y,
            'I':I
        }

        return returns



    def training_step(self, batch, batch_nb, *optimizer_idx):
        # Process on individual mini-batches
        """
        (batch) -> (dict or OrderedDict)
        # Caution: key for loss function must exactly be 'loss'.
        """
        return self.step(batch, batch_nb, *optimizer_idx)
    

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
        outs = self.gather_outputs(outputs)
        loss = outs['loss']
        I = outs['I']
        Y = outs['Y']
        T = outs['T']

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
        return self.step(batch, batch_nb)

    
    
    def validation_end(self, outputs):
        """
        For single dataloader:
            outputs(list of dict) -> (dict or OrderedDict)
        For multiple dataloaders:
            outputs(list of (list of dict)) -> (dict or OrderedDict)
        """  
        outs = self.gather_outputs(outputs)
        loss = outs['loss']
        I = outs['I']
        Y = outs['Y']
        T = outs['T']
            
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
        return self.step(batch, batch_nb)


    def test_epoch_end(self, outputs):
        """
        For single dataloader:
            outputs(list of dict) -> (dict or OrderedDict)
        For multiple dataloaders:
            outputs(list of (list of dict)) -> (dict or OrderedDict)
        """
        outs = self.gather_outputs(outputs)
        loss = outs['loss']
        I = outs['I']
        Y = outs['Y']
        T = outs['T']

        get_logger(self.hparams.version).info(f'========== Test ==========')
        get_logger(self.hparams.version).info(f'Loss: {loss.item()}')
        get_logger(self.hparams.version).info(f'Entity-wise classification report\n{seq_classification_report(T, Y, 4)}')
        get_logger(self.hparams.version).info(f'Token-wise classification report\n{span_classification_report(T, Y, 4)}')

        progress_bar = {'test_loss':loss}
        returns = {'loss':loss, 'progress_bar':progress_bar}
        return returns


    def configure_optimizers(self):
        if self.hparams.fine_tune_bioelmo:
            optimizer_bioelmo_1 = optim.Adam(self.bioelmo.parameters(), lr=float(self.hparams.lr_bioelmo))
            optimizer_bioelmo_2 = optim.Adam(self.hidden_to_tag.parameters(), lr=float(self.hparams.lr_bioelmo))
            optimizer_crf = optim.Adam(self.crf.parameters(), lr=float(self.hparams.lr))
            return [optimizer_bioelmo_1, optimizer_bioelmo_2, optimizer_crf]
        else:        
            optimizer = optim.Adam(self.parameters(), lr=float(self.hparams.lr))
            return optimizer


    def train_dataloader(self):
        ds_train_val = EBMNLPDataset(*path_finder(self.hparams.data_dir)['train'])

        ds_train, _ = train_test_split(ds_train_val, train_size=0.8, random_state=self.hparams.random_state)
        dl_train = EBMNLPDataLoader(ds_train, batch_size=self.hparams.batch_size, shuffle=True)
        return dl_train


    def val_dataloader(self):
        ds_train_val = EBMNLPDataset(*path_finder(self.hparams.data_dir)['train'])

        _, ds_val = train_test_split(ds_train_val, train_size=0.8, random_state=self.hparams.random_state)
        dl_val = EBMNLPDataLoader(ds_val, batch_size=self.hparams.batch_size, shuffle=False)
        return dl_val


    def test_dataloader(self):
        ds_test = EBMNLPDataset(*path_finder(self.hparams.data_dir)['test'])
        dl_test = EBMNLPDataLoader(ds_test, batch_size=self.hparams.batch_size, shuffle=False)
        return dl_test



class EBMNLPBioBERTTagger(EBMNLPTagger):
    def __init__(self, hparams): 
        """
        input:
            hparams: namespace with the following items:
                'data_dir' (str): Data Directory. default: './official/ebm_nlp_1_00'
                'bioelmo_dir' (str): BioELMo Directory. default: './models/bioelmo', help='BioELMo Directory')
                'max_length' (int): Max Length. default: 1024
                'lr' (float): Learning Rate. default: 1e-2
                'fine_tune_bioelmo' (bool): Whether to Fine Tune BioELMo. default: False
                'lr_bioelmo' (float): Learning Rate in BioELMo Fine-tuning. default: 1e-4
        """
        super().__init__(hparams)

        # Load Pretrained BioBERT
        DIR_BERT = Path(str(self.hparams.biobert_dir))
        BERT_CKPT_PATH = os.path.splitext(glob(str(DIR_BERT / '*ckpt*'))[0])[0]

        self.bertconfig = BertConfig.from_pretrained('bert-base-cased')
        self.berttokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.biobert_for_pretraining = BertForPreTraining.from_pretrained('bert-base-cased')
        self.biobert_for_pretraining.load_tf_weights(self.bertconfig, BERT_CKPT_PATH)
        self.biobert = self.biobert_for_pretraining.bert
        self.biobert_pad_token = self.berttokenizer.pad_token
        self.biobert_output_dim = self.bertconfig.hidden_size

        # Initialize Intermediate Affine Layer 
        self.hidden_to_tag = nn.Linear(int(self.biobert_output_dim), len(self.itol))


    def bert_last_hidden_state_of_long_sequences(self, seqs):
        # subwords -> input_ids with [CLS], [SEP] and [PAD] tokens 
        lengths = [len(seq) for seq in seqs]

        # preparation for long sequence
        # window: length other than [CLS], [SEP]
        window = min(self.hparams.max_length, self.bertconfig.max_position_embeddings - 2)
        stride = window // 2
        n_chunks = max(0, math.ceil((max(lengths) - window) / stride)) + 1

        subword_chunks = [
            [
                seq[stride*i : stride*i + window] if len(seq[stride*i : stride*i + window]) > 0 else [''] for seq in subwords
            ]
            for i in range(n_chunks)
        ]

        # to token IDs

        def bert_encode(seqs): 
            result = {
                k : torch.tensor(v)
                for k, v in self.berttokenizer.batch_encode_plus(
                    seqs,
                    max_length=window+2,
                    is_pretokenized=True,
                    pad_to_max_length=True
                ).items()
            }
            return result

        encoded_chunks = list(map(bert_encode, subword_chunks))

        # to GPU
        encoded_chunks = list(
            map(
                lambda x:
                    {
                        k : v.to(self.get_device()) for k, v in x.items()
                    },
                encoded_chunks
            )
        )

        # to BERT
        outs = list(map(lambda chunk: self.biobert(**chunk)[0], encoded_chunks))

        # 


    def forward(self, tokens, gold_tags=None):
        """
        input:
            hidden (torch.tensor) (n_batch, seq_length, hidden_dim)
            gold_tags_padded (torch.tensor) (n_batch, seq_length)
            crf_mask (torch.bool) (n_batch, seq_length)
        output:
            result (dict)
                'log_likelihood' : torch.tensor
                'pred_tags_packed' : torch.nn.utils.rnn.PackedSequence
                'gold_tags_padded' : torch.tensor
        """
        # tokens: list(list(str))
        # convert tokens into subwords by wordpiece tokenization
        wordpiece_tokenize = lambda x: list(itertools.chain(*map(self.berttokenizer.tokenize, x)))
        subwords = list(map(wordpiece_tokenize, tokens))



        # Turn on gradient tracking
        # Affine transformation (Hidden_dim -> N_tag)
        # hidden = hogehoge
        # hidden.requires_grad_()
        # hidden = self.hidden_to_tag(hidden)

        # # crf_mask = hogehoge

        if gold_tags is not None:
            gold_tags = [torch.tensor(seq) for seq in gold_tags]
            gold_tags_padded = rnn.pad_sequence(gold_tags, batch_first=True, padding_value=self.ltoi['O'])
            gold_tags_padded = gold_tags_padded[:, :len_max, :]
            gold_tags_padded = gold_tags_padded.to(self.get_device())
        else:
            gold_tags_padded = None

        result = self._forward_crf(hidden, gold_tags_padded, crf_mask)
        return result


        # tokens: list(list(str))
        # # check if tokens have the same lengths
        len_min = min(lengths)

        # # if sequences have different lengths, pad with self.bioelmo_pad_token
        # # in addition, sequences longer than self.hparams.max_length will be truncated
        tags = [seq[:min(length, len_max)] + [int(self.ltoi['O'])] * max(0, len_max - length) for seq, length in zip(tags, lengths)]
        tags = torch.tensor(tags)

        crf_mask = torch.stack([torch.cat([torch.ones(min(length, len_max)), torch.zeros(max(0, len_max - length))]).to(bool) for length in lengths])
        
        

        # input_ids -> hidden_state
        input_ids = torch.tensor(encode['input_ids']).to(self.get_device())
        attention_mask = torch.tensor(encode['attention_mask']).to(self.get_device())
        out, _ = self.biobert(input_ids, attention_mask)
        out.requires_grad_()
        
        # hidden states of subwords must shrink
        # e.g., hidden states of 'faithful', '##ly' must be averaged to representate 'faithfully'
        

        # Affine transformation (Hidden_dim -> N_tag)
        out = self.affine(out)

        if tags is not None:
            tags = tags.to(self.get_device())
            crf_mask = crf_mask.to(self.get_device())
        
            # Log probability
            log_prob = self.crf.forward(out, tags, crf_mask)
        
            # top k=1 tagging
            Y = [torch.tensor(result[0]) for result in self.crf.viterbi_tags(logits=out, mask=crf_mask)]
            Y = rnn.pack_sequence(Y, enforce_sorted=False)
        
            return log_prob, Y

        else:
            crf_mask = crf_mask.to(self.get_device())
        
            # top k=1 tagging
            Y = [torch.tensor(result[0]) for result in self.crf.viterbi_tags(logits=out, mask=crf_mask)]
            Y = rnn.pack_sequence(Y, enforce_sorted=False)
        
            return Y


    def configure_optimizers(self):
        if self.hparams.fine_tune_biobert:
            optimizer_biobert_1 = optim.Adam(self.biobert.parameters(), lr=float(self.hparams.lr_biobert))
            optimizer_biobert_2 = optim.Adam(self.hidden_to_tag.parameters(), lr=float(self.hparams.lr_biobert))
            optimizer_crf = optim.Adam(self.crf.parameters(), lr=float(self.hparams.lr))
            return [optimizer_biobert_1, optimizer_biobert_2, optimizer_crf]
        else:        
            optimizer = optim.Adam(self.parameters(), lr=float(self.hparams.lr))
            return optimizer



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
def main():
    config = get_args()
    print(config)


    # ### 4-0. Print config
    create_logger(config.version)
    get_logger(config.version).info(config)
    
    
    # ### 4-1. DataLoader -> BioELMo -> CRF
    if config.model == 'bioelmo':
        ebmnlp = EBMNLPTagger(config)
    elif config.model == 'biobert':
        ebmnlp = EBMNLPBioBERTTagger(config)

    # ### 4-2. Training
    if config.cuda is None:
        device = torch.device('cuda')
    else:
        device = torch.device(f'cuda:{config.cuda}')

    ebmnlp.to(device)

    MODEL_CHECK_POINT_PATH = {
        'bioelmo' : './models/ebmnlp_bioelmo_crf/ebmnlp_bioelmo_crf.ckpt',
        'biobert' : './models/ebmnlp_biobert_crf/ebmnlp_biobert_crf.ckpt'
    }

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=MODEL_CHECK_POINT_PATH[config.model]
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
    main()
