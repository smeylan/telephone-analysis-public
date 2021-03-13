# Run the commands in the bert_scores.ipynb version first

from new_models import new_model_funcs
from new_models.new_model_funcs import prepSentence

# 2/26 https://github.com/awslabs/mlm-scoring
import os

## 2/27 Installation code: https://github.com/awslabs/mlm-scoring

import torch
import torch.nn.functional as F

import transformers

# Code taken directly from 2/26 https://github.com/awslabs/mlm-scoring

true_dir = os.getcwd()
os.chdir('./mlm-scoring')

import mlm

from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx

def score_sentences(sentences):

    # TODO: Re-enable this, disabled for softmax discrepancy tracking
    #sentences = list(map(prepSentence, sentences))
    ctxs = [mx.cpu()]

    # MXNet MLMs (use names from mlm.models.SUPPORTED_MLMS)
    model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-en-cased')
    
    print('scoring started')
    scorer = MLMScorer(model, vocab, tokenizer, ctxs)

    return list(map(lambda x : x * -1, scorer.score_sentences(sentences))) # Note that scoring sentences with punctuation...?

    #End directly taken code
    
os.chdir(true_dir
