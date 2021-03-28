import transformers
import pandas as pd

from transformers import BartForConditionalGeneration, BartTokenizer
import torch

from new_models import new_model_funcs
from new_models.new_model_funcs import prepSentence

import torch.nn.functional as F


def score_inputs(): 
    
    """
    "Word"-only -- note that 
    """
    
    