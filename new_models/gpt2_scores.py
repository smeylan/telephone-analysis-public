import transformers
import torch

import numpy as np

import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 3/11: importlib help: https://stackoverflow.com/questions/1254370/reimport-a-module-in-python-while-interactive

import importlib
import new_models
importlib.reload(new_models)

import new_models
from new_models import new_model_funcs
from new_models.new_model_funcs import prepSentence


def get_sentence_prefixes(raw_sentence, tokenizer):
    
    """
    Note: This also does capitalization, as well adding punctuation.
    """
    
    sentence = prepSentence(raw_sentence)
    
    tokens = tokenizer.encode(tokenizer.bos_token + sentence + tokenizer.eos_token)
    prefixes = []; next_words = []
    
    for idx in range(1, len(tokens)-2):
        # Omit the last word, the punctuation, and the EOS token in the prefixes.
        # The reason to omit the last word is because no prediction will be made using the last word as the end of the prefix,
        #   because that would be predicting on punctuation.
        this_prefix = tokens[:idx]
        this_next_word = tokens[idx]
        prefixes.append(this_prefix)
        next_words.append(this_next_word)
        
    return prefixes, torch.Tensor(next_words) 
 
def get_gpt2_sentence_score(sentence, tokenizer, model):
    """
    Inputs:
        sentence, a str, the raw sentence to be scored
        tokenizer, for GPT-2.
    Outputs:
        this_sum_score, a float sum of the porbs of the ground truth words 
    """
    
    this_prefixes, this_ground_truth_tokens = get_sentence_prefixes(sentence, tokenizer)
    this_next_word_probs = get_next_word_probs(this_prefixes, tokenizer, model)
    this_probs_targets = new_model_funcs.get_next_word_probs(this_next_word_probs, this_ground_truth_tokens)
    
    this_sum_score = torch.sum(this_probs_targets).item() # This will be averaged in the main ipynb analysis.
    return this_sum_score


# The below function was generally based on/taken from the following:
# 2/26 https://huggingface.co/transformers/quickstart.html
# 2/27 information on variable-length batches: https://github.com/huggingface/transformers/issues/2001
# 2/27 Debugging help on trying to predict on end tokens by accident
#    https://github.com/huggingface/transformers/issues/3021

def get_next_word_probs(sentences, tokenizer, model):
    """
    Inputs:
        sentences, a list of str, the sentences to be analyzed.
        tokenizer, for GPT-2
    """
    token_inputs, attention_inputs, real_lengths = pad_sentences(sentences, tokenizer)

    with torch.no_grad():
        
        raw_predictions = model(input_ids = token_inputs.long(), attention_mask = attention_inputs)[0]
        
        # Gather/indexing from 2/27: https://github.com/huggingface/transformers/issues/3021
        predictions = torch.gather(raw_predictions, dim = 1, index = real_lengths.long()).squeeze()
        #end taken gather/indexing code

        next_word_probs = F.softmax(predictions, dim = -1)

    return next_word_probs

def score_sentences(sentences, model_type = '', verbose = True):
    """
    This is the actual desired return of the main ipynb notebook.
    Inputs:
        sentences, a List of str.
        model_type, one of: medium, large, x
    """
    
    
    # 2/26: https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
    # GPT2LMHeadModel returns unnormalized probabilities over the next word -- requires softmax.

    # or, gpt-2{medium, large, xl}
    # 2/26: options from here https://huggingface.co/transformers/pretrained_models.html

    model_name = f'gpt2{model_type}'
    
    print(model_name)
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    sentence_scores = []
    for idx, this_sentence in enumerate(sentences):
        if verbose and idx % 30 == 0:
            print(idx)
        sentence_scores.append(get_gpt2_sentence_score(this_sentence, tokenizer, model))
        
    return sentence_scores


# Advice on how to incorporate attention, getting the next word
#   2/26: https://stackoverflow.com/questions/62852940/how-to-get-immediate-next-word-probability-using-gpt2-model
# For tokenizer usage
#   2/26: https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
# GPT2LMHeadModel returns unnormalized probabilities over the next word -- requires softmax.

def pad_sentences(sentence_tokens, tokenizer):
    """
    Inputs:
        sentences, a tokenized tensor of all prefixes of the sentence
        tokenizer, for GPT-2
    Outputs:
        new_tokens, a version of sentences such that all sentences
            are padded to the max sentence length in the data.
        attentions, a Tensor (batch, max sentence length) of attention masks (1 for all tokens except padding ones)
        real_lengths, the actual length of each input (excluding end padding)
    """
    
    #assert False, ' Need to fix this so it merges with the pre-encoding happening in the prefix code. '
    max_sentence_len = max(map(len, sentence_tokens))
    
    new_tokens = []; attention_arrs = []
    
    eos_int_val = tokenizer.encode(tokenizer.eos_token)[0]
    
    for sentence in sentence_tokens:
        
        new_sentence = sentence[:]
        diff = max_sentence_len - len(sentence)
        
        if diff > 0:
            new_sentence.extend([eos_int_val for _ in range(diff)])
        new_tokens.append(new_sentence) # It seems that padding is not occuring?
        
        this_attention_arr = torch.ones((len(sentence),))
        if diff > 0:
            pad_shape = (diff,)
            this_attention_arr = torch.hstack([this_attention_arr, torch.zeros(pad_shape)])
        attention_arrs.append(this_attention_arr)
        
    new_tokens = torch.Tensor(new_tokens)
    attentions = torch.stack(attention_arrs)

    
    # 2/27: https://github.com/huggingface/transformers/issues/3021
    real_lengths = torch.Tensor([len(sentence) - 1 for sentence in sentence_tokens])
    real_lengths = real_lengths.view(-1, 1).repeat(1, 50257).unsqueeze(1).int()
    # end directly taken code
    
    return new_tokens, attentions, real_lengths
   