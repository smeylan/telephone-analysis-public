import transformers
import torch

import numpy as np

import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 3/11: importlib help: https://stackoverflow.com/questions/1254370/reimport-a-module-in-python-while-interactive

import importlib
import new_models
importlib.reload(new_models)

from new_models import new_model_funcs
from new_models.new_model_funcs import prepSentence


def get_sentence_prefixes(raw_sentence, tokenizer, mode):
    
    """
    Note: This also does capitalization, as well adding punctuation.
    """
    
    
    assert new_model_funcs.valid_mode(mode)
    
    sentence = prepSentence(raw_sentence)
    
    tokens = tokenizer.encode(tokenizer.bos_token + sentence + tokenizer.eos_token)
    
    all_range = list(range(len(tokens)))
    ranges = {
        # These ranges indicate the index of the target word.
        'single_word': all_range[1:-2] + all_range[-1:],
        # Omit the period. 
        'sentence' : all_range[1:-2]
        # Per convention in telephone_analysis.py:
        # Omit the last word, the punctuation, and the EOS token in the prefixes.
        # The reason to omit the last word is because no prediction will be made using the last word as the end of the prefix,
        #   because that would be predicting on punctuation.
    } 

    prefixes = []; next_words = []
    
    for idx in ranges[mode]:
        this_prefix = tokens[:idx]
        this_next_word = tokens[idx]
        prefixes.append(this_prefix)
        next_words.append(this_next_word)
        
    return prefixes, torch.Tensor(next_words) 
 
    
def get_gpt2_target_word_probs(sentence, tokenizer, model, mode):
    
    """
    Inputs:
        sentence, a str, the raw sentence to be scored
        tokenizer, for GPT-2.
    Outputs:
        probs_targets, a (position to predict,) Tensor of softmax probabilities (scores)
            from ground truth words.
        this_ground_truth_tokens, a (position to predict,) Tensor of IDs of words.
    """
    
    assert new_model_funcs.valid_mode(mode)
    
    this_prefixes, this_ground_truth_tokens = get_sentence_prefixes(sentence, tokenizer, mode)
    this_next_word_probs = get_next_word_probs(this_prefixes, tokenizer, model)
    probs_targets = new_model_funcs.get_next_word_probs(this_next_word_probs, this_ground_truth_tokens)
    
    return probs_targets, this_ground_truth_tokens
    
    
def get_gpt2_sentence_score(sentence, tokenizer, model):
    """
    Inputs:
        sentence, a str, the raw sentence to be scored
        tokenizer, for GPT-2.
        model, GPT-2.
    Outputs:
        this_prod_score, a float product of the probs of the ground truth words
    This is a probability, but analysis requires log10(Pr[sentence])!
    """
    
    this_probs_targets, _ = get_gpt2_target_word_probs(sentence, tokenizer, model, mode = 'sentence')
    
    this_prod_score = torch.prod(this_scores_targets).item() # This will be averaged in the main ipynb analysis.
    return this_prod_score

def get_gpt2_word_score(sentence, tokenizer, model):
    
    """
    For a single sentence, return the Logistic Regression notebook style DF.
    Inputs:
        sentence, a str, the raw sentence to be scored
        tokenizer, for GPT-2.
        model, GPT-2.
    Outputs:
        the Logistic Regression notebook style DF.
    """
    
    this_probs_targets, this_ground_truth_tokens = get_gpt2_target_word_probs(sentence, tokenizer, model, mode = 'single_word')
    sentence_df = new_model_funcs.word_probs_to_df(this_probs_targets, this_ground_truth_tokens, tokenizer)
    
    return sentence_df
    
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

def score_inputs(sentences, mode, model_type = '', verbose = True):
    
    """
    This is the actual desired return of the main ipynb notebook.
    Inputs:
        sentences, a List of str.
        model_type, one of: medium, large, x
        mode, 
    """
    
    assert new_model_funcs.valid_mode(mode), "Mode must be either 'single_word' or 'sentence'."
    scoring_func = get_gpt2_word_score if mode == 'single_word' else get_gpt2_sentence_score
    
    # 2/26: https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
    # GPT2LMHeadModel returns unnormalized probabilities over the next word -- requires softmax.

    # or, gpt-2{medium, large, xl}
    # 2/26: options from here https://huggingface.co/transformers/pretrained_models.html

    model_name = f'gpt2{model_type}'
    
    print(model_name)
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    print(f'Scoring with mode: {mode}') 
    scores = []
    for idx, this_sentence in enumerate(sentences):
        if verbose and idx % 50 == 0:
            print(f'Index: {idx}')
        scores.append(scoring_func(this_sentence, tokenizer, model))
        
    return scores

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
   