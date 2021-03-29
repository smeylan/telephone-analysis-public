import transformers
import pandas as pd

from transformers import BartForConditionalGeneration, BartTokenizer
import torch

from new_models import new_model_funcs, bertlike_funcs
from new_models.new_model_funcs import prepSentence

import torch.nn.functional as F

def get_positions_from_encoded(raw_tokens, mask_token_id):
    ## Similar to the BERT function of the same name
    
    all_range = list(range(raw_tokens.shape[-1]))
    this_range = all_range[1:-2] + all_range[-1:]
    # For consistency, this is the "single_word" range, even though only sentence scoring will be used.
        
    extract_pos_arr = torch.Tensor(this_range)
    num_repeats = len(this_range)
    
    tokens = raw_tokens.clone().repeat(num_repeats, 1)
    next_words = torch.zeros((num_repeats,))
    
    for prediction_idx, word_idx in enumerate(this_range):

        old_word = raw_tokens[0][word_idx].clone()
        tokens[prediction_idx][word_idx] = mask_token_id

        next_words[prediction_idx] = old_word 

    return tokens, next_words, extract_pos_arr

def get_bart_probabilities(sentence, tokenizer, model, verifying = False):
    
    mask_token_int = tokenizer(tokenizer.mask_token)['input_ids'][-2] # Need to extract this
    
    assert tokenizer.decode(mask_token_int) == '<mask>', 'Mask integer form was incorrect.'
    
    sentence = new_model_funcs.prepSentence(sentence)
    
    # 3/27: Tokenizing reference https://huggingface.co/transformers/model_doc/bart.html
    batch = tokenizer(sentence, return_tensors = 'pt')['input_ids']
    
    # Need to generate the prediction positions for the "word-based" prediction only, since sentence scoring from model will be discontinued.
    # Note that BART is really only used for sentence scoring -- also, will start filtering punctuation in the df processing.
    
    this_tokens, this_next_words, this_extract_pos_arr = get_positions_from_encoded(batch, mask_token_int)
    # 3/27: LM reference https://huggingface.co/transformers/model_doc/bart.html#barttokenizer
    
    with torch.no_grad():
        all_position_logits = model.forward(this_tokens)['logits']
        all_position_probs = F.softmax(all_position_logits, dim = -1)

    select_position_probs = bertlike_funcs.select_prediction_position(all_position_probs, 
                                                                      this_extract_pos_arr)
    probs = new_model_funcs.get_next_word_probs(select_position_probs, this_next_words)
    
    return (probs, this_next_words) if not verifying else (probs, this_next_words, select_position_probs)

def get_bart_word_score(sentence, tokenizer, model):
    
    probs, next_words = get_bart_probabilities(sentence, tokenizer, model)
    df = new_model_funcs.word_probs_to_df(probs, next_words, tokenizer)
   
    return df


def score_inputs(sentences): 
    """
    Note that this scores by "word", but BART doesn't have whole word tokenization.
    The "word" probabilities will be processed in sentences via the code that does this for BERT, GPT-2.
    """
    
    # 3/27: https://huggingface.co/transformers/model_doc/bart.html
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    all_scores = []
    for idx, sentence in enumerate(sentences):
        if idx % 100 == 0:
            print(f'Index: {idx}')
        all_scores.append(get_bart_word_score(sentence, tokenizer, model))
    
    return all_scores
    