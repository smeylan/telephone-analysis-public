# Defines the specific scoring functions for each model.

# For both prefix generation scenarios, omit index = 0, that is, [CLS]

from new_models import model_prefixes, model_score_utils
import transformers

from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertForMaskedLM, BertTokenizer, BartForConditionalGeneration, BartTokenizer

import importlib
importlib.reload(model_prefixes)
importlib.reload(model_score_utils)



def get_gpt2_modules(model_type = ''):
    
    model_name = f'gpt2{model_type}'
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer, model_prefixes.get_gpt2_prefixes


def get_bert_modules():
    
    #2/20: https://huggingface.co/transformers/quickstart.html
    #2/20: https://albertauyeung.github.io/2020/06/19/bert-tokenization.html
    
    model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')
    model.eval()
   
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking")
    
    get_bert_masks = model_prefixes.get_bertlike_mask_func(tokenizer)
    
    return model, tokenizer, get_bert_masks
    
    
def get_bart_modules():
    
    # 3/27: https://huggingface.co/transformers/model_doc/bart.html
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model.eval()
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    get_bart_masks = model_prefixes.get_bertlike_mask_func(tokenizer)
    
    return model, tokenizer, get_bart_masks
    
#########################
###### PRIMARY SCORING ##
#########################

def get_gpt2_scores(sentences, model_type = '', verbose = True):
    
    # 2/26: https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
    # 2/26: options from here https://huggingface.co/transformers/pretrained_models.html

    
    model, tokenizer, gpt2_prefix_func = get_gpt2_modules(model_type)
    scores = model_score_utils.score_inputs(sentences, model, tokenizer,
                                            gpt2_prefix_func, verbose = verbose)
        
    return scores


def get_bert_scores(sentences, verbose = True):

    
    model, tokenizer, get_bert_masks = get_bert_modules()
    scores = model_score_utils.score_inputs(sentences, model, tokenizer, get_bert_masks, verbose = verbose)
    
    return scores
    
    
def get_bart_scores(sentences, verbose = True):
    
    model, tokenizer, get_bart_masks = get_bart_modules()
    scores = model_score_utils.score_inputs(sentences, model, tokenizer, get_bart_masks, verbose = verbose)
    
    return scores

    