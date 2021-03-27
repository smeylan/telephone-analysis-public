import transformers
import pandas as pd

from transformers import BertForMaskedLM, BertTokenizer
import torch

from new_models import new_model_funcs
from new_models.new_model_funcs import prepSentence

import torch.nn.functional as F

def get_encoded_text(sentence, tokenizer):
    
    # 2/20: https://albertauyeung.github.io/2020/06/19/bert-tokenization.html
    # 2/20: https://huggingface.co/transformers/quickstart.html
    # 2/20: https://stackoverflow.com/questions/61708486/whats-difference-between-tokenizer-encode-and-tokenizer-encode-plus-in-hugging#:~:text=and%20the%20description%20of%20encode_plus,if%20a%20max_length%20is%20specified.
    # How to encode sentence.

    # 2/20: https://yashuseth.blog/2019/06/12/bert-explained-faqs-understand-bert-working/
    # On paired/unpaired outputs

    sentence = new_model_funcs.prepSentence(sentence)
    inputs = tokenizer.encode_plus(text=sentence, return_tensors = 'pt')

    #2/20: https://huggingface.co/transformers/quickstart.html

    sentence_input, segments_ids = inputs['input_ids'], inputs['token_type_ids']
    return sentence_input, segments_ids


def get_positions_from_encoded(raw_tokens, segments_ids, mask_token_id, mode):
    
    assert raw_tokens.shape[0] == 1, "Tokens has non-1 batch dimension. This will break assumptions about [0] in this function."
    assert new_model_funcs.valid_mode(mode)
    
    all_range = list(range(raw_tokens.shape[-1]))
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
    
    extract_pos_arr = torch.Tensor(ranges[mode])
 
    num_repeats = len(ranges[mode])
    
    tokens = raw_tokens.clone().repeat(num_repeats, 1)
    
    new_segment_ids = segments_ids.clone().repeat(num_repeats, 1)
    
    next_words = torch.zeros((num_repeats,))
    

    for prediction_idx, word_idx in enumerate(ranges[mode]):
    
        old_word = raw_tokens[0][word_idx].clone()
        tokens[prediction_idx][word_idx] = mask_token_id
        
        next_words[prediction_idx] = old_word 
        
    return tokens, new_segment_ids, next_words, extract_pos_arr

def select_prediction_position(this_probs, extract_word_pos):
    """
    Selects the softmax corresponding to the word position prediction of interest.
    Inputs:
        this_probs, a (prediction position or batch, language model prediction positions, vocabulary size) Tensor
        extract_word_pos, the position in the original sequence to extract from (the location of the masked word)
    Outputs:
        softmax_targets,
            (Returns an ascending dim = 1 slice for all of the positions
                at which a prediction of interest is being made, for [0, num_repeats))
        shape: (batch, vocab size)
    Note that the omission of positions is taken care of in the original generation of the "batch".
    """
    
    # Gather/indexing from 2/27: https://github.com/huggingface/transformers/issues/3021
    extract_word_pos = extract_word_pos.view(-1, 1).repeat(1, this_probs.shape[-1]).unsqueeze(1).long()
    
    softmax_targets = torch.gather(this_probs, dim = 1, index = extract_word_pos).squeeze()
    
    return softmax_targets
   
    
def get_bert_probabilities(sentence, tokenizer, model, mode, verifying = False):
    
    mask_token_int = get_encoded_text("[MASK]", tokenizer)[0][0, -3:-2]
    assert tokenizer.convert_ids_to_tokens(mask_token_int) == ['[MASK]'], 'Incorrect mask ID token assigned.'
    
    this_single_sentence, this_single_segment = get_encoded_text(sentence, tokenizer)
    
    this_sentence_input, this_segment_ids, this_next_words, extract_pos_arr = get_positions_from_encoded(this_single_sentence, this_single_segment, mask_token_int, mode)
    
    this_logits = get_logits(this_sentence_input, this_segment_ids, model)
            
    this_probs = F.softmax(this_logits, dim = -1)
    this_probs = select_prediction_position(this_probs, extract_pos_arr)
    
    probs = new_model_funcs.get_next_word_probs(this_probs, this_next_words)
    
    return (probs, this_next_words) if not verifying else (probs, this_next_words, this_probs)
    
    
def get_bert_sentence_score(sentence, tokenizer, model, verifying = False):
    """
    Verifying is just for use in some sanity checks.
    Removed some of the code for use in verifications (i.e. debugging)
    
    Note: This computes a probability, NOT a log10(Pr[word]).
    """
    
    probs, _, this_probs = get_bert_probabilities(sentence, tokenizer, model, 'sentence', verifying)
    this_prod_score = torch.prod(probs).item() # This will be averaged in the main ipynb analysis.
   
    return this_prod_score if not verifying else (this_sum_score, this_probs)

def get_bert_word_score(sentence, tokenizer, model):
    
    probs, next_words = get_bert_probabilities(sentence, tokenizer, model, 'single_word')
    df = new_model_funcs.word_probs_to_df(probs, next_words, tokenizer)
    
    return df

def score_inputs(sentences, mode):
    
    assert new_model_funcs.valid_mode(mode)
    mode_score_func = get_bert_word_score if mode == 'single_word' else get_bert_sentence_score
    
    #2/20: https://huggingface.co/transformers/quickstart.html
    
    model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')
    model.eval()
    
    #2/20: https://albertauyeung.github.io/2020/06/19/bert-tokenization.html
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking")
    
    all_scores = []
    for idx, sentence in enumerate(sentences):
        if idx % 50 == 0:
            print(f'Index: {idx}')
        all_scores.append(mode_score_func(sentence, tokenizer, model))
    
    return all_scores
                          
def get_logits(sentence_input, segments_ids, model):
    #2/20: logits vs. softmax: https://jamesmccaffrey.wordpress.com/2020/06/11/pytorch-crossentropyloss-vs-nllloss-cross-entropy-loss-vs-negative-log-likelihood-loss/
    
    with torch.no_grad():
        outputs = model(input_ids = sentence_input, token_type_ids = segments_ids)
        logits = outputs[0]

    return logits

def report_mask_words(scores, sentence, tokenizer):
    """
    raw_scores = a (vocabulary,) tensor of selected softmax values for a pre-selected position.
    mask_idx, the position to select for analysis.
    
    sentence = the prefix to do the prediction on
    tokenizer = BERT tokenizer
    """
    
    # It should intake the raw scores itself.
    score_vals, word_idxs = torch.sort(scores, descending = True)
    
    words = tokenizer.convert_ids_to_tokens(word_idxs)

    print(f"Reporting most likely tokens to complete '{sentence}' in descending order")

    num_report = 20

    score_df = pd.DataFrame.from_dict({
      'Word': words,
      'Score value': list(map(lambda x : round(x, 5), score_vals.numpy().tolist()))
      })

    return score_df[:num_report], score_vals
