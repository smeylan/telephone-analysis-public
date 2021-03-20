import transformers
import pandas as pd

from transformers import BertForMaskedLM, BertTokenizer
import torch

import new_model_funcs
#from new_models.new_model_funcs import prepSentence (for running from the main ipynbs)

import importlib
importlib.reload(new_model_funcs)

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


def get_positions_from_encoded(raw_tokens, segments_ids, mask_token_id):
    
    assert raw_tokens.shape[0] == 1, "Tokens has non-1 batch dimension. This will break assumptions about [0] in this function."
    
    num_repeats = raw_tokens.shape[1] - 3
    # Omit the last word, the punctuation, and the EOS token in the prefixes.
    # The reason to omit the last word is because no prediction will be made using the last word as the end of the prefix,
    #   because that would be predicting on punctuation.
 
    tokens = raw_tokens.clone().repeat(num_repeats, 1)
    
    new_segment_ids = segments_ids.clone().repeat(num_repeats, 1)
    
    next_words = torch.zeros((num_repeats,))
    
    for idx in range(num_repeats): 
        
        word_idx = idx + 1
        # Above: In the raw_tokens array. Start after CLS and read until before the added punctuation.
        
        old_word = raw_tokens[0][word_idx].clone()
        tokens[idx][word_idx] = mask_token_id
        
        next_words[idx] = old_word
        
    return tokens, new_segment_ids, next_words

def select_prediction_position(this_probs):
    """
    Selects the softmax corresponding to the word position prediction of interest.
    Inputs:
        this_probs, a (prediction position or batch, language model prediction positions, vocabulary size) Tensor
    Outputs:
        softmax_targets,
            (Returns an ascending dim = 1 slice for all of the positions
                at which a prediction of interest is being made, for [0, num_repeats))
        shape: (batch, vocab size)
    Note that the omission of positions is taken care of in the original generation of the "batch".
    """
    
    extract_word_pos = torch.Tensor(list(range(this_probs.shape[0]))) + 1 # Omit the following positions as "next word": CLS, ".", EOS
    
    # Gather/indexing from 2/27: https://github.com/huggingface/transformers/issues/3021
    extract_word_pos = extract_word_pos.view(-1, 1).repeat(1, this_probs.shape[-1]).unsqueeze(1).long()
    softmax_targets = torch.gather(this_probs, dim = 1, index = extract_word_pos).squeeze()
    
    return softmax_targets
    
def get_bert_sentence_score(sentence, tokenizer, model, verifying = False):
    """
    Verifying is just for use in some sanity checks.
    """
    
    mask_token_int = get_encoded_text("[MASK]", tokenizer)[0][0, -3:-2]
    assert tokenizer.convert_ids_to_tokens(mask_token_int) == ['[MASK]'], 'Incorrect mask ID token assigned.'
    
    this_single_sentence, this_single_segment = get_encoded_text(sentence, tokenizer)
    
    this_sentence_input, this_segment_ids, this_next_words = get_positions_from_encoded(this_single_sentence, this_single_segment, mask_token_int)
    
    this_logits = get_logits(this_sentence_input, this_segment_ids, model)
        
    # Omit the non-predicting words from the analysis.
    this_next_words = this_next_words.view(-1, 1).long()
    
    this_probs = F.softmax(this_logits, dim = -1)
    this_probs = select_prediction_position(this_probs)
    
    surprisals = new_model_funcs.get_next_word_surprisal(this_probs, this_next_words)
    
    this_sum_score = torch.sum(surprisals).item() # This will be averaged in the main ipynb analysis.
   
    return this_sum_score if not verifying else (this_sum_score, this_probs)
    
def score_sentences(sentences):
    
    #2/20: https://huggingface.co/transformers/quickstart.html

    model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')
    model.eval()
    
    #2/20: https://albertauyeung.github.io/2020/06/19/bert-tokenization.html
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking")
    
    all_scores = []
    for sentence in sentences:
        all_scores.append(get_bert_sentence_score(sentence, tokenizer, model))
    
    return all_scores
                          
def get_logits(sentence_input, segments_ids, model):
    #2/20: logits vs. softmax: https://jamesmccaffrey.wordpress.com/2020/06/11/pytorch-crossentropyloss-vs-nllloss-cross-entropy-loss-vs-negative-log-likelihood-loss/
    
    with torch.no_grad():
        outputs = model(input_ids = sentence_input, token_type_ids = segments_ids)
        logits = outputs[0]

    return logits

def decode_token_list(tokens, tokenizer):
    result = tokenizer.convert_ids_to_tokens(tokens)
    return result

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
