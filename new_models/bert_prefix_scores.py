from transformers import BertForMaskedLM, BertTokenizer
import torch

from new_model_funcs import prepSentence
#from new_models.new_model_funcs import prepSentence (for running from the main ipynbs)

import torch.nn.functional as F

def get_encoded_text(sentence, tokenizer):
    
    # 2/20: https://albertauyeung.github.io/2020/06/19/bert-tokenization.html
    # 2/20: https://huggingface.co/transformers/quickstart.html
    # 2/20: https://stackoverflow.com/questions/61708486/whats-difference-between-tokenizer-encode-and-tokenizer-encode-plus-in-hugging#:~:text=and%20the%20description%20of%20encode_plus,if%20a%20max_length%20is%20specified.
    # How to encode sentence.

    # 2/20: https://yashuseth.blog/2019/06/12/bert-explained-faqs-understand-bert-working/
    # On paired/unpaired outputs

    sentence = prepSentence(sentence)
    inputs = tokenizer.encode_plus(text=sentence, return_tensors = 'pt')

    #2/20: https://huggingface.co/transformers/quickstart.html

    sentence_input, segments_ids = inputs['input_ids'], inputs['token_type_ids']
    return sentence_input, segments_ids


def get_prefixes_from_encoded(tokens, segments_ids):
    
    next_word_reps = []; next_words = []
    for idx in range(1, len(tokens)-2):
        
        # Omit the last word, the punctuation, and the EOS token in the prefixes.
        # The reason to omit the last word is because no prediction will be made using the last word as the end of the prefix,
        #   because that would be predicting on punctuation.
        this_prefix = tokens[:idx]
        this_segments_ids = segments_ids[:idx]
        this_next_word = tokens[idx]
        
        next_word_reps.append(this_prefix, this_segment_ids)
        next_words.append(this_next_word)
        
    return next_word_reps, next_words

def get_bert_sentence_score(sentence, tokenizer, model):
    
    this_sentence_input, this_segment_ids = get_encoded_text(sentence, tokenizer)
    this_logits = get_logits(this_sentence_input, this_segment_ids, model)
    
    return this_logits
    
def score_sentences(sentences):
    
    #2/20: https://huggingface.co/transformers/quickstart.html

    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    
    #2/20: https://albertauyeung.github.io/2020/06/19/bert-tokenization.html
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
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

def report_mask_words(logits, mask_idx, sentence):

    last_logits = logits[0][mask_idx]
    last_softmax = F.softmax(last_logits)

    softmax_vals, softmax_idxs = torch.sort(last_softmax, descending = True)
    words = preprocessor.convert_ids_to_tokens(softmax_idxs)

    print(f"Reporting most likely tokens to complete '{sentence}' in descending order")

    num_report = 20

    softmax_df = pd.DataFrame.from_dict({
      'Word': words,
      'Softmax value': list(map(lambda x : round(x, 5), softmax_vals.numpy().tolist()))
      })

    return softmax_df[:num_report], softmax_vals
