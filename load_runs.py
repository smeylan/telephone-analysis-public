import os
import pandas as pd

import telephone_analysis
import numpy as np

# 3/26: From the original Aggregate Chains.ipynb

# Combined analysis of Telephone Chains
# Load the output files from the individual analyses

def loadRun(runpath):
    basename = os.path.basename(runpath).replace('.csv','')
    df = pd.read_csv(runpath)
    if u'Unnamed: 0' in df.columns:
        df = df.drop([u'Unnamed: 0'], axis=1)
    df['run'] = basename
    return(df)

def loadAndCombineRuns(runPath, runnames, minItemsInChain=0):
    dfs = []
    for runname in runnames:
        dfs.append(loadRun(os.path.join(runPath, runname)))
    # !!! combine and index the runname + chain
    all_runs = pd.concat(dfs)
    all_runs['user_candidate_transcription'] = [x.lower() for x in all_runs['user_candidate_transcription']]
    all_runs['unique_chain_identifier'] = all_runs.run.map(str) + '_'+ all_runs.chain.astype('str')
    all_runs['global_chain'] = all_runs.unique_chain_identifier.astype('category').cat.codes

    # get the number of items in each chain
    chain_counts = all_runs.global_chain.value_counts().reset_index()
    retained_chains = chain_counts[chain_counts.global_chain > minItemsInChain]['index']
    print(retained_chains)
    all_runs = all_runs[all_runs.global_chain.isin(retained_chains)]    
    
    return(all_runs)
    
    
########################
#### PRIMARY FUNCS #####
########################
    
def load_runs():
    return loadAndCombineRuns('output',['180419_AMT_lengthLimitedGPU.csv','180624_AMT_lengthLimitedGPU.csv'], 80)

def postprocess_runs(filename):
    
    ## This is the code from the Aggregate notebook moved here, to always postprocess on raw data. 
    
    all_runs = pd.read_csv(filename)
    language_model_measures = ['biglm_probability', u'WSJ_Roark_Negative.Log.Probability', 
          u'BNC_KNN_unigramProb', u'BNC_KNN_trigramProb','kenlm_probability','bllip_probability',
        'bllip_wsj_probability', 'mikolov_wsj_probability','WSJ_gt_unigramProb'
        ,'WSJ_gt_trigramProb', 'WSJ_gt_5gramProb',
        'gpt2_normal_probability', 'gpt2_medium_probability', 'bert_probability', 'bart_probability'
    ]
    # normalize all language model measures by the number of words
    all_runs['length_in_words'] = [telephone_analysis.getLengthInWords(x, {}) for x in all_runs['user_candidate_transcription']]
    for x in language_model_measures:
        all_runs['normalized_'+x] = all_runs[x] / all_runs['length_in_words']
        
    # delete all columns starting with initial
    cols_to_drop = np.array(all_runs.columns.tolist())[np.argwhere(['initial_' in x for x in  all_runs.columns])].flatten()
    all_runs = all_runs.drop(labels = cols_to_drop, axis = 'columns')

    # mege them in and recompute initial per-word probabilities
    for language_model_measure in language_model_measures:    
        initialProbs = telephone_analysis.getInitialProbs(all_runs, language_model_measure) 

        if 'initial_'+language_model_measure not in all_runs.columns:
            all_runs = all_runs.merge(initialProbs, on='stimulus_id', how='outer')
            
    # # R cannot handle nans, so need to replace with nones
    for x in ['condition','flag_type','gold_candidate_transcription',
              'gold_comparison_transcription','reason', 'subject_id',
             'upstream_subject_id', 'user_comparison_transcription', 'user_short']:
        all_runs[x] = all_runs[x].fillna(value="none")
        
    all_runs['thread_id'] = all_runs['global_chain'].map(str) + '_' + [str(x) for x in all_runs['stimulus_id']]
    return all_runs

def pickle_logistic_prep(data, model_name, save_folder):
    
    # 3/27: https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type
    save_path = join(save_folder, f"{model_name}_predictions.txt")
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
        
    print(f'Saved per-word scores to {save_path}.')
       
    return data
