import os
import pandas as pd

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
    
    
def load_runs():
    return loadAndCombineRuns('output',['180419_AMT_lengthLimitedGPU.csv','180624_AMT_lengthLimitedGPU.csv'], 80)