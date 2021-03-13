
# 3/5 Below function: Taken from Dr. Meylan's telephone_analysis.py
def prepSentence(x): # Big LM expects initial capitalization and sentences end with a period
    return(' '.join([x.capitalize(),'.']))