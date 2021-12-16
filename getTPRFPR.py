import pandas as pd
import numpy as np

def getTPRFPR(scores):
    unique_scores = np.arange(0.01,1,0.01)
    TprFpr = pd.DataFrame(columns=['threshold','fpr', 'tpr'])
    total_positive = len(scores[scores['class']==1])
    total_negative = len(scores[scores['class']==0])
    for threshold in unique_scores:
        fp = len(scores[(scores['scores'] > threshold) & (scores['class']==0)])
        tp = len(scores[(scores['scores'] > threshold) & (scores['class']==1)])
        tpr = round(tp/total_positive,2)
        fpr = round(fp/total_negative,2)
    
        aux = pd.DataFrame([[round(threshold,2), fpr, tpr]])
        aux.columns = ['threshold', 'fpr', 'tpr']    
        TprFpr = pd.concat([TprFpr, aux])

    return TprFpr

