import pandas as pd
import numpy as np

def getTPRandFPRbyThreshold (validation_scores):
    unique_scores = np.arange(0,1,0.01)
    arrayOfTPRandFPRByTr = pd.DataFrame(columns=['threshold','fpr', 'tpr'])
    total_positive = len(validation_scores[validation_scores['class']==1])
    total_negative = len(validation_scores[validation_scores['class']==0])
    for threshold in unique_scores:
        fp = len(validation_scores[(validation_scores['score'] > threshold) & (validation_scores['class']==0)])
        tp = len(validation_scores[(validation_scores['score'] > threshold) & (validation_scores['class']==1)])
        tpr = round(tp/total_positive,2)
        fpr = round(fp/total_negative,2)
    
        aux = pd.DataFrame([[round(threshold,2), fpr, tpr]])
        aux.columns = ['threshold', 'fpr', 'tpr']    
        arrayOfTPRandFPRByTr = pd.concat([arrayOfTPRandFPRByTr, aux])

    return arrayOfTPRandFPRByTr