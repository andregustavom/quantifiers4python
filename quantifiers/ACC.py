import pandas as pd
import numpy as np

def ACC(test_scores, tprfpr, thr = 0.5):
    
    count = len([i for i in test_scores if i >= thr])
    cc_ouput = round(count/len(test_scores),2)
    TprFpr = tprfpr[tprfpr['threshold'] == thr]
    diff_tpr_fpr = (float(TprFpr['tpr']) - float(TprFpr['fpr']))
    
    pos_prop = (cc_ouput - float(TprFpr['fpr'])) / diff_tpr_fpr
     
    if pos_prop <= 0:                           #clipping the output between [0,1]
        pos_prop = 0
    elif pos_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = pos_prop

    return pos_prop
