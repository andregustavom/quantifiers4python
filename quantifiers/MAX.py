import pandas as pd
import numpy as np

def MAX(test_scores, tprfpr):
    
    diff_tpr_fpr = list(abs(tprfpr['tpr'] - tprfpr['fpr']))
    max_index = diff_tpr_fpr.index(max(diff_tpr_fpr))         #Finding index where (tpr-fpr) is maximum
    threshold, fpr, tpr = tprfpr.loc[max_index]            #taking threshold,tpr and fpr where (tpr - fpr) is maximum
    
    class_prop = len(np.where(test_scores >= threshold)[0])/len(test_scores)
    
    pos_prop = round(abs(class_prop - fpr)/abs(tpr - fpr),2)   #adjusted class proportion
  
    if pos_prop <= 0:                           #clipping the output between [0,1]
        pos_prop = 0
    elif pos_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = pos_prop

    return pos_prop
