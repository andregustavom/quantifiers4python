import pandas as pd
import numpy as np


def X(test_scores, tprfpr):
       
    min_index = (np.abs((1 - tprfpr['tpr']) - tprfpr['fpr'])).idxmin()
    threshold, fpr, tpr = tprfpr.loc[min_index]            #taking threshold,tpr and fpr where [(1 -tpr) - fpr] is minimum
    
    class_prop = len(np.where(test_scores >= threshold)[0])/len(test_scores)
    
    pos_prop = round(abs(class_prop - fpr)/abs(tpr - fpr),2)   #adjusted class proportion
  
    if pos_prop <= 0:                           #clipping the output between [0,1]
        pos_prop = 0
    elif pos_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = pos_prop

    return pos_prop
