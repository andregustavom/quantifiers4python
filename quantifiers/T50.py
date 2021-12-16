import pandas as pd
import numpy as np


def T50(test_scores, tprfpr):
       
    index = np.abs(tprfpr['tpr'] - 0.5).idxmin()      #taking threshold tpr and fpr where tpr=50% or (tpr - 50%) is minimum
   
    threshold, fpr, tpr = tprfpr.loc[index]            
    
    class_prop = len(np.where(test_scores >= threshold)[0])/len(test_scores)
    
    pos_prop = round(abs(class_prop - fpr)/abs(tpr - fpr),2)   #adjusted class proportion
  
    if pos_prop <= 0:                           #clipping the output between [0,1]
        pos_prop = 0
    elif pos_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = pos_prop

    return pos_prop
