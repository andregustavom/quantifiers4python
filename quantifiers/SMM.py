import pandas as pd
import numpy as np


def SMM(pos_scores, neg_scores,test_scores):
    
    mean_pos_scr = np.mean(pos_scores)
    mean_neg_scr = np.mean(neg_scores)  #finding mean of pos & neg scores
    
    mean_te_scr = np.mean(test_scores)              #Mean of test scores
         
    alpha =  (mean_te_scr - mean_neg_scr)/(mean_pos_scr - mean_neg_scr)     #evaluating Positive class proportion
    
    
    if alpha <= 0:   #clipping the output between [0,1]
        pos_prop = 0
    elif alpha >= 1:
        pos_prop = 1
    else:
        pos_prop = alpha
    
    pos_prop = round(pos_prop, 2)
    return pos_prop
