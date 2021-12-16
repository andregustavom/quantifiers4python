import pandas as pd
import numpy as np

def CC(test_scores,thr=0.5):
    
    count = len([i for i in test_scores if i >= thr])
    
    pos_prop = round(count/len(test_scores),2)
    num_predict = count
    
    return pos_prop





