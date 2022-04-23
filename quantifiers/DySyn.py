import pandas as pd
import numpy as np
from quantifiers.DyS import DyS
import qnt_utils as qntu
import pdb

"""DySyn is a member of the DyS framework that uses synthetic scores to find best match with the test scores distribution.
 
Parameters
----------
test_scores : array
    An array containing the score estimated for the positive class from each test set instance.
Returns
-------
float
    A float value representing the predicted label distribution for positive instances. 
 """
def DySyn(test_scores):
     
    MF = [0.1,0.3,0.5,0.7,0.9] #These are the mergim factor used to search the best match
    result  = []
    for mfi in MF:        
        scores = qntu.MoSS(1000, 0.5, mfi)        
        prop = DyS(scores[scores['label']=='1']['score'], scores[scores['label']=='2']['score'], test_scores, 'topsoe')
        result.append(prop)                                           
                        
    pos_prop = round(np.median(result),2)
    print(pos_prop)
    return pos_prop