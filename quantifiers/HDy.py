import pandas as pd
import numpy as np
import qnt_utils as qntu

def HDy(pos_scores, neg_scores,test_scores):
    
    bin_size = np.linspace(10,110,11)       #creating bins from 10 to 110 with step size 10
    alpha_values = [round(x, 2) for x in np.linspace(0,1,101)]
    
    result = []
    num_bins = []
    for bins in bin_size:
        
        p_bin_count = qntu.getHist(pos_scores, bins)
        n_bin_count = qntu.getHist(neg_scores, bins)
        te_bin_count = qntu.getHist(test_scores, bins)

        vDist = []
        
        for x in range(0,len(alpha_values),1):
            
            vDist.append(qntu.DyS_distance(((p_bin_count*alpha_values[x]) + (n_bin_count*(1-alpha_values[x]))), te_bin_count, measure="hellinger"))

        result.append(alpha_values[np.argmin(vDist)])
        
    pos_prop = round(np.median(result),2)    
    return pos_prop
    


    
        




