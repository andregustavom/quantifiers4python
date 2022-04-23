import numpy as np
import qnt_utils as qntu

def DyS(pos_scores, neg_scores, test_scores, measure='topose'):
    
    bin_size = np.linspace(2,20,10)  #[10,20] range(10,111,10) #creating bins from 2 to 10 with step size 2
    bin_size = np.append(bin_size, 30)
    
    result  = []
    for bins in bin_size:
        #....Creating Histograms bins score\counts for validation and test set...............
        
        p_bin_count = qntu.getHist(pos_scores, bins)
        n_bin_count = qntu.getHist(neg_scores, bins)
        te_bin_count = qntu.getHist(test_scores, bins)
        
        def f(x):            
            return(qntu.DyS_distance(((p_bin_count*x) + (n_bin_count*(1-x))), te_bin_count, measure = measure))
    
        result.append(qntu.TernarySearch(0, 1, f))                                           
                        
    pos_prop = round(np.median(result),2)
    return pos_prop
    


    
        




