import numpy as np



"""SORD

It quantifies events based on their scores applying the framework DyS with the Sample ORD Dissimilarity (SORD) proposed by Maletzke et al.(2019).

Parameters
----------
test : array
    A numeric vector of scores predicted from the test set.
    
p_score : array
    A numeric vector of positive scores estimated either from a validation set or from a cross-validation method.

n_score : array
    A numeric vector of negative scores estimated either from a validation set or from a cross-validation method.
    
Returns
-------
array
    the class distribution of the test.
"""
def SORD(pos_scores, neg_scores,test_scores):
    
    #alpha = np.arange(0,1,0.01)  
    alpha = np.linspace(0,1,101)
    sc_1  = pos_scores
    sc_2  = neg_scores
    ts    = test_scores
    
    vDist   = []
    
    for k in alpha:        
        pos = np.array(sc_1)
        neg = np.array(sc_2)
        test = np.array(ts)
        pos_prop = k        
        
        p_w = pos_prop / len(pos)
        n_w = (1 - pos_prop) / len(neg)
        t_w = -1 / len(test)

        p = list(map(lambda x: (x, p_w), pos))
        n = list(map(lambda x: (x, n_w), neg))
        t = list(map(lambda x: (x, t_w), test))

        v = sorted(p + n + t, key = lambda x: x[0])

        acc = v[0][1] 
        total_cost = 0

        for i in range(1, len(v)):
            cost_mul = v[i][0] - v[i - 1][0] 
            total_cost = total_cost + abs(cost_mul * acc)
            acc = acc + v[i][1]

        vDist.append(total_cost)        
        
    pos_prop = alpha[vDist.index(min(vDist))]
    
    return pos_prop
