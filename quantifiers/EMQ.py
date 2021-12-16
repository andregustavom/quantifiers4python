import pandas as pd
import numpy as np

def EMQ(p_score, n_score, test, it=5, e=None):
    test = pd.Series(test)
    
    pTr = np.array([len(p_score), len(n_score)])
    pTr = pTr/sum(pTr)
    predTe_s = pd.concat([1-test, test], axis=1)
    
    px = pd.DataFrame(predTe_s)
    pTe = pd.DataFrame(pTr)
    nE = test.shape[0]
    nC = 1
    si = 0       
    if e is None:
        while si < it:
            aux = pd.DataFrame(np.zeros((nE, nC+1)))
            auxC = []
            for ic in range(0, nC+1):
                for ie in range(0, nE):
                    numerator = (pTe.iloc[ic,si]/pTr[ic])*predTe_s.iloc[ie,ic]
                    denominator = []
                    for ic2 in range(0, nC+1):
                        denominator.append((pTe.iloc[ic2,si]/pTr[ic2])*predTe_s.iloc[ie,ic2])
                    aux.iloc[ie,ic] = numerator/sum(denominator)
                    
                auxC.append(sum(px.iloc[:,(si*2)+ic])/test.shape[0])          
                
            pTe = pd.concat([pTe, pd.DataFrame([auxC[1],auxC[0]])], axis=1)
            px = pd.concat([px, pd.DataFrame(aux)], axis=1)
            si   = si + 1
    else:
        while(np.abs(pTe.iloc[0,pTe.shape[1]-2] - pTe.iloc[0,pTe.shape[1]-1]) <= e):
            aux = pd.DataFrame(np.zeros((nE, nC+1)))
            auxC = []
            for ic in range(0, nC+1):
                for ie in range(0, nE):
                    numerator = (pTe.iloc[ic,si]/pTr[ic])*predTe_s.iloc[ie,ic]
                    denominator = []
                    for ic2 in range(0, nC+1):
                        denominator.append((pTe.iloc[ic2,si]/pTr[ic2])*predTe_s.iloc[ie,ic2])
                    aux.iloc[ie,ic] = numerator/sum(denominator)
        
                auxC.append(sum(px.iloc[:,(si*2)+ic])/test.shape[0])          
            pTe = pd.concat([pTe, pd.DataFrame([auxC[1],auxC[0]])], axis=1)
            px = pd.concat([px, pd.DataFrame(aux)], axis=1)
            si   = si + 1   

    result = pTe.iloc[0,si]
    if result < 0: 
        result = 0
    if result > 1:
        result = 1

    pos_prop= result
    
    return pos_prop
