import pandas as pd
import numpy as np

class Distances(object):
    
    def __init__(self,P,Q):
        if sum(P)<1e-20 or sum(Q)<1e-20:
            raise "One or both vector are zero (empty)..."
        if len(P)!=len(Q):
            raise "Arrays need to be of equal sizes..."
        #use numpy arrays for efficient coding
        P=np.array(P,dtype=float);Q=np.array(Q,dtype=float)
        #Correct for zero values
        P[np.where(P<1e-20)]=1e-20
        Q[np.where(Q<1e-20)]=1e-20
        self.P=P
        self.Q=Q
        
    def sqEuclidean(self):
        P=self.P; Q=self.Q; d=len(P)
        return sum((P-Q)**2)
    
    def probsymm(self):
        P=self.P; Q=self.Q; d=len(P)
        return 2*sum((P-Q)**2/(P+Q))
    
    def topsoe(self):
        P=self.P; Q=self.Q
        return sum(P*np.log(2*P/(P+Q))+Q*np.log(2*Q/(P+Q)))
    def hellinger(self):
        P=self.P; Q=self.Q
        return 2 * np.sqrt(1 - sum(np.sqrt(P * Q)))


def DyS_distance(sc_1, sc_2, measure):
    
    dist = Distances(sc_1, sc_2)
    
    if measure == 'topsoe':
        return dist.topsoe()
    if measure == 'probsymm':
        return dist.probsymm()
    if measure == 'hellinger':
        return dist.hellinger()
    return 100


def TernarySearch(left, right, f, eps=1e-4):

    while True:
        if abs(left - right) < eps:
            return(left + right) / 2
    
        leftThird  = left + (right - left) / 3
        rightThird = right - (right - left) / 3
    
        if f(leftThird) > f(rightThird):
            left = leftThird
        else:
            right = rightThird 


def getHist(scores, nbins):
    breaks = np.linspace(0, 1, int(nbins)+1)
    breaks = np.delete(breaks, -1)
    breaks = np.append(breaks,1.1)
    
    re = np.repeat(1/(len(breaks)-1), (len(breaks)-1))  
    for i in range(1,len(breaks)):
        re[i-1] = (re[i-1] + len(np.where((scores >= breaks[i-1]) & (scores < breaks[i]))[0]) ) / (len(scores)+1)
    return re


"""This function implements MoSS. MoSS produces synthetic scores simulating distinct overlap scenarios of score distributions.
 
Parameters
----------
n : integer
    The number of scores for each label (posirive and negative).
alpha : float
    A float value ranged between [0-1], representing the positive label distribution.
m : float
    A float value representing the merging factor. When m=0, all positive observations have a score of one, while all negative 
    observations have a score of zero, turning classification perfect and, as a consequence, quantification a trivial task.
    On the other hand, when m=1, all scores for both classes are uniformly distributed within the interval [0,1], 
    turning it impossible to distinguish between classes.
Returns
-------
DataFrame
    A DataFrame containing the artificial scores for positive and negative labels. 
 """

def MoSS(n, alpha, m):
    p_score = np.random.uniform(0,1,int(n*alpha))**m
    n_score = 1-np.random.uniform(0,1,int(n*(1- alpha)))**m    
    scores  = pd.concat([pd.DataFrame(np.append(p_score, n_score)), pd.DataFrame(np.append(['1']*len(p_score), ['2']*len(n_score)))], axis=1)
    scores.columns = ['score', 'label']
    return scores
