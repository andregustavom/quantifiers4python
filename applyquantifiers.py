from Heronian_mean_matching_CF import Heronian_MM_CF
from Heronian_mean_matching import Heronian_MM
from CC import classify_count
from ACC import ACC
from PCC import PCC
from PACC import PACC
from HDy import Hdy
from X import X
from MAX import Max
from SMM import SMM  
from SMM_2 import SMM_2
from SMM_modified import SMM_modified
from SMM_check import SMM_check
from GMM import GMM
from GMM_basic import GMM_basic
from HMM import HMM
from DYs_Ts import Dys_Ts
from SMM_T_test import SMM_T_test
from dys_method import dys_method
from sord import SORD_method
from MS import MS_method
from MS_2 import MS_method2
from T50 import T50
from zscore import SMM_Zscore
from tscore_eq_var import tscore_eq_var
from tscore_one_sample import one_sample_Ttest
from EMQ import EMQ
from SMM_max import SMM_max
from SMM_x import SMM_x
from SMM_t50 import SMM_t50
from SMM_ms import SMM_ms
from SMM_Outlier_std import SMM_Outlier_std
from SMM_Outlier_iqr import SMM_Outlier_iqr
from GMM_cf import GMM_cf
from SMM_median import SMM_median
from Heronian_mean_matching import Heronian_MM
from Heronian_mean_matching_CF import Heronian_MM_CF


"""This function is an interface for running different quantification methods.
 
Parameters
----------
qntMethod : string
    Quantification method name
p_score : array
    A numeric vector of positive scores estimated either from a validation set or from a cross-validation method.
n_score : array
    A numeric vector of negative scores estimated either from a validation set or from a cross-validation method.
test : array
    A numeric vector of scores predicted from the test set.
TprFpr : matrix
    A matrix of true positive (tpr) and false positive (fpr) rates estimated on training set, using the function getScoreKfolds().
thr : float
    The threshold value for classifying and counting. Default is 0.5.
measure : string
    Dissimilarity function name used by the DyS method. Default is "topsoe".

Returns
-------
array
    the class distribution of the test calculated according to the qntMethod quantifier. 
 """
def apply_quantifier(qntMethod, p_score, n_score,test_score, TprFpr, thr, measure, calib_clf, te_data):

    if qntMethod == "cc":
        return classify_count(test_score, thr)
    if qntMethod == "acc":        
        return ACC(test_score, TprFpr)
    if qntMethod == "emq":        
        return EMQ(p_score, n_score, test_score)
    if qntMethod == "smm":
        return SMM(p_score, n_score, test_score)
    if qntMethod == "heronian_mm":
        return Heronian_MM(p_score, n_score, test_score)
    if qntMethod == "heron_mm_cf":
        return Heronian_MM_CF(p_score, n_score, test_score)
    if qntMethod == "smm_median":
        return SMM_median(p_score, n_score, test_score)
    if qntMethod == "smmmax":
        return SMM_max(p_score, n_score, test_score, TprFpr)
    if qntMethod == "smmx":
        return SMM_x(p_score, n_score, test_score, TprFpr)
    if qntMethod == "smmt50":
        return SMM_t50(p_score, n_score, test_score, TprFpr)
    if qntMethod == "smm_ms":
        return SMM_ms(p_score, n_score, test_score, TprFpr)
    if qntMethod == "smm_std":
        return SMM_Outlier_std(p_score, n_score, test_score)
    if qntMethod == "smm_iqr":
        return SMM_Outlier_iqr(p_score, n_score, test_score)
    if qntMethod == "gmm":
        return GMM(p_score, n_score, test_score)
    if qntMethod == "hmm":
        return HMM(p_score, n_score, test_score)
    if qntMethod == "gmm_cf":
        return GMM_cf(p_score, n_score, test_score)
    if qntMethod == "gmm_1":
        return GMM_basic(p_score, n_score, test_score)
    if qntMethod == "tscore":
        return SMM_T_test(p_score, n_score, test_score)
    if qntMethod == "tscore_1":
        return tscore_eq_var(p_score, n_score, test_score)
    if qntMethod == "tscore_2":
        return one_sample_Ttest(p_score, n_score, test_score)
    if qntMethod == "zscore":
        return SMM_Zscore(p_score, n_score, test_score)
    if qntMethod == "dys_ts":
        return Dys_Ts(p_score, n_score, test_score)
    if qntMethod == "hdy":
        return Hdy(p_score, n_score, test_score)
    if qntMethod == "dys_method":
        return dys_method(p_score, n_score, test_score,measure)
    if qntMethod == "sord":
        return SORD_method(p_score, n_score,test_score)
    if qntMethod == "ms":
        return MS_method(test_score, TprFpr)
    if qntMethod == "ms2":
        return MS_method2(test_score, TprFpr)
    if qntMethod == "max":
        return Max(test_score, TprFpr)
    if qntMethod == "x":
        return X(test_score, TprFpr)
    if qntMethod == "t50":
        return T50(test_score, TprFpr)
    if qntMethod == "pcc":
        return PCC(calib_clf, te_data,thr)
    if qntMethod == "pacc":
        return PACC(calib_clf, te_data, TprFpr, thr)