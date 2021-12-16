import pandas as pd
import numpy as np

def PACC(calib_clf, test_data, tprfpr, thr = 0.5):

    calibrated_predictions = calib_clf.predict_proba(test_data)[:,1]
    pcc_count = sum(calibrated_predictions[calibrated_predictions > thr])
    pcc_ouput = pcc_count/len(calibrated_predictions)

    TprFpr = tprfpr[tprfpr['threshold'] == thr]
    
    diff_tpr_fpr = (float(TprFpr['tpr']) - float(TprFpr['fpr']))

    pos_prop = (pcc_ouput - float(TprFpr['fpr'])) / diff_tpr_fpr
     
    if pos_prop <= 0:                           #clipping the output between [0,1]
        pos_prop = 0
    elif pos_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = pos_prop

    return pos_prop
