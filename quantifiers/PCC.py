import pandas as pd
import numpy as np

def PCC(calib_clf,test_data, thr = 0.5):

    calibrated_predictions = calib_clf.predict_proba(test_data)[:,1]
    
    pcc_count = sum(calibrated_predictions[calibrated_predictions > thr])
    pos_prop = round(pcc_count/len(calibrated_predictions),2)

    if pos_prop <= 0:                           #clipping the output between [0,1]
        pos_prop = 0
    elif pos_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = pos_prop
    
    return pos_prop
