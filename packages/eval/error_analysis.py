import numpy as np
from numpy import average
from sklearn.metrics import average_precision_score
import pandas as pd

def perform_suppletion_analysis(supp_challenge_test_frame: pd.DataFrame):
    """Determines precision at using plural sources as the root for inflection.

    Args:
        supp_challenge_test_frame (pd.DataFrame): Predictions from an inflection model for only the challenge test condition and containing only suppletive forms as targets.
    """
    supp_challenge_test_frame = supp_challenge_test_frame.copy()
    supp_challenge_test_frame['uses_supp_source'] =  supp_challenge_test_frame['MSD_src'].apply(lambda row: 1 if 'PL' in row else 0)
    ap_total = 0
    total_weight = 0
    for paradigm_i in set(supp_challenge_test_frame['paradigm_i'].values):
        paradigm = supp_challenge_test_frame.loc[supp_challenge_test_frame['paradigm_i'] == paradigm_i]
        weight = len(paradigm.loc[paradigm['uses_supp_source'] == 1])
        if weight != 0:
            total_weight += weight
            ap = average_precision_score(paradigm['uses_supp_source'], paradigm['confidences'])
            ap_total += weight * ap
    print(f"Weighted average precision score: {ap_total/total_weight}")