"""Create data splits for training, validation, as well as two test splits:
    - a standard test split
    - a challenge test split (paradigms unobserved during training).
"""
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

np.random.seed(0)

def divide_challenge_rest_frame(all_paradigms_frame, challenge_test_frame_size):
    ctfz = challenge_test_frame_size
    paradigm_indices = list(set(all_paradigms_frame['paradigm_i'].values))
    num_forms_extracted = 0
    paradigms_extracted = []

    while num_forms_extracted < ctfz:
        rand_paradigm_i = np.random.randint(len(paradigm_indices))
        sampled_paradigm = paradigm_indices[rand_paradigm_i]
        paradigm_frame = all_paradigms_frame[all_paradigms_frame['paradigm_i']==sampled_paradigm]
        paradigms_extracted.append(paradigm_frame)
        num_forms_extracted += len(paradigm_frame)

    challenge_paradigm_frame = pd.concat(paradigms_extracted)
    unseen_inds = challenge_paradigm_frame.index.values
    rest_frame = all_paradigms_frame.drop(unseen_inds, axis=0)
    return challenge_paradigm_frame, rest_frame

def create_train_dev_test_split(rest_frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

    Args:
        rest_frame (pd.DataFrame): The paradigm dataset after the challenge test set was extracted from it.
        challenge_test_frame_fraction (float): The number of forms that were taken from the complete paradigm
            dataset to make the challenge dataset. This is used to determine the number of forms we should take
            from {rest_frame} to make the standard test split. 

    Returns:
        (pd.DataFrame, pd.DataFrame, pd.DataFrame): train, validation, and standard test dataframes. 
    """
    train_elems, dev_elems, standard_test_elems = [], [], []
    rest_frame = rest_frame.sample(len(rest_frame)) # shuffled
    for paradigm in rest_frame['paradigm_i'].unique():
        paradigm_frame = rest_frame.loc[rest_frame['paradigm_i']==paradigm]
        paradigm_size = len(paradigm_frame)
        if paradigm_size < 3: # directly assign to train. since we're trying to build a seen test set
            train_elems.append(paradigm_frame)
        else: # >= 3
            num_samples = np.random.randint(1, paradigm_size - 1) # between 1 and paradigm_size - 2 (inclusive)
            assign_to_dev = True if np.random.randint(0,2) == 0 else False
            paradigm_samples_frame = paradigm_frame.sample(n=num_samples)
            rest_paradigm_frame = paradigm_frame.drop(paradigm_samples_frame.index.values, axis=0)
            train_elems.append(rest_paradigm_frame)
            if assign_to_dev:
                dev_elems.append(paradigm_samples_frame)
            else:
                standard_test_elems.append(paradigm_samples_frame)
    return pd.concat(train_elems), pd.concat(dev_elems), pd.concat(standard_test_elems)

def make_all_pairs_frame(source_frame, target_frame):
    """Make the all-pairs frame (where every entry in the paradigm serves as a 
    source for every other input).

    Args:
        source_frame (pd.DataFrame): Frame with one word per row. Has the |paradigm_i| column.
        target_frame (pd.DataFrame): Frame with one word per row. Has the |paradigm_i| column.

    Returns:
        pd.DataFrame: A DataFrame with the desired entries.
    """
    paradigm_join_frame = pd.merge(source_frame, target_frame, on="paradigm_i", suffixes=["_src", "_tgt"])
    dataset_frame = paradigm_join_frame.loc[paradigm_join_frame["MSD_src"] != paradigm_join_frame["MSD_tgt"]]
    return dataset_frame