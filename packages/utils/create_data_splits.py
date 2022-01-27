"""Create data splits for training, validation, as well as two test splits:
    - a standard test split
    - a challenge test split (paradigms unobserved during training).
"""
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


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

def create_train_dev_test_split(rest_frame: pd.DataFrame, challenge_test_frame_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

    Args:
        rest_frame (pd.DataFrame): The paradigm dataset after the challenge test set was extracted from it.
        challenge_test_frame_fraction (float): The number of forms that were taken from the complete paradigm
            dataset to make the challenge dataset. This is used to determine the number of forms we should take
            from {rest_frame} to make the standard test split. 

    Returns:
        (pd.DataFrame, pd.DataFrame, pd.DataFrame): train, validation, and standard test dataframes. 
    """
    is_valid_split = False

    def _is_valid_paradigm_distribution(x_train, x_val, x_test) -> bool:
        valid_train_split = x_train["paradigm_i"].value_counts().gt(1).all()
        valid_dev_split = pd.concat([x_val, x_train])["paradigm_i"].value_counts().gt(1).all()
        valid_test_split = pd.concat([x_test, x_train])["paradigm_i"].value_counts().gt(1).all()
        return valid_train_split and valid_dev_split and valid_test_split


    num_iters = 0
    train_frames = []
    test_frames = []
    valid_frames = []

    while not is_valid_split: 
        p_standard_test_frame = (challenge_test_frame_fraction) / (1-challenge_test_frame_fraction)  
        x_remainder, x_test = train_test_split(rest_frame, test_size=p_standard_test_frame, shuffle=True)
        x_train, x_val = train_test_split(x_remainder, test_size=0.1, shuffle=True)


        is_valid_split = _is_valid_paradigm_distribution(x_train, x_val, x_test) 
        num_iters += 1
        print(f"Num iters: {num_iters}")
    return x_train, x_val, x_test

# # TODO: test this
def create_train_dev_test_split(rest_frame: pd.DataFrame, challenge_test_frame_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

    Args:
        rest_frame (pd.DataFrame): The paradigm dataset after the challenge test set was extracted from it.
        challenge_test_frame_fraction (float): The number of forms that were taken from the complete paradigm
            dataset to make the challenge dataset. This is used to determine the number of forms we should take
            from {rest_frame} to make the standard test split. 

    Returns:
        (pd.DataFrame, pd.DataFrame, pd.DataFrame): train, validation, and standard test dataframes. 
    """
    def _transplant_to_train(train_frame: pd.DataFrame, other_frame: pd.DataFrame, paradigm_i: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
        # collect all indices from paradigm_i in other_frame
        paradigm_i_other_frame = other_frame.loc[other_frame["paradigm_i"]==paradigm_i]

        # randomly sample one entry
        sampled_row = paradigm_i_other_frame.sample(n=1)
        transplant_index = sampled_row.index.values[0]

        # remove that entry from other_frame
        mask = other_frame.index.isin([transplant_index])
        other_frame = other_frame.iloc[~mask] 

        # add that entry to train_frame
        sampled_row["split_type"] = ["train"]
        train_frame = pd.concat([train_frame, sampled_row])
        return train_frame, other_frame.loc[other_frame["split_type"]=="val"], other_frame.loc[other_frame["split_type"]=="test"]

    def _get_paradigms_with_one_entry(frame: pd.DataFrame) -> List[int]:
        paradigm_i_series = frame["paradigm_i"]
        paradigm_i_count_series = paradigm_i_series.value_counts()
        return paradigm_i_count_series.loc[paradigm_i_count_series == 1].index.values

    p_standard_test_frame = (challenge_test_frame_fraction) / (1-challenge_test_frame_fraction)  
    x_remainder, x_test = train_test_split(rest_frame, test_size=p_standard_test_frame, shuffle=True, random_state=0)
    x_train, x_val = train_test_split(x_remainder, test_size=0.1, shuffle=True, random_state=0)

    x_train["split_type"] = ["train"] * len(x_train)
    x_val["split_type"] = ["val"] * len(x_val)
    x_test["split_type"] = ["test"] * len(x_test)

    for paradigm_i in _get_paradigms_with_one_entry(pd.concat([x_val, x_train])): # ensuring that we can predict every form in validation.
        x_train, x_val, x_test = _transplant_to_train(x_train, pd.concat([x_val, x_test]), paradigm_i)
    for paradigm_i in _get_paradigms_with_one_entry(pd.concat([x_test, x_train])): # ensuring that we can predict every form in test.
       x_train, x_val, x_test = _transplant_to_train(x_train, pd.concat([x_val, x_test]), paradigm_i)
    for paradigm_i in _get_paradigms_with_one_entry(x_train): # ensuring that we can predict every form in train.
        x_train, x_val, x_test = _transplant_to_train(x_train, pd.concat([x_val, x_test]), paradigm_i)
    return x_train, x_val, x_test

def make_all_pairs_frame(source_frame, target_frame):
    """Make the all-pairs frame (where every entry in the paradigm serves as a 
    source for every other input)

    Args:
        source_frame ([pd.DataFrame]): Frame with one word per row. Has the |paradigm_i| column.
        target_frame ([type]): Frame with one word per row. Has the |paradigm_i| column.

    Returns:
        pd.DataFrame: A DataFrame with the desired entries.
    """
    paradigm_join_frame = pd.merge(source_frame, target_frame, on="paradigm_i", suffixes=["_src", "_tgt"])
    dataset_frame = paradigm_join_frame.loc[paradigm_join_frame["MSD_src"] != paradigm_join_frame["MSD_tgt"] ]
    return dataset_frame