import pytest
import pandas as pd
# from ..packages.utils.gitksan_table_utils import get_paradigm_to_counts, obtain_seen_test_frame
from packages.utils.gitksan_table_utils import get_paradigm_to_counts, obtain_seen_test_frame, strip_accents, stream_all_paradigms
from itertools import combinations, permutations


# def test_read_paradigms():
#     paradigms = extract_non_empty_paradigms()
#     x_train, x_val, x_test = obtain_train_dev_test_split(paradigms)
#     print(f"There are {len(x_train)} paradigms in the train set")
#     print(f"There are {len(x_val)} paradigms in the dev set")
#     print(f"There are {len(x_test)} paradigms in the test set")

#     print(f"There are {count_num_forms(x_train)} forms in the train set")
#     print(f"There are {count_num_forms(x_val)} forms in the dev set")
#     print(f"There are {count_num_forms(x_test)} forms in the test set")

def test_produce_paradigm_to_counts():
    d = {'form': ['abc', 'def', 'ghi', 'jkl'], 'paradigm': ['A', 'B', 'A', 'C']}
    frame = pd.DataFrame(
        data=d
    )
    assert get_paradigm_to_counts(frame) == {'A': 2, 'B': 1, 'C': 1}

def test_obtain_seen_test_frame():
    # TODO: test that the seen test frame and the returned train frame are exclusive
    # TODO: test that the seen test frame items are from the same paradigm as other items in the returned train frame
    d = {'form': ['abc', 'def', 'ghi', 'jkl', 'mno'], 'paradigm': ['A', 'B', 'A', 'C', 'D']}
    train_frame = pd.DataFrame(
        data=d
    )
    train_frame_wo_seen_test_frame, seen_test_frame = obtain_seen_test_frame(train_frame) 
    assert len(seen_test_frame) == 1
    assert seen_test_frame['paradigm'].value_counts().to_dict() == {'A': 1}
    assert train_frame_wo_seen_test_frame['paradigm'].value_counts().to_dict() == {'A': 1, 'B': 1, 'C': 1, 'D': 1} 

def test_paradigm_combinations():
    form_tag_pairs = [("sdik'eekwt", "X;IN:ROOT;IN:3.II"), ("sdik'eegwin", "X;IN:ROOT;IN:2SG.II"), ("sdik'eegwin", "X;IN:ROOT;IN:2SG.II")]
    source_target_combinations = permutations(form_tag_pairs, 2)
    assert (len(list(source_target_combinations))) == 6

# TODO: this only tests if consecutive tags are the same; not global ones...
def test_all_paradigms_with_same_msds():
    first_msds = None
    for paradigm in stream_all_paradigms('whitespace-inflection-tables-gitksan-productive.txt'):
        if first_msds is None:
            first_msds = paradigm.get_all_msds()
        else:
            assert first_msds == paradigm.get_all_msds()
            

# NOTE: i already tested this visually; see 2021-09-23 report 
# def test_all_cells_filled_at_least_once():
#     msds = None
#     for paradigm in stream_all_paradigms('whitespace-inflection-tables-gitksan-productive.txt'):
#         if msds is None:
#             msds = paradigm.get_all_msds()