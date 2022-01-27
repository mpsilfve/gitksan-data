import pandas as pd
from packages.utils.create_data_splits import make_all_pairs_frame, create_train_dev_test_split 
from packages.utils.constants import *

# TODO: This currently fails since the train / dev / test / split can sometimes result in too few entries
def test_make_all_pairs_frame():
    train_frame = pd.read_csv(f"{STD_CHL_SPLIT_PATH}/type_split/train_frame.csv")
    train_frame_group_sizes = train_frame.groupby("paradigm_i").size().values
    train_dataset = make_all_pairs_frame(train_frame, train_frame) 
    train_dataset_group_sizes = train_dataset.groupby("paradigm_i").size().values
    print(train_dataset.groupby("paradigm_i").size())
    print(train_frame.groupby("paradigm_i").size())
    for i in range(len(train_dataset_group_sizes)):
        print(i)
        assert train_dataset_group_sizes[i] == ((train_frame_group_sizes[i]) ** 2) - train_frame_group_sizes[i]

def test_get_paradigms_with_one_entry():
    frame = pd.DataFrame(
        data={
            "paradigm_i": [1,1,3,3,4,6],
        }
    )
    s = frame['paradigm_i'].value_counts()
    s = s.loc[s == 1]
    print(s.index.values)

def test_create_train_dev_test_split():
    all_paradigms_frame = pd.read_csv(STD_CHL_SPLIT_ALL_PARADIGMS)
    train_frame, dev_frame, test_frame = create_train_dev_test_split(all_paradigms_frame, 0.1)
    assert train_frame["paradigm_i"].value_counts().gt(1).all()
    assert pd.concat([dev_frame["paradigm_i"],train_frame["paradigm_i"]]).value_counts().gt(1).all()
    assert pd.concat([train_frame["paradigm_i"],test_frame["paradigm_i"]]).value_counts().gt(1).all()

    assert len(train_frame) + len(dev_frame) + len(test_frame) == len(all_paradigms_frame)
    print(len(train_frame))
    print(len(dev_frame))
    print(len(test_frame))

    # assert test_frame["paradigm_i"].value_counts().gt(1).all()