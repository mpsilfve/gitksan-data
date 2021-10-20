import pytest
import os 
from ..packages.fairseq.fairseq_format import reformat

def test_reformat():
    train_fname = "data/spreadsheets/seen_unseen_split_w_root_cross_table/gitksan_productive.train"
    fsrc = "data/spreadsheets/tests/crosstable_fairseq.src"
    ftrg = "data/spreadsheets/tests/crosstable_fairseq.trg"
    reformat(train_fname, fsrc, ftrg)

@pytest.fixture
def fairseq_format_files():
    fsrc = "data/spreadsheets/tests/crosstable_fairseq.src"
    ftrg = "data/spreadsheets/tests/crosstable_fairseq.trg"
    yield 
    print("removing test fairseq data files")
    os.remove(fsrc)
    os.remove(ftrg)