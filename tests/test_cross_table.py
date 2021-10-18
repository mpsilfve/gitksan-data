import pandas as pd
import pytest

from packages.augmentation.cross_table import create_cross_table_reinflection_frame
from packages.utils.gitksan_table_utils import stream_all_paradigms

def test_create_cross_table_reinflection_frame(reinflection_frame, paradigms):
    cross_table_frame = create_cross_table_reinflection_frame(reinflection_frame, paradigms)
    assert len(cross_table_frame) == len(reinflection_frame) # checking that the correct cross-table paradigm is selected.
    assert cross_table_frame['cross_table_i'].values[0] == 7
    # print(cross_table_frame['cross_table_i'])

# def test_():

@pytest.fixture()
def reinflection_frame():
    return pd.read_csv('tests/reinflection_frame.csv')

@pytest.fixture()
def paradigms():
    pdgms = []
    for paradigm in stream_all_paradigms('tests/test_whitespace_inflection_tables_gitksan_1.txt'):
        pdgms.append(paradigm)
    return pdgms