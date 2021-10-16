import pytest
from packages.utils.gitksan_table_utils import extract_non_empty_paradigms, stream_all_paradigms, obtain_orthographic_value

def test_contains_true():
    paradigm_fname = "tests/test_whitespace_inflection_tables_gitksan_1.txt"
    all_paradigms = []
    for paradigm in stream_all_paradigms(paradigm_fname):
        all_paradigms.append(paradigm) 
    first_paradigm = all_paradigms[3]
    assert "ROOT-3.II" in first_paradigm

def test_contains_false():
    paradigm_fname = "tests/test_whitespace_inflection_tables_gitksan_1.txt"
    all_paradigms = []
    for paradigm in stream_all_paradigms(paradigm_fname):
        all_paradigms.append(paradigm) 
    first_paradigm = all_paradigms[3]
    assert not "PL-ROOT" in first_paradigm

def test_getitem():
    paradigm_fname = "tests/test_whitespace_inflection_tables_gitksan_1.txt"
    all_paradigms = []
    for paradigm in stream_all_paradigms(paradigm_fname):
        all_paradigms.append(paradigm) 
    first_paradigm = all_paradigms[3]
    assert first_paradigm["ROOT-3.II"] == "ayook̲t"