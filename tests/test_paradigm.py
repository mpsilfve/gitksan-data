import pytest
from packages.utils.gitksan_table_utils import extract_non_empty_paradigms, stream_all_paradigms, obtain_orthographic_value, calculate_mst_weights_for_perms, obtain_paradigm_frames
from packages.utils.paradigm import generate_permutations 

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

def test_paradigm_contains_plural_root():
    paradigm_fname = "tests/test_whitespace_inflection_tables_gitksan_1.txt"
    all_paradigms = []
    for paradigm in stream_all_paradigms(paradigm_fname):
        all_paradigms.append(paradigm) 
    plural_root_paradigm = all_paradigms[5]
    no_plural_root_paradigm = all_paradigms[0]
    assert plural_root_paradigm.has_root_pl()
    assert not no_plural_root_paradigm.has_root_pl()

def test_paradigm_to_dataframe():
    paradigm_fname = "tests/test_whitespace_inflection_tables_gitksan_1.txt"
    all_paradigms = []
    for paradigm in stream_all_paradigms(paradigm_fname):
        all_paradigms.append(paradigm) 
    no_plural_root_paradigm = all_paradigms[3]
    paradigm_frame = no_plural_root_paradigm._to_dataframe()
    assert len(paradigm_frame) == 4
    # TODO: make assertions

def test_paradigm_has_multiple_entries_for_msd():
    paradigm_fname = "tests/test_whitespace_inflection_tables_gitksan_1.txt"
    all_paradigms = []
    for paradigm in stream_all_paradigms(paradigm_fname):
        all_paradigms.append(paradigm) 
    no_plural_root_paradigm = all_paradigms[5]
    assert no_plural_root_paradigm.has_multiple_entries_for_msd()

def test_generate_permutations():
    perm_result = generate_permutations([['ayook'], ['ayookt'], ['ayoogam'], ['ayookdiit']])
    assert perm_result == [['ayook', 'ayookt', 'ayoogam', 'ayookdiit']]

    perm_result = generate_permutations([['ayook', 'ayookt']])
    assert perm_result == [['ayook'], ['ayookt']]

    perm_result = generate_permutations([['ayook'], ['ayookt', 'ayoogam']])
    assert perm_result == [['ayook', 'ayookt'], ['ayook', 'ayoogam']]

def test_get_msds_forms_sequence():
    paradigm_fname = "tests/test_whitespace_inflection_tables_gitksan_1.txt"
    all_paradigms = []
    for paradigm in stream_all_paradigms(paradigm_fname):
        all_paradigms.append(paradigm) 
    paradigm = all_paradigms[-1]
    msd_form_sequence = paradigm.get_msds_forms_sequence()
    assert len(msd_form_sequence) == 7
    assert (msd_form_sequence[0][0]) == ('ROOT')
    assert all(msd_form_sequence[0][1] == ['we', 'wa'])

def test_calculate_mst():
    paradigm_fname = "tests/test_whitespace_inflection_tables_gitksan_1.txt"
    all_paradigms = []
    for paradigm in stream_all_paradigms(paradigm_fname):
        all_paradigms.append(paradigm) 
    paradigm = all_paradigms[-1]
    mst_weights = calculate_mst_weights_for_perms([paradigm])
    print(mst_weights)
    assert mst_weights != []

def test_obtain_paradigm_frames(dialectal_variation_paradigm):
    filtered_paradigm_frames = obtain_paradigm_frames(dialectal_variation_paradigm)
    frame = filtered_paradigm_frames[0]
    print(frame)
    assert len(frame) == 7

@pytest.fixture
def dialectal_variation_paradigm():
    paradigm_fname = "tests/test_whitespace_inflection_tables_gitksan_1.txt"
    all_paradigms = []
    for paradigm in stream_all_paradigms(paradigm_fname):
        all_paradigms.append(paradigm) 
    variation_paradigm = [all_paradigms[-1]]
    return variation_paradigm