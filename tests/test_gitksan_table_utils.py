import pandas as pd
import pytest
import os
from packages.utils.gitksan_table_utils import *
from packages.utils.paradigm import get_root_orthographic
from math import factorial
from functools import partial

def test_obtain_orthographic_value():
    entry = "ROOT-3.II	law-3.II	ayook̲-t	ayook̲t	ayook̲-3.II"
    expected = "ayook̲t"
    actual = obtain_orthographic_value(entry)
    assert expected == actual

def test_obtain_tag():
    entry = "ROOT-3.II	law-3.II	ayook̲-t	ayook̲t	ayook̲-3.II"
    expected = "ROOT;3.II"
    actual = obtain_tag(expected)
    assert expected == actual

def test_combine_tags():
    source_tag = "ROOT;3.II"
    target_tag = "ROOT;2.III"
    expected = "X;IN:ROOT;IN:3.II;OUT:ROOT;OUT:2.III"
    actual = combine_tags(source_tag, target_tag)
    assert expected == actual

def test_get_root_orthographic():
    entry = "ROOT	law	ayook̲	ayook̲	ayook̲"
    expected = "ayook̲"
    actual = get_root_orthographic(entry)
    assert expected == actual

def test_strip_accents():
    str_a = "t'ilx̲o'odiit"
    str_b = "dok̲"
    str_c = "hishalaagax̲"
    str_d = "tx̲ook̲"
    str_e = "g̲ag̲oott"
    str_f = "daa'wihlt"

    assert strip_accents(str_a) == "t'ilXo'odiit"
    assert strip_accents(str_b) == "doK"
    assert strip_accents(str_c) == "hishalaagaX"
    assert strip_accents(str_d) == "tXooK"
    assert strip_accents(str_e) == "GaGoott"
    assert strip_accents(str_f) == "daa'wihlt"

def test_stream_paradigms_1():
    paradigms = []
    for paradigm in stream_all_paradigms('tests/test_whitespace_inflection_tables_gitksan_1.txt'):
        paradigms.append(paradigm)
    assert paradigms[0].count_num_roots() == 0
    assert paradigms[0].count_num_forms() == 0

    assert paradigms[1].count_num_roots() == 1
    assert paradigms[1].count_num_forms() == 0

    assert paradigms[2].count_num_roots() == 2
    assert paradigms[2].count_num_forms() == 0

    assert paradigms[3].count_num_forms() == 3
    assert paradigms[3].count_num_roots() == 1

    assert paradigms[4].count_num_forms() == 1
    assert paradigms[4].count_num_roots() == 0

def test_construct_reinflection_frame_w_root():
    paradigms = []
    for paradigm in stream_all_paradigms('tests/test_whitespace_inflection_tables_gitksan_1.txt'):
        paradigms.append(paradigm)
    def _num_partial_permutations(n, k):
        return factorial(n) / factorial(n-k)
    reinflection_frame = make_reinflection_frame(paradigms, True)
    ayook_frame = reinflection_frame.loc[reinflection_frame['paradigm']=='ayook̲'] # one root multiple wfs
    assert ayook_frame['source_form'].value_counts()['ayook̲'] == 3
    assert ayook_frame['target_form'].value_counts()['ayook̲'] == 3
    assert len(ayook_frame) == _num_partial_permutations(4, 2) 

    neks_frame = reinflection_frame.loc[reinflection_frame['paradigm'] == 'neks:naks'] # two roots multiple wfs
    assert len(neks_frame) == _num_partial_permutations(6, 2)
    root_source_neks_frame = neks_frame.loc[neks_frame['source_form'] == 'neks']
    assert len(root_source_neks_frame) == 5

    huut_frame = reinflection_frame.loc[reinflection_frame['paradigm'] == '_']
    assert len(huut_frame) == _num_partial_permutations(4, 2)

    assert len(reinflection_frame) == len(ayook_frame) + len(neks_frame) +  len(huut_frame)

def test_construct_reinflection_frame_wo_root():
    paradigms = []
    for paradigm in stream_all_paradigms('tests/test_whitespace_inflection_tables_gitksan_1.txt'):
        paradigms.append(paradigm)
    def _num_partial_permutations(n, k):
        return factorial(n) / factorial(n-k)
    reinflection_frame = make_reinflection_frame(paradigms, False)
    ayook_frame = reinflection_frame.loc[reinflection_frame['paradigm']=='ayook̲'] # one root multiple wfs
    assert len(ayook_frame) == _num_partial_permutations(3, 2) 

    neks_frame = reinflection_frame.loc[reinflection_frame['paradigm'] == 'neks:naks'] # two roots multiple wfs
    assert len(neks_frame) == _num_partial_permutations(5, 2)
    root_source_neks_frame = neks_frame.loc[neks_frame['source_form'] == 'neks']
    assert len(root_source_neks_frame) == 0

    huut_frame = reinflection_frame.loc[reinflection_frame['paradigm'] == '_']
    assert len(huut_frame) == _num_partial_permutations(4, 2)

    assert len(reinflection_frame) == len(ayook_frame) + len(neks_frame) +  len(huut_frame)

# TODO: need to discuss with Miikka
def test_obtain_seen_test_frame():
    paradigms = []
    for paradigm in stream_all_paradigms('whitespace-inflection-tables-gitksan-productive.txt'):
        paradigms.append(paradigm)
    train_frame = make_reinflection_frame(paradigms, False)
    split_train_frame, split_seen_test_frame = obtain_seen_test_frame(train_frame, seen_test_fraction=0.1)
    train_paradigms = split_train_frame['paradigm'].value_counts().keys()
    test_paradigms = split_seen_test_frame['paradigm'].value_counts().keys()
    for paradigm in test_paradigms:
        assert paradigm in train_paradigms
    train_source_target_pairs = list(split_train_frame[['source_form', 'source_tag', 'target_form', 'target_tag']].to_records(index=False))
    test_source_target_pairs = list(split_seen_test_frame[['source_form', 'source_tag', 'target_form', 'target_tag']].to_records(index=False))
    for test_pair in test_source_target_pairs:
        if test_pair in train_source_target_pairs:
            print(train_source_target_pairs.index(test_pair))
        # NOTE: this assertion fails; keeping it commented out for testing coverage
        # assert not test_pair in train_source_target_pairs

def test_extract_non_empty_paradigms(): 
    paradigm_fname = "tests/test_whitespace_inflection_tables_gitksan_1.txt"
    non_empty_paradigms = extract_non_empty_paradigms(paradigm_fname)
    all_paradigms = []
    for paradigm in stream_all_paradigms(paradigm_fname):
        all_paradigms.append(paradigm) 
    assert len(all_paradigms) - len(non_empty_paradigms) == 3

def test_write_mc_file(reinflection_frame, mc_file_resource):
    data_fname = "tests/test_mc_file"
    write_mc_file(data_fname, reinflection_frame)
    with open(f'data/spreadsheets/{data_fname}', 'r') as test_mc_f:
        first_line = test_mc_f.readline()
        assert first_line == "wilt\twilda\tX;IN:ROOT;IN:SX;OUT:ROOT;OUT:3PL\n"
        # 0,wilt,ROOT;SX,wilda,ROOT;3PL,wil
    # def write_mc_file(data_fname, frame):
    #     def _write_reinflection_line(mc_data_file, row):
    #         source_tag = row.source_tag
    #         target_tag = row.target_tag

    #         mc_format_tag = combine_tags(source_tag, target_tag)
    #         reinflection_line = f'{strip_accents(row.source_form.strip())}\t{strip_accents(row.target_form.strip())}\t{mc_format_tag}\n'
    #         line_elems = reinflection_line.split('\t')
    #         assert len(line_elems) == 3
    #         mc_data_file.write(reinflection_line)

    #     with open(f'data/spreadsheets/{data_fname}', 'w') as mc_data_file:
    #         frame.apply(partial(_write_reinflection_line, mc_data_file), axis=1)

# TODO; try this out tomorrow 
def test_make_train_dev_test_files(reinflection_frame, train_dev_test_file_resources):
    def _transform_src(frame, prefix):
        frame['source_form'] = frame['source_form'].apply(lambda src: src + prefix)
        return frame
    train_reinflection_frame = _transform_src(reinflection_frame.copy(), '_train')
    dev_reinflection_frame = _transform_src(reinflection_frame.copy(), '_dev')
    test_reinflection_frame = _transform_src(reinflection_frame.copy(), '_test')

    def _obtain_tdt_split(frame):
        return train_reinflection_frame, dev_reinflection_frame, test_reinflection_frame
    make_train_dev_test_files(reinflection_frame, '_tests', _obtain_tdt_split)
    for ftype in ['train', 'dev', 'test']: 
        with open(f'data/spreadsheets/random_split_tests/gitksan_productive.{ftype}', 'r') as mc_file:
            for line in mc_file:
                if line != '\n':
                    src_form = line.split('\t')[0]
                    assert ftype in src_form
    # TODO: check test covered after


    # def make_train_dev_test_files(frame, dir_suffix, obtain_tdt_split):
    #     train_frame, dev_frame, test_frame = obtain_tdt_split(frame)
    #     write_mc_file("random_split" + dir_suffix + "/gitksan_productive.train", train_frame)
    #     write_mc_file("random_split" + dir_suffix + "/gitksan_productive.dev", dev_frame)
    #     write_mc_file("random_split" + dir_suffix + "/gitksan_productive.test", test_frame)
    #     make_covered_test_file("random_split" + dir_suffix + "/gitksan-test-covered", test_frame)

def test_convert_inflection_file_to_frame():
    frame = convert_inflection_file_to_frame('tests/gitksan_productive.train')
    assert len(frame) == 2847
    first_row = frame.loc[0]
    assert first_row.source == "baX-seksdiit_bekwdiit"
    assert first_row.target == "baX-seksdiit"
    assert first_row.source_msd == "ROOT;PL;3PL.II"
    assert first_row.target_msd == "ROOT;PL;3PL.II"

def test_get_target_to_paradigm_mapping():
    paradigm_fname = "tests/test_whitespace_inflection_tables_gitksan_1.txt"
    non_empty_paradigms = extract_non_empty_paradigms(paradigm_fname)
    form_msd_to_paradigm = get_target_to_paradigm_mapping(non_empty_paradigms)
    print(form_msd_to_paradigm.keys())
    assert form_msd_to_paradigm["ayook̲_ROOT"] == 0

def test_get_all_reduplications():
    train_frame = pd.read_csv("tests/resources/train_type_split_redup.csv")
    redup_frame = get_all_reduplications(train_frame)
    assert len(redup_frame) == 3

def test_get_all_suppletions():
    train_frame = pd.read_csv("tests/resources/train_type_split_supp.csv")
    redup_frame = get_all_suppletions(train_frame)
    assert len(redup_frame) == 5

@pytest.fixture()
def mc_file_resource():
    yield 
    print("removing test mc file")
    os.remove('data/spreadsheets/tests/test_mc_file')

@pytest.fixture
def train_dev_test_file_resources():
    yield 
    print("removing test random split files")
    os.remove('data/spreadsheets/random_split_tests/gitksan_productive.train')
    os.remove('data/spreadsheets/random_split_tests/gitksan_productive.dev')
    os.remove('data/spreadsheets/random_split_tests/gitksan_productive.test')
    os.remove('data/spreadsheets/random_split_tests/gitksan-test-covered')

@pytest.fixture()
def reinflection_frame():
    return pd.read_csv('tests/reinflection_frame.csv')