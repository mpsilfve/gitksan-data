from packages.utils.gitksan_table_utils import obtain_orthographic_value, obtain_tag, combine_tags, strip_accents, stream_all_paradigms, make_reinflection_frame
from packages.utils.paradigm import get_root_orthographic
from math import factorial

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

def test_construct_reinflection_frame():
    paradigms = []
    for paradigm in stream_all_paradigms('tests/test_whitespace_inflection_tables_gitksan_1.txt'):
        paradigms.append(paradigm)
    def _num_partial_permutations(n, k):
        return factorial(n) / factorial(n-k)
    reinflection_frame = make_reinflection_frame(paradigms)
    ayook_frame = reinflection_frame.loc[reinflection_frame['paradigm']=='ayook̲'] # one root multiple wfs
    assert len(ayook_frame) == _num_partial_permutations(4, 2) 

    neks_frame = reinflection_frame.loc[reinflection_frame['paradigm'] == 'neks:naks'] # two roots multiple wfs
    assert len(neks_frame) == _num_partial_permutations(6, 2)
    root_source_neks_frame = neks_frame.loc[neks_frame['source_form'] == 'neks']
    assert len(root_source_neks_frame) == 5

    huut_frame = reinflection_frame.loc[reinflection_frame['paradigm'] == '_']
    assert len(huut_frame) == _num_partial_permutations(4, 2)

    assert len(reinflection_frame) == len(ayook_frame) + len(neks_frame) +  len(huut_frame)