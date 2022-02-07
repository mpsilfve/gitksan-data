import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial
import pandas as pd
from collections import Counter
from typing import List

from packages.utils.paradigm_tree import calculate_mst_weight, gen_fully_connected_graph

from .paradigm import *
from .utils import map_list 
from .paradigm_tree import *

np.random.seed(0)

def is_empty_entry(entry):
    return "\t_\t_\t_\t_" in entry

def filter_prefix(prefix, tag):
    elems = tag.split(f'{prefix}:')
    tag = ''.join(elems)
    tag = ';'.join(tag.split(';'))
    return tag

def get_paradigm_to_counts(frame):
    paradigm_to_counts = frame['paradigm'].value_counts()
    return paradigm_to_counts.to_dict()

def obtain_unseen_test_frame(reinflection_frame, unseen_test_fraction=0.1):
    num_required_unseen_forms = len(reinflection_frame) * unseen_test_fraction
    paradigms = list(set(reinflection_frame['paradigm'].values))
    num_forms_extracted = 0
    paradigms_extracted = []

    while num_forms_extracted < num_required_unseen_forms:
        rand_paradigm_i = np.random.randint(len(paradigms), )
        sampled_paradigm = paradigms[rand_paradigm_i]
        paradigm_frame = reinflection_frame[reinflection_frame['paradigm']==sampled_paradigm]
        paradigms_extracted.append(paradigm_frame)
        num_forms_extracted += len(paradigm_frame)

    unseen_paradigm_frame = pd.concat(paradigms_extracted)
    unseen_inds = unseen_paradigm_frame.index.values
    rest_frame = reinflection_frame.drop(unseen_inds, axis=0)
    return unseen_paradigm_frame, rest_frame

def get_char_counts(reinflection_frame, form_col_name='source_form'):
    char_counter = Counter()
    def update_counter(c):
        char_counter[c] += 1
    
    source_form_series = reinflection_frame[form_col_name]
    source_form_series.apply(lambda form: [update_counter(c) for c in form])
    return char_counter

def get_tag_feat_counts(reinflection_frame, form_col_name='source_tag'):
    tag_feat_counter = Counter()
    def update_counter(c):
        tag_feat_counter[c] += 1
    
    source_form_series = reinflection_frame[form_col_name]
    source_form_series.apply(lambda tag: [update_counter(feat) for feat in tag.split(';')])
    return tag_feat_counter

def stream_all_paradigms(fname):
    with open(fname, 'r') as gp_f:
        gp_f.readline()
        paradigm_num = 0
        for paradigm_block in gp_f:
            line = paradigm_block
            paradigm = []
            while line != "\n" and line != "": # (start of new paradigm) 
                paradigm.append(line)
                line = gp_f.readline()
            yield Paradigm(paradigm, paradigm_num)
            paradigm_num += 1

def filter_paradigms(paradigms):
    """

    Args:
        paradigms ([Paradigm]): List of paradigms
    
    Returns:
        [Paradigm]: Array of paradigms.
    """
    pass_paradigms = []
    for paradigm in paradigms:
        num_roots = paradigm.count_num_roots()
        if num_roots >= 1:
            num_forms = paradigm.count_num_forms()
            if num_forms >= 1:
                pass_paradigms.append(paradigm)
    return pass_paradigms

def obtain_paradigm_frames(paradigms: List[Paradigm]):
    """Filters MSDs that have duplicate entries for paradigms using 
    a Minimum Spanning Tree algorithm.

    Args:
        paradigms (List[Paradigm]): Paradigms extracted from raw paradigms file (whitespace....txt)
    
    Returns:
        [pd.DataFrame]: DataFrames with one entry per MSD.
    """
    filtered_frames = []
    for paradigm in paradigms:
        if paradigm.has_multiple_entries_for_msd():
            msd_forms_sequence = paradigm.get_msds_forms_sequence()
            forms_sequence = map_list(lambda msd_forms: msd_forms[1], msd_forms_sequence)
            perms = generate_permutations(forms_sequence)
            mst_weights = calculate_mst_weights_for_perms(perms)
            filtered_frame = extract_filtered_frame(perms, mst_weights, msd_forms_sequence, paradigm.frame)
            filtered_frames.append(filtered_frame)
        else:
            filtered_frames.append(paradigm.frame)
    return filtered_frames

def extract_filtered_frame(perms, mst_weights, msd_form_seq, paradigm_frame):
    """Extract the filtered frame from ...

    Args:
        perms ([[str]]): all possible realizations of the paradigm. 
        mst_weights ([int]): MST cumulative weights; parallel to {perms}
        paradigm_frame (pd.DataFrame): [description]
    """
    min_weight_i = mst_weights.index(min(mst_weights))
    best_perm = perms[min_weight_i]
    for i in range(len(msd_form_seq)):
        msd = msd_form_seq[i][0]
        forms = msd_form_seq[i][1] 
        if len(forms) > 1:
            best_form = best_perm[i]
            msd_inds = paradigm_frame[paradigm_frame["MSD"] == msd].index.values
            best_form_ind = paradigm_frame[(paradigm_frame["MSD"] == msd) & (paradigm_frame["form"] == best_form)].index.values[0]
            suboptimal_inds = set(msd_inds).difference([best_form_ind])
            paradigm_frame = paradigm_frame.drop(index=[ind for ind in suboptimal_inds])
    return paradigm_frame

def calculate_mst_weights_for_perms(perms):
    mst_weights = []
    for perm in perms:
        graph = gen_fully_connected_graph(perm)
        graph_mst = obtain_mst(graph)
        mst_weight = calculate_mst_weight(graph_mst)
        mst_weights.append(mst_weight)
    return mst_weights

def extract_non_empty_paradigms(paradigm_fname):
    num_paradigms = 0
    non_empty_paradigms = []
    for paradigm in stream_all_paradigms(paradigm_fname):
        num_paradigms += 1
        if not paradigm.is_empty():
            non_empty_paradigms.append(paradigm)
    print(f"Read {num_paradigms} paradigms!")
    print(f"There are {len(non_empty_paradigms)} non-empty paradigms")
    return non_empty_paradigms

def write_mc_file(data_fname, frame, write_reinflection_line):
    """[summary]

    Args:
        data_fname ([type]): [description]
        frame ([type]): [description]
        write_reinflection_line ([type]): [description]
    """
    with open(f'data/spreadsheets/{data_fname}', 'w') as mc_data_file:
        frame.apply(partial(write_reinflection_line, mc_data_file), axis=1)

def make_train_dev_test_files(frame, dir_suffix, obtain_tdt_split):
    train_frame, dev_frame, test_frame = obtain_tdt_split(frame)
    write_mc_file("random_split" + dir_suffix + "/gitksan_productive.train", train_frame)
    write_mc_file("random_split" + dir_suffix + "/gitksan_productive.dev", dev_frame)
    write_mc_file("random_split" + dir_suffix + "/gitksan_productive.test", test_frame)
    make_covered_test_file("random_split" + dir_suffix + "/gitksan-test-covered", test_frame)

def obtain_train_dev_test_split(frame, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    x_train, x_test = train_test_split(frame, test_size=1 - train_ratio, random_state=0)
    x_val, x_test = train_test_split(x_test, test_size=test_ratio/(test_ratio + dev_ratio), shuffle=False, random_state=0)
    return x_train, x_val, x_test


def get_target_to_paradigm_mapping(paradigms):
    """Returns a target to paradigm mapping.

    Args:
        paradigms ([Paradigm]): List of non_empty paradigms read from `whitespace-inflection-tables-gitksan-productive.txt`. Should be in order according to that file.
    """
    form_msd_to_paradigm = {}
    for i in range(len(paradigms)):
        paradigm = paradigms[i]
        for form, tag in paradigm.stream_form_tag_pairs():
            if f'{form}_{tag}' in form_msd_to_paradigm:
                assert form_msd_to_paradigm[f'{form}_{tag}'] == i, f"{form}_{tag} in paradigm {i} and in {form_msd_to_paradigm[f'{form}_{tag}']}" 
            form_msd_to_paradigm[f'{form}_{tag}'] = i
    return form_msd_to_paradigm

def convert_inflection_file_to_frame(inflection_fname):
    """

    Args:
        inflection_fname (str): e.g., "data/spreadsheets/seen_unseen_split_w_root_cross_table/gitksan_productive_seen.test

    Returns:
        pd.DataFrame: |source|target|source_msds|target_msds
    """
    sources = []
    targets = []
    source_msds = []
    target_msds = []
    paradigm_inds = []
    for line in open(inflection_fname):
        entries = line.split('\t')
        source = entries[0]
        target = entries[1]
        out_start_i = entries[2].index("OUT:") 
        source_msd = entries[2][2:out_start_i - 1]# -1 to exclude semicolon. Start from 2 to remove X: marker
        target_msd = entries[2][out_start_i:].strip()
        sources.append(source)
        targets.append(target)
        source_msds.append(source_msd)
        target_msds.append(target_msd)
        paradigm_ind = entries[-1].strip()
        paradigm_inds.append(paradigm_ind)
    frame = pd.DataFrame({
        "form_src": sources, 
        "form_tgt": targets,
        "MSD_src": map_list( partial(filter_prefix, 'IN'), source_msds), 
        "MSD_tgt": map_list( partial(filter_prefix, 'OUT'), target_msds),
        "paradigm_i": paradigm_inds
    })
    return frame

def get_paradigm_inds(inflection_fname):
    paradigm_inds = []
    with open(inflection_fname, 'r') as inflection_f:
        for line in inflection_f:
            paradigm_i = line.split('\t')[-1]
            paradigm_inds.append(paradigm_i)
    return paradigm_inds