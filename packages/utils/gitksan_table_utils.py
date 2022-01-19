import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial
import pandas as pd
from collections import Counter
from itertools import combinations, permutations
from .paradigm import *
from .utils import map_list 


def is_empty_entry(entry):
    return "\t_\t_\t_\t_" in entry

def combine_tags(source_tag, target_tag):
    source_feats = source_tag.split(";")
    target_feats = target_tag.split(";")
    source_feats = list(map(lambda feat: f"IN:{feat}", source_feats))
    target_feats = list(map(lambda feat: f"OUT:{feat}", target_feats))

    source_tag_str = ";".join(source_feats)
    target_tag_str = ";".join(target_feats)
    return f"X;{source_tag_str};{target_tag_str}"

def filter_prefix(prefix, tag):
    elems = tag.split(f'{prefix}:')
    tag = ''.join(elems)
    tag = ';'.join(tag.split(';'))
    return tag

def get_paradigm_to_counts(frame):
    paradigm_to_counts = frame['paradigm'].value_counts()
    return paradigm_to_counts.to_dict()

# TODO: there is a bug here
# def obtain_seen_test_frame(train_frame, seen_test_fraction=0.1):
#     train_frame = train_frame.sample(frac=1)
#     paradigm_to_counts = get_paradigm_to_counts(train_frame)
#     extracted_inds = set([]) 
#     num_seen_test_samples = int(seen_test_fraction * len(train_frame))
#     num_extracted = 0
#     for row in train_frame.itertuples(): 
#         ind = row[0] 
#         paradigm = row.paradigm
#         if ind in extracted_inds or paradigm_to_counts[paradigm] == 1:
#             continue
#         else:
#             extracted_inds.add(ind)
#             paradigm_to_counts[paradigm] -= 1
#             num_extracted += 1  
        
#         if num_extracted == num_seen_test_samples:
#             break
#     seen_test_frame = train_frame.loc[extracted_inds]
#     train_frame_wo_seen_test_frame = train_frame.drop(extracted_inds, axis=0)
#     return train_frame_wo_seen_test_frame, seen_test_frame

# TODO: need to test
def obtain_unseen_test_frame(reinflection_frame, unseen_test_fraction=0.1):
    num_required_unseen_forms = len(reinflection_frame) * unseen_test_fraction
    paradigms = list(set(reinflection_frame['paradigm'].values))
    num_forms_extracted = 0
    paradigms_extracted = []

    while num_forms_extracted < num_required_unseen_forms:
        rand_paradigm_i = np.random.randint(len(paradigms))
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

def make_reinflection_frame(paradigms, include_root):
    """Creates a DataFrame containing every two word-form permutation in every paradigm

    Arguments:
        paradigms [Paradigm,...]: List of Paradigms extracted from 
    """
    def _filter_paradigm(p):
        num_roots = p.count_num_roots()
        num_wfs = p.count_num_forms()
        if num_roots == 0:
            return num_wfs >= 2
        elif num_wfs == 0:
            return False 
        elif num_wfs == 1:
            # return num_roots != 0
            return False
        else:
            return True

    mult_entry_paradigms = filter(_filter_paradigm, paradigms)
    source_forms = []
    source_tags = []
    target_forms = []
    target_tags = []
    paradigm_indices = []
    for paradigm in mult_entry_paradigms:
        form_tag_pairs = []
        for entry in paradigm.entries: 
            if not is_empty_entry(entry): 
                form = obtain_orthographic_value(entry)
                tag = obtain_tag(entry)
                form_tag_pairs.append((form, tag))

        if not paradigm.is_empty_roots() and include_root:
            first_root = paradigm.roots[0]
            root_form = obtain_orthographic_value(first_root)
            tag = "ROOT"
            form_tag_pairs.append((root_form, tag))

        # this will be of the form [((form, tag), (form, tag)), ((form, tag), (form, tag)), ...]
        source_target_combinations = permutations(form_tag_pairs, 2)
        num_combs = 0
        for source_target_comb in source_target_combinations:
            source_forms.append(source_target_comb[0][0])
            source_tags.append(source_target_comb[0][1])
            target_forms.append(source_target_comb[1][0])
            target_tags.append(source_target_comb[1][1])
            num_combs += 1
        paradigm_indices.extend([paradigm.paradigm_index] * num_combs)
    paradigm_frame = pd.DataFrame({
        "source_form": source_forms, 
        "source_tag": source_tags, 
        "target_form": target_forms,
        "target_tag": target_tags, 
        "paradigm": paradigm_indices 
    })
    return paradigm_frame 

def filter_paradigms(paradigms):
    """

    Args:
        paradigms ([Paradigm]): List of paradigms
    
    Returns:
        [pd.DataFrame]: Array of DataFrames
    """
    pass_paradigms = []
    for paradigm in paradigms:
        num_roots = paradigm.count_num_roots()
        if num_roots >= 1:
            num_forms = paradigm.count_num_forms()
            if num_forms >= 1:
                pass_paradigms.append(paradigm.to_dataframe())
    return pass_paradigms

def select_among_duplicate_entries(paradigms):
    for paradigm in paradigms:
        if paradigm.has_dup_entries():
            # TODO: fill in
            pass

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
    x_val, x_test = train_test_split(x_test, test_size=test_ratio/(test_ratio + dev_ratio), shuffle=False)
    return x_train, x_val, x_test

def make_covered_test_file(path_fname, test_frame):
    def _write_test_covered_line(test_covered_f, row):
        target_tag = row.target_tag

        reinflection_line = f'{strip_accents(row.source_form.strip())}\t{target_tag}\n'
        line_elems = reinflection_line.split('\t')
        assert len(line_elems) == 2
        test_covered_f.write(reinflection_line)
    with open(f'data/spreadsheets/{path_fname}', 'w') as test_covered_f:
        test_frame.apply(partial(_write_test_covered_line, test_covered_f), axis=1)

# TODO: need to rewrite this function...
    # the frame should be 
def make_train_dev_seen_unseen_test_files(frame, dir_suffix, proc_frame_row):
    """
    Args:
        frame (pd.DataFrame): |word|tag|paradigm_i|
        dir_suffix (str): w_root
        write_reinflection_line ((pd.DataFrame) => str): Converts row in {frame} to a string representing entry in inflection dataset.
    """
    unseen_test_frame_ratio = 0.1
    unseen_test_frame, rest_frame = obtain_unseen_test_frame(frame, unseen_test_frame_ratio) 

    train_frame, dev_frame, test_frame = obtain_train_dev_test_split(rest_frame, 1 - 2/9, 1/9, 1/9)
    train_paradigms = set(train_frame['paradigm'].values)
    test_paradigms = set(test_frame['paradigm'].values)
    non_train_paradigms = test_paradigms.difference(train_paradigms)
    non_train_paradigm_frame = test_frame[test_frame['paradigm'].isin(non_train_paradigms)]
    seen_test_frame = test_frame.drop(non_train_paradigm_frame.index.values, axis=0)
    unseen_test_frame = pd.concat([non_train_paradigm_frame, unseen_test_frame])

    write_mc_file("seen_unseen_split" + dir_suffix + '/gitksan_productive.train', train_frame, proc_frame_row)
    write_mc_file("seen_unseen_split" + dir_suffix + '/gitksan_productive.dev', dev_frame, proc_frame_row)
    write_mc_file("seen_unseen_split" + dir_suffix + '/gitksan_productive_unseen.test', unseen_test_frame, proc_frame_row)
    write_mc_file("seen_unseen_split" + dir_suffix + '/gitksan_productive_seen.test', seen_test_frame, proc_frame_row)
    make_covered_test_file("seen_unseen_split" + dir_suffix + '/gitksan_productive_unseen-covered', unseen_test_frame)
    make_covered_test_file("seen_unseen_split" + dir_suffix + '/gitksan_productive_seen-covered', seen_test_frame)

    return non_train_paradigms

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
        "source": sources, 
        "target": targets,
        "source_msd": map_list( partial(filter_prefix, 'IN'), source_msds), 
        "target_msd": map_list( partial(filter_prefix, 'OUT'), target_msds),
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