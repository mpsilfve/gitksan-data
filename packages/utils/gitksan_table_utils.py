import unicodedata as ud
import pandas as pd
from collections import Counter
from itertools import combinations, permutations
from .paradigm import Paradigm

def obtain_orthographic_value(entry):
    values = entry.split('\t') 
    return values[3].strip()

def obtain_tag(entry):
    values = entry.split('\t')
    return values[0].strip().replace("-", ";")


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

def get_paradigm_to_counts(frame):
    paradigm_to_counts = frame['paradigm'].value_counts()
    return paradigm_to_counts.to_dict()

def obtain_seen_test_frame(train_frame, seen_test_fraction=0.1):
    train_frame = train_frame.sample(frac=1)
    paradigm_to_counts = get_paradigm_to_counts(train_frame)
    extracted_inds = set([]) 
    num_seen_test_samples = int(seen_test_fraction * len(train_frame))
    num_extracted = 0
    for row in train_frame.itertuples(): 
        ind = row[0]
        paradigm = row.paradigm
        if ind in extracted_inds or paradigm_to_counts[paradigm] == 1:
            continue
        else:
            extracted_inds.add(ind)
            paradigm_to_counts[paradigm] -= 1
            num_extracted += 1
        
        if num_extracted == num_seen_test_samples:
            break
    seen_test_frame = train_frame.loc[extracted_inds]
    train_frame_wo_seen_test_frame = train_frame.drop(extracted_inds, axis=0)
    return train_frame_wo_seen_test_frame, seen_test_frame

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

def strip_accents(s):
    """Replace x̲ with X
               k̲ with K
               g̲ with G

    Important for application of the MNC tool, which doesn't handle diacritics well.

    Args:
        s (str): Gitksan word

    Returns:
        [str]: String with diacritics substituted for capitals.
    """
    clean_s = ""
    # return ''.join(c for c in unicodedata.normalize('NFD', s)
    #                 if unicodedata.category(c) != 'Mn')
    i = 0
    while i < len(s):
        ud_name_cur = ud.name(s[i])
        if ud_name_cur in ["latin small letter k".upper(), "latin small letter g".upper(), "latin small letter x".upper()]:
            if (i+1) < len(s):
                ud_name_next = ud.name(s[i+1])
                if ud_name_next == "combining low line".upper():
                    clean_s += s[i].upper() # replace diacritic with uppercase
                else:
                    clean_s += s[i] # k, g, x
        else:
            if ud_name_cur != "combining low line".upper():
                clean_s += s[i] # all other characters (e.g., letters and apostrophes)
        i += 1
    return clean_s

def stream_all_paradigms(fname):
    with open(fname, 'r') as gp_f:
        gp_f.readline()
        line_num = 0
        for paradigm_block in gp_f:
            line = paradigm_block
            paradigm = []
            while line != "\n" and line != "": # (start of new paradigm) 
                paradigm.append(line)
                line = gp_f.readline()
            yield Paradigm(paradigm)

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
    paradigm_names = []
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
        paradigm_names.extend([paradigm.get_roots()] * num_combs)
    paradigm_frame = pd.DataFrame({
        "source_form": source_forms, 
        "source_tag": source_tags, 
        "target_form": target_forms,
        "target_tag": target_tags, 
        "paradigm": paradigm_names
    })
    return paradigm_frame 