import math
import pandas as pd
import argparse
from itertools import combinations, permutations
from collections import Counter


from packages.utils.gitksan_table_utils import filter_paradigms, obtain_orthographic_value, obtain_paradigm_frames, obtain_tag, is_empty_entry , get_paradigm_to_counts, stream_all_paradigms, strip_accents, extract_non_empty_paradigms, make_train_dev_test_files, obtain_train_dev_test_split, write_mc_file, convert_inflection_file_to_frame
from packages.utils.create_data_splits import *
from packages.pkl_operations.pkl_io import store_csv_dynamic
from packages.utils.constants import *
from packages.visualizations.plot_summary_distributions import plot_character_distribution, plot_feat_distribution, plot_fullness_dist, plot_msd_distribution
from packages.utils.inspect_paradigm_file import inspect_root_distribution
from packages.augmentation.cross_table import create_cross_table_reinflection_frame


def count_num_forms(paradigms):
    return sum([paradigm.count_num_forms() for paradigm in paradigms])

def read_reinflection_file_into_frame(fname):
    source_forms = []
    target_forms = []
    inflection_tags = []
    with open(fname, 'r') as reinflection_f:
        for line in reinflection_f:
            if line == "\n":
                continue
            line = line.strip()
            contents = line.split('\t')
            source_forms.append(contents[0])
            target_forms.append(contents[1])
            inflection_tags.append(contents[2])
    frame = pd.DataFrame({
        "source_form": source_forms, 
        "target_form": target_forms,
        "reinflection_tag": inflection_tags 
    })
    return frame

def extract_all_unique_tag_features(reinflection_frame):
    tag_series = reinflection_frame['reinflection_tag']
    unique_tag_features = set([])
    tag_series.apply(lambda tag: [unique_tag_features.update(tag.split(';')) ])
    return unique_tag_features

def extract_all_unique_characters(reinflection_frame):
    all_chars = set([])
    source_series = reinflection_frame['source_form']
    target_series = reinflection_frame['target_form']
    form_series = pd.concat([source_series, target_series])
    form_series.apply(lambda form: [all_chars.update(set(form))])
    return all_chars

def diagnose_train_dev_test_files():
    train_fname = "data/spreadsheets/random_split/gitksan_productive.train"
    dev_fname = "data/spreadsheets/random_split/gitksan_productive.dev"
    test_fname = "data/spreadsheets/random_split/gitksan_productive.test"
    train_frame = read_reinflection_file_into_frame(train_fname)
    dev_frame = read_reinflection_file_into_frame(dev_fname)
    test_frame = read_reinflection_file_into_frame(test_fname)

    unique_train_tags = extract_all_unique_tag_features(train_frame)
    unique_dev_tags = extract_all_unique_tag_features(dev_frame)
    unique_test_tags = extract_all_unique_tag_features(test_frame)
    print(unique_dev_tags <= unique_train_tags)
    print(unique_test_tags <= unique_train_tags)

    unique_train_chars = extract_all_unique_characters(train_frame)
    unique_dev_chars = extract_all_unique_characters(dev_frame)
    unique_test_chars = extract_all_unique_characters(test_frame)
    print(unique_dev_chars)
    print(unique_dev_chars <= unique_train_chars)
    print(unique_test_chars <= unique_train_chars)

def plot_char_distribution():
    reinflection_frame = pd.read_csv('results/2021-09-18/reinflection_frame.csv')
    plot_character_distribution(reinflection_frame)
    plot_feat_distribution(reinflection_frame)
    
def make_reinflection_frame_csv(include_root):
    fname = "whitespace-inflection-tables-gitksan-productive.txt"
    paradigms = extract_non_empty_paradigms(fname)
    paradigm_frame = make_reinflection_frame(paradigms, include_root)
    include_root_suffix = "_w_root" if include_root else ""
    store_csv_dynamic(paradigm_frame, "reinflection_frame" + include_root_suffix)

def plot_paradigm_fullness_distribution():
    plot_fullness_dist(extract_non_empty_paradigms("whitespace-inflection-tables-gitksan-productive.txt"))

def plot_num_forms_per_msd():
    fname = "whitespace-inflection-tables-gitksan-productive.txt"
    paradigms = extract_non_empty_paradigms(fname)
    all_forms = []
    all_tags = []
    for paradigm in paradigms:
        for form, tag in paradigm.stream_form_tag_pairs():
            all_tags.append(tag)
            all_forms.append(form)
    frame = pd.DataFrame({"form": all_forms, "tag": all_tags})
    print(frame['tag'].value_counts())
    plot_msd_distribution(frame)

def make_train_dev_test_split():
    """Make word-level train/dev/challenge_test/standard_test splits.

    Stored under data/spreadsheets/standard_challenge_split
    """
    all_paradigms_frame = pd.read_csv("data/spreadsheets/standard_challenge_split/condensed_paradigms.csv", usecols=list(range(1,7)))
    challenge_test_frame_fraction = 0.1
    challenge_test_frame_size = math.ceil(challenge_test_frame_fraction * len(all_paradigms_frame))
    challenge_test_frame, rest_frame = divide_challenge_rest_frame(all_paradigms_frame, challenge_test_frame_size)
    train_frame, dev_frame, standard_test_frame = create_train_dev_test_split(rest_frame) 
    store_csv_dynamic(challenge_test_frame, 'challenge_test_frame', "data/spreadsheets/standard_challenge_split", False)
    store_csv_dynamic(train_frame, 'train_frame', "data/spreadsheets/standard_challenge_split", False)
    store_csv_dynamic(dev_frame, 'dev_frame', "data/spreadsheets/standard_challenge_split", False)
    store_csv_dynamic(standard_test_frame, 'standard_test_frame', "data/spreadsheets/standard_challenge_split", False)

def make_condensed_paradigms_spreadsheet(): # top level
    """See https://glacier-impatiens-3c9.notion.site/Dataset-construction-dd569deb5ddd43349b1498703b51a6da
    under "Construction" heading for a description of this process.
    """
    all_paradigms = []
    for paradigm in stream_all_paradigms('whitespace-inflection-tables-gitksan-productive.txt'):
        all_paradigms.append(paradigm)
    filtered_paradigms = filter_paradigms(all_paradigms)
    condensed_paradigms = obtain_paradigm_frames(filtered_paradigms)
    all_paradigms_frame = pd.concat(condensed_paradigms)
    store_csv_dynamic(all_paradigms_frame, "condensed_paradigms.csv", "data/spreadsheets/standard_challenge_split")

def make_all_pairs_dataset():
    train_frame = pd.read_csv(f"{STD_CHL_SPLIT_PATH}/type_split/train_frame.csv")
    train_dataset = make_all_pairs_frame(train_frame, train_frame) 

def inspect_split_sizes():
    challenge_test_frame = pd.read_csv(f"{STD_CHL_SPLIT_PATH}/challenge_test_frame.csv")
    standard_test_frame = pd.read_csv(f"{STD_CHL_SPLIT_PATH}/standard_test_frame.csv")
    num_paradigms_challenge = len(set(challenge_test_frame['paradigm_i'].values))
    num_paradigms_standard = len(set(standard_test_frame['paradigm_i'].values))
    print(f"Number of paradigms in challenge: {num_paradigms_challenge}")
    print(f"Number of paradigms in standard: {num_paradigms_standard}")

def main(args):
    if args.make_reinflection_frame_csv:
        make_reinflection_frame_csv(args.include_root)
    elif args.make_condensed_paradigms_spreadsheet: 
        make_condensed_paradigms_spreadsheet()
    elif args.make_train_dev_test_split: 
        make_train_dev_test_split()
    elif args.diagnose_train_dev_test_files:
        diagnose_train_dev_test_files()
    elif args.plot_char_distribution:
        plot_char_distribution()
    elif args.count_num_root_variation_tables:
        inspect_root_distribution()
    elif args.plot_paradigm_fullness_distribution:
        plot_paradigm_fullness_distribution()
    elif args.plot_num_forms_per_msd:
        plot_num_forms_per_msd()
    # elif args.check_unseen_test_files:
    #     check_unseen_test_files()
    # elif args.make_cartesian_product_source_files:
    #     make_cartesian_product_source_files()
    # elif args.make_random_source:
    #     make_random_source_files()
    elif args.inspect_split_sizes:
        inspect_split_sizes()


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--make_reinflection_frame_csv', action='store_true')
    parser.add_argument('--make_cross_table_reinflection_frame_csv', action='store_true')
    parser.add_argument('--include_root', action='store_true')

    parser.add_argument('--make_train_dev_test_split', action='store_true')
    parser.add_argument('--make_condensed_paradigms_spreadsheet', action='store_true')
    parser.add_argument('--make_cross_table_train_dev_seen_unseen_test_files', action='store_true')

    parser.add_argument('--make_covered_test_file', action='store_true')
    parser.add_argument('--inspect_split_sizes', action='store_true')
    parser.add_argument('--diagnose_train_dev_test_files', action='store_true')
    parser.add_argument('--check_unseen_test_files', action='store_true')
    parser.add_argument('--plot_char_distribution', action='store_true')
    parser.add_argument('--plot_num_forms_per_msd', action='store_true')
    parser.add_argument('--count_num_root_variation_tables', action='store_true')
    parser.add_argument('--plot_paradigm_fullness_distribution', action='store_true')

    main(parser.parse_args())
    # paradigms = extract_non_empty_paradigms()
    # paradigm_frame = make_reinflection_frame(paradigms)
    # print(paradigm_frame)
    # store_csv_dynamic(paradigm_frame, "reinflection_frame")
    # x_train, x_val, x_test = obtain_train_dev_test_split(paradigms)