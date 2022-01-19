import pandas as pd
import argparse
from itertools import combinations, permutations
from collections import Counter

from packages.utils.gitksan_table_utils import obtain_orthographic_value, obtain_tag, is_empty_entry , combine_tags, get_paradigm_to_counts, stream_all_paradigms, strip_accents, make_reinflection_frame, extract_non_empty_paradigms, make_train_dev_test_files, obtain_train_dev_test_split, write_mc_file, make_covered_test_file, make_train_dev_seen_unseen_test_files, convert_inflection_file_to_frame
from packages.pkl_operations.pkl_io import store_csv_dynamic
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

def make_cross_table_reinflection_frame_csv():
    fname = "whitespace-inflection-tables-gitksan-productive.txt"
    paradigms = extract_non_empty_paradigms(fname)
    reinflection_frame = pd.read_csv(f'results/2021-10-18/reinflection_frame_w_root.csv')
    cross_table_frame = create_cross_table_reinflection_frame(reinflection_frame, paradigms)
    store_csv_dynamic(cross_table_frame, "cross_table_reinflection_frame")


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

def make_train_dev_seen_unseen_test_files_tl(): # top level
    def _write_reinflection_line(mc_data_file, row):
        source_tag = row.source_tag
        target_tag = row.target_tag

        mc_format_tag = combine_tags(source_tag, target_tag)
        paradigm_num = row.paradigm
        reinflection_line = f'{strip_accents(row.source_form.strip())}\t{strip_accents(row.target_form.strip())}\t{mc_format_tag}\t{paradigm_num}\n'
        line_elems = reinflection_line.split('\t')
        assert len(line_elems) == 4
        mc_data_file.write(reinflection_line)
    reinflection_frame_fname = 'reinflection_frame_w_root.csv' 
    frame = pd.read_csv(f'results/2021-10-18/{reinflection_frame_fname}')
    make_train_dev_seen_unseen_test_files(frame, '_w_root', _write_reinflection_line)

def make_cross_table_train_dev_seen_unseen_test_files():
    def _write_reinflection_line(mc_data_file, row):
        source_tag = row.source_tag
        target_tag = row.target_tag

        mc_format_tag = combine_tags(source_tag, target_tag)
        paradigm_num = row.paradigm
        cross_table_paradigm_num = row.cross_table_i

        print(row)
        reinflection_line = f'{strip_accents(row.source_form.strip())}_{strip_accents(row.cross_table_src)}\t{strip_accents(row.target_form.strip())}\t{mc_format_tag}\t{paradigm_num}_{cross_table_paradigm_num}\n'
        line_elems = reinflection_line.split('\t')
        assert len(line_elems) == 4
        mc_data_file.write(reinflection_line)
    reinflection_frame_fname = 'cross_table_reinflection_frame.csv' 
    frame = pd.read_csv(f'results/2021-11-22/{reinflection_frame_fname}')
    make_train_dev_seen_unseen_test_files(frame, '_w_root_cross_table', _write_reinflection_line)

def check_unseen_test_files():
    train_inflection_fname = "data/spreadsheets/seen_unseen_split_w_root_cross_table/gitksan_productive.train"
    train_frame = convert_inflection_file_to_frame(train_inflection_fname)
    unseen_inflection_fname = "data/spreadsheets/seen_unseen_split_w_root_cross_table/gitksan_productive_unseen.test"
    unseen_frame = convert_inflection_file_to_frame(unseen_inflection_fname)

    seen_inflection_fname = "data/spreadsheets/seen_unseen_split_w_root_cross_table/gitksan_productive_seen.test"
    seen_frame = convert_inflection_file_to_frame(seen_inflection_fname)

    test_inds = set(seen_frame['paradigm_i'].values).union(set(unseen_frame['paradigm_i'].values))
    unseen_test_inds = set(unseen_frame['paradigm_i'].values)
    seen_test_inds = set(seen_frame['paradigm_i'].values)
    train_inds = set(train_frame['paradigm_i'].values)
    print(unseen_test_inds.intersection(train_inds))
    print(unseen_test_inds.intersection(seen_test_inds))
    print(len(unseen_frame))
    print(len(seen_frame))


def main(args):
    if args.make_reinflection_frame_csv:
        make_reinflection_frame_csv(args.include_root)
    elif args.make_cross_table_reinflection_frame_csv:
        make_cross_table_reinflection_frame_csv()
    elif args.make_train_dev_seen_unseen_test_files: 
        make_train_dev_seen_unseen_test_files_tl()
    elif args.make_cross_table_train_dev_seen_unseen_test_files: 
        make_cross_table_train_dev_seen_unseen_test_files()
    elif args.diagnose_train_dev_test_files:
        diagnose_train_dev_test_files()
    elif args.make_covered_test_file:
        make_covered_test_file()
    elif args.plot_char_distribution:
        plot_char_distribution()
    elif args.count_num_root_variation_tables:
        inspect_root_distribution()
    elif args.plot_paradigm_fullness_distribution:
        plot_paradigm_fullness_distribution()
    elif args.plot_num_forms_per_msd:
        plot_num_forms_per_msd()
    elif args.check_unseen_test_files:
        check_unseen_test_files()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--make_reinflection_frame_csv', action='store_true')
    parser.add_argument('--make_cross_table_reinflection_frame_csv', action='store_true')
    parser.add_argument('--include_root', action='store_true')

    parser.add_argument('--make_train_dev_seen_unseen_test_files', action='store_true')
    parser.add_argument('--make_cross_table_train_dev_seen_unseen_test_files', action='store_true')

    parser.add_argument('--make_covered_test_file', action='store_true')
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