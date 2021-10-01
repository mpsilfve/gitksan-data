import pandas as pd
import argparse
from itertools import combinations, permutations
from sklearn.model_selection import train_test_split
from functools import partial

from packages.utils.gitksan_table_utils import obtain_orthographic_value, obtain_tag, is_empty_entry , combine_tags, get_paradigm_to_counts, obtain_seen_test_frame, stream_all_paradigms, strip_accents, make_reinflection_frame
from packages.pkl_operations.pkl_io import store_csv_dynamic
from packages.visualizations.plot_summary_distributions import plot_character_distribution, plot_feat_distribution
from packages.utils.inspect_paradigm_file import count_num_paradigms_with_multiple_roots



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

def obtain_train_dev_test_split(frame, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    x_train, x_test = train_test_split(frame, test_size=1 - train_ratio)
    x_val, x_test = train_test_split(x_test, test_size=test_ratio/(test_ratio + dev_ratio), shuffle=False)
    return x_train, x_val, x_test

def count_num_forms(paradigms):
    return sum([paradigm.count_num_forms() for paradigm in paradigms])


def write_reinflection_line(mc_data_file, row):
    source_tag = row.source_tag
    target_tag = row.target_tag
    
    mc_format_tag = combine_tags(source_tag, target_tag)
    reinflection_line = f'{strip_accents(row.source_form.strip())}\t{strip_accents(row.target_form.strip())}\t{mc_format_tag}\n'
    line_elems = reinflection_line.split('\t')
    assert len(line_elems) == 3
    mc_data_file.write(reinflection_line)

def write_mc_file(data_fname, frame):
    with open(f'data/spreadsheets/{data_fname}', 'w') as mc_data_file:
        frame.apply(partial(write_reinflection_line, mc_data_file), axis=1)

def make_train_dev_test_files(frame, dir_suffix):
    train_frame, dev_frame, test_frame = obtain_train_dev_test_split(frame)
    write_mc_file("random_split" + dir_suffix + "/gitksan_productive.train", train_frame)
    write_mc_file("random_split" + dir_suffix + "/gitksan_productive.dev", dev_frame)
    write_mc_file("random_split" + dir_suffix + "/gitksan_productive.test", test_frame)

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

def make_covered_test_file():
    inputs = []
    tags = []
    with open("data/spreadsheets/gitksan_productive.test", 'r') as gitksan_test_file:
        for line in gitksan_test_file:
            i_form, o_form, tag = line.split('\t') 
            inputs.append(i_form)
            tags.append(tag)
    with open('data/spreadsheets/gitksan_productive_covered.test', 'w') as gitksan_test_file_covered:
        for i in range(len(inputs)):
            i_form = inputs[i]
            tag = tags[i]
            gitksan_test_file_covered.write(f'{i_form}\t{tag}')
# TODO: complete and then test this. 

def make_train_dev_seen_unseen_test_files(frame, dir_suffix):
    train_frame, dev_frame, test_frame = obtain_train_dev_test_split(frame)
    train_frame, seen_test_frame = obtain_seen_test_frame(train_frame)
    write_mc_file("seen_unseen_split" + dir_suffix + '/gitksan_productive.train', train_frame)
    write_mc_file("seen_unseen_split" + dir_suffix + '/gitksan_productive.dev', dev_frame)
    write_mc_file("seen_unseen_split" + dir_suffix + '/gitksan_productive_unseen.test', test_frame)
    write_mc_file("seen_unseen_split" + dir_suffix + '/gitksan_productive_seen.test', seen_test_frame)
    # TODO: save these under data.

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

def main(args):
    if args.make_reinflection_frame_csv:
        make_reinflection_frame_csv(args.include_root)
    elif args.make_train_dev_test_files_random_sample:
        reinflection_frame_fname = 'reinflection_frame.csv' if args.make_train_dev_test_files_random_sample != '_w_root' else 'reinflection_frame_w_root.csv'
        frame = pd.read_csv(f'results/2021-09-30/{reinflection_frame_fname}')
        make_train_dev_test_files(frame, args.make_train_dev_test_files_random_sample)
    elif args.make_train_dev_seen_unseen_test_files:
        reinflection_frame_fname = 'reinflection_frame.csv' if args.make_train_dev_seen_unseen_test_files != '_w_root' else 'reinflection_frame_w_root.csv'
        frame = pd.read_csv(f'results/2021-09-30/{reinflection_frame_fname}')
        make_train_dev_seen_unseen_test_files(frame, args.make_train_dev_seen_unseen_test_files)
    elif args.diagnose_train_dev_test_files:
        diagnose_train_dev_test_files()
    elif args.make_covered_test_file:
        make_covered_test_file()
    elif args.plot_char_distribution:
        plot_char_distribution()
    elif args.count_num_root_variation_tables:
        count_num_paradigms_with_multiple_roots()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--make_reinflection_frame_csv', action='store_true')
    parser.add_argument('--include_root', action='store_true')

    parser.add_argument('--make_train_dev_test_files_random_sample', nargs='?', type=str, const='')
    parser.add_argument('--make_train_dev_seen_unseen_test_files', nargs='?', type=str, const='' )

    parser.add_argument('--make_covered_test_file', action='store_true')
    parser.add_argument('--diagnose_train_dev_test_files', action='store_true')
    parser.add_argument('--plot_char_distribution', action='store_true')
    parser.add_argument('--count_num_root_variation_tables', action='store_true')

    main(parser.parse_args())
    # paradigms = extract_non_empty_paradigms()
    # paradigm_frame = make_reinflection_frame(paradigms)
    # print(paradigm_frame)
    # store_csv_dynamic(paradigm_frame, "reinflection_frame")
    # x_train, x_val, x_test = obtain_train_dev_test_split(paradigms)