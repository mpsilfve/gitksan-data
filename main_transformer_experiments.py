from os import system 
from os import makedirs
import pandas as pd
import argparse
from datetime import datetime
import subprocess

from packages.utils.constants import STD_CHL_SPLIT_PATH, ONESOURCE_RESULTS_PATH, RESULTS_FNAMES
from packages.pkl_operations.pkl_io import store_csv_dynamic
from packages.utils.create_data_splits import make_all_pairs_frame, add_hallucination_examples
from packages.fairseq.fairseq_format import produce_fairseq_data, reformat_from_frame
from packages.fairseq.utils import extract_hypotheses, extract_hypotheses_mbr
from packages.utils.gitksan_table_utils import get_target_to_paradigm_mapping, extract_non_empty_paradigms, convert_inflection_file_to_frame, get_paradigm_inds
from packages.utils.utils import map_list, try_makedirs
from packages.eval.eval import eval_accuracy_oracle, eval_paradigm_accuracy_random, eval_paradigm_accuracy_max, eval_accuracy_per_row
from packages.fairseq.parse_results_files import parse_results_w_logit_file, expand_to_char_frame
from packages.calibration.temperature_scale import ModelWithTemperature

def produce_fairseq_data_cross_table():
    """Saves fairseq train/validation/test data
    """
    produce_fairseq_data("gitksan_productive.train", "gitksan_productive.dev", "gitksan_productive_seen.test", "gitksan_productive_unseen.test", "data/spreadsheets/seen_unseen_split_w_root_cross_table")

def produce_fairseq_data_regular():
    """Produces fairseq format data for the `seen_unseen_split_w_root`.
    """
    produce_fairseq_data("gitksan_productive.train", "gitksan_productive.dev", "gitksan_productive_seen.test", "gitksan_productive_unseen.test", "data/spreadsheets/seen_unseen_split_w_root")

def produce_fairseq_data_hall():
    """Produces fairseq format data for the `seen_unseen_split_w_root_hall` directory.
    """
    produce_fairseq_data("gitksan_productive_w_hall.train", "gitksan_productive.dev", "gitksan_productive_seen.test", "gitksan_productive_unseen.test", "data/spreadsheets/seen_unseen_split_w_root_hall")
    subprocess.call("data/spreadsheets/seen_unseen_split_w_root_hall/postproc-fairseq-train-hall.sh")

def eval_fairseq_cross_table_max(inflection_fname, prediction_fname, num_hypotheses):
    """Conducts PCFP evaluation on the cross-table data augmentation format =)
    """
    frame = convert_inflection_file_to_frame(inflection_fname)

    # prediction_fname = "results/2021-11-19/results_seen_test.txt"
    predictions, confidences = extract_hypotheses(prediction_fname, num_hypotheses)
    frame['predictions'] = predictions
    frame['confidences'] = confidences

    cross_tab_paradigm_inds = frame['paradigm_i'].values.copy()
    frame['paradigm_i'] = map_list( lambda x: int(x.split('_')[0]), cross_tab_paradigm_inds)
    frame['cross_table_i']= map_list( lambda x: int(x.split('_')[1]), cross_tab_paradigm_inds)

    # all_paradigms = extract_non_empty_paradigms("whitespace-inflection-tables-gitksan-productive.txt")
    # target_to_paradigm_mapping = get_target_to_paradigm_mapping(all_paradigms)

    # frame['paradigm_i'] = frame.apply(lambda row: target_to_paradigm_mapping[f'{row.target}_{row.target_msd}'], axis=1)

    eval_paradigm_accuracy_max(frame)
    return frame

def eval_fairseq_cross_table_random(inflection_fname, prediction_fname, num_hypotheses):
    """Conducts PCFP evaluation on the cross-table data augmentation format =)
    """
    frame = convert_inflection_file_to_frame(inflection_fname)

    # prediction_fname = "results/2021-11-19/results_seen_test.txt"
    predictions, confidences = extract_hypotheses(prediction_fname, num_hypotheses)
    frame['predictions'] = predictions
    frame['confidences'] = confidences

    cross_tab_paradigm_inds = frame['paradigm_i'].values.copy()
    frame['paradigm_i'] = map_list( lambda x: int(x.split('_')[0]), cross_tab_paradigm_inds)
    frame['cross_table_i']= map_list( lambda x: int(x.split('_')[1]), cross_tab_paradigm_inds)

    eval_paradigm_accuracy_random(frame)
    eval_accuracy_oracle(frame)
    eval_accuracy_per_row(frame)
    return frame
    
def eval_fairseq_1_source_random(inflection_fname, prediction_fname, num_hypotheses):
    """Conducts PCFP evaluation on the cross-table data augmentation format =)
    """
    frame = pd.read_csv(inflection_fname)
    # frame = convert_inflection_file_to_frame(inflection_fname)

    predictions, confidences = extract_hypotheses(prediction_fname, num_hypotheses)
    frame['predictions'] = predictions
    frame['confidences'] = confidences

    # all_paradigms = extract_non_empty_paradigms("whitespace-inflection-tables-gitksan-productive.txt")
    # target_to_paradigm_mapping = get_target_to_paradigm_mapping(all_paradigms)

    # frame['paradigm_i'] = frame.apply(lambda row: target_to_paradigm_mapping[f'{row.target}_{row.target_msd}'], axis=1)

    # eval_paradigm_accuracy_random(frame)
    # eval_accuracy_oracle(frame)
    eval_paradigm_accuracy_max(frame)
    # eval_accuracy_per_row(frame)
    return frame

def eval_fairseq_1_source_max(inflection_fname, prediction_fname, num_hypotheses):
    frame = convert_inflection_file_to_frame(inflection_fname)

    predictions, confidences = extract_hypotheses(prediction_fname, num_hypotheses)
    frame['predictions'] = predictions
    frame['confidences'] = confidences

    eval_paradigm_accuracy_max(frame)
    return frame

def generate_fairseq_files(split_type_to_frame, aug_strategy, split_path):
    """Generate fairseq files for training Transformer.

    Args:
        split_type_to_frame ({str: pd.DataFrame}): [description]
        aug_strategy (str): one of {one_source, cross_product_source, hallucination}
        save_path (str): Where to save the fairseq files (f"{save_path}/fairseq/...") and 
            the (re)inflection frames. 
    """
    if aug_strategy in ["cross_product_source", "cross_product_hallucination"]:
        make_reinflection_frame = make_all_pairs_frame

    inflection_frame_save_path = f"{split_path}/{aug_strategy}"
    fairseq_save_dir = f"{inflection_frame_save_path}/fairseq"
    try_makedirs(fairseq_save_dir)

    train_frame = split_type_to_frame["train"]
    dev_frame = split_type_to_frame["dev"]
    standard_test_frame = split_type_to_frame["standard_test"]
    challenge_test_frame = split_type_to_frame["challenge_test"]

    for split_type in split_type_to_frame:
        if split_type == "train":
            reinflection_frame = make_reinflection_frame(train_frame, train_frame)
            if aug_strategy == "cross_product_hallucination":
                reinflection_frame = add_hallucination_examples(reinflection_frame, inflection_frame_save_path)
            store_csv_dynamic(reinflection_frame, "train_frame", root_folder=inflection_frame_save_path, include_date=False)
            reformat_from_frame(reinflection_frame, f"{fairseq_save_dir}/gitksan-train.src", f"{fairseq_save_dir}/gitksan-train.tgt")
        elif split_type == "dev":
            reinflection_frame = make_reinflection_frame(pd.concat([train_frame, dev_frame]), dev_frame) 
            store_csv_dynamic(reinflection_frame, "dev_frame", root_folder=inflection_frame_save_path, include_date=False)
            reformat_from_frame(reinflection_frame, f"{fairseq_save_dir}/gitksan-dev.src", f"{fairseq_save_dir}/gitksan-dev.tgt")
        elif split_type == "standard_test":
            reinflection_frame = make_reinflection_frame(pd.concat([train_frame, standard_test_frame]), standard_test_frame) 
            store_csv_dynamic(reinflection_frame, "standard_test_frame", root_folder=inflection_frame_save_path, include_date=False)
            reformat_from_frame(reinflection_frame, f"{fairseq_save_dir}/gitksan-standard_test.src", f"{fairseq_save_dir}/gitksan-standard_test.tgt")
        elif split_type == "challenge_test":
            reinflection_frame = make_reinflection_frame(pd.concat([challenge_test_frame]), challenge_test_frame) 
            store_csv_dynamic(reinflection_frame, "challenge_test_frame", root_folder=inflection_frame_save_path, include_date=False)
            reformat_from_frame(reinflection_frame, f"{fairseq_save_dir}/gitksan-challenge_test.src", f"{fairseq_save_dir}/gitksan-challenge_test.tgt")


def generate_fairseq_files_all_pairs():
    split_type_to_frame = {
        "train": pd.read_csv(f"{STD_CHL_SPLIT_PATH}/train_frame.csv"),
        "dev": pd.read_csv(f"{STD_CHL_SPLIT_PATH}/dev_frame.csv"),
        "standard_test": pd.read_csv(f"{STD_CHL_SPLIT_PATH}/standard_test_frame.csv"),
        "challenge_test": pd.read_csv(f"{STD_CHL_SPLIT_PATH}/challenge_test_frame.csv")
    }
    generate_fairseq_files(split_type_to_frame, "cross_product_source", STD_CHL_SPLIT_PATH)

def generate_fairseq_files_hallucination():
    split_type_to_frame = {
        "train": pd.read_csv(f"{STD_CHL_SPLIT_PATH}/train_frame.csv"),
        "dev": pd.read_csv(f"{STD_CHL_SPLIT_PATH}/dev_frame.csv"),
        "standard_test": pd.read_csv(f"{STD_CHL_SPLIT_PATH}/standard_test_frame.csv"),
        "challenge_test": pd.read_csv(f"{STD_CHL_SPLIT_PATH}/challenge_test_frame.csv")
    }
    generate_fairseq_files(split_type_to_frame, "cross_product_hallucination", STD_CHL_SPLIT_PATH)

def pull_onesource_results():
    date = datetime.today().strftime('%Y-%m-%d')
    folder = f"results/{date}/onesource"
    try:
        makedirs(folder)
    except FileExistsError:
        pass

    for fname in RESULTS_FNAMES:
        system(f"scp {ONESOURCE_RESULTS_PATH}/{fname} {folder}/{fname}")

def main(args):
    if args.produce_fairseq_data_cross_table:
        produce_fairseq_data_cross_table()
    elif args.produce_fairseq_data_regular:
        produce_fairseq_data_regular()
    elif args.produce_fairseq_data_hall:
        produce_fairseq_data_hall()
    elif args.eval_fairseq_cross_table_random:
        inflection_fname_prefix = "data/spreadsheets/seen_unseen_split_w_root_cross_table"
        prediction_fname_prefix = "results/2021-11-22/cross_table"
        eval_fairseq_cross_table_random(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test.txt", 5)
        eval_fairseq_cross_table_random(f"{inflection_fname_prefix}/gitksan_productive_unseen.test", f"{prediction_fname_prefix}/results_unseen_test.txt", 5)
    elif args.eval_fairseq_1_source_random:
        inflection_fname_prefix = f"{STD_CHL_SPLIT_PATH}/cross_product_source"
        prediction_fname_prefix = "results/2022-01-30/onesource"
        eval_fairseq_1_source_random(f"{inflection_fname_prefix}/challenge_test_frame.csv", f"{prediction_fname_prefix}/results_challenge_test.txt", 5)
        eval_fairseq_1_source_random(f"{inflection_fname_prefix}/standard_test_frame.csv", f"{prediction_fname_prefix}/results_standard_test.txt", 5)
    elif args.eval_fairseq_1_source_max:
        inflection_fname_prefix = "data/spreadsheets/seen_unseen_split_w_root"
        prediction_fname_prefix = "results/2021-11-19/1_source"
        eval_fairseq_1_source_max(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test_sampling.txt", 5)
        eval_fairseq_1_source_max(f"{inflection_fname_prefix}/gitksan_productive_unseen.test", f"{prediction_fname_prefix}/results_unseen_test_sampling.txt", 4)
    elif args.eval_fairseq_hallucination_random:
        inflection_fname_prefix = "data/spreadsheets/seen_unseen_split_w_root_hall"
        prediction_fname_prefix = "results/2021-12-11"
        eval_fairseq_1_source_random(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test.txt", 5)
        eval_fairseq_1_source_random(f"{inflection_fname_prefix}/gitksan_productive_unseen.test", f"{prediction_fname_prefix}/results_unseen_test.txt", 5)
    elif args.generate_fairseq_files_all_pairs:
        generate_fairseq_files_all_pairs()
    elif args.generate_fairseq_files_hallucination:
        generate_fairseq_files_hallucination()
    elif args.pull_onesource_results:
        pull_onesource_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pull_onesource_results', action='store_true')
    parser.add_argument('--produce_fairseq_data_cross_table', action='store_true')
    parser.add_argument('--generate_fairseq_files_all_pairs', action='store_true')
    parser.add_argument('--generate_fairseq_files_hallucination', action='store_true')
    parser.add_argument('--produce_fairseq_data_regular', action='store_true')
    parser.add_argument('--produce_fairseq_data_hall', action='store_true')
    parser.add_argument('--eval_fairseq_cross_table_random', action='store_true')
    parser.add_argument('--eval_fairseq_hallucination_random', action='store_true')
    parser.add_argument('--eval_fairseq_1_source_random', action='store_true')
    parser.add_argument('--eval_fairseq_hallucination_max', action='store_true')
    parser.add_argument('--eval_fairseq_1_source_max', action='store_true')
    parser.add_argument('--optimize_temperature_global', action='store_true')

    main(parser.parse_args())