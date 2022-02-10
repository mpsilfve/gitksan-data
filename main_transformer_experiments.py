from os import system 
from os import makedirs
import pandas as pd
import argparse
from datetime import datetime
import subprocess
from typing import List

from packages.utils.constants import HALLUCINATION_RESULTS_PATH, STD_CHL_SPLIT_PATH, ONESOURCE_RESULTS_PATH, RESULTS_FNAMES
from packages.pkl_operations.pkl_io import store_csv_dynamic
from packages.utils.create_data_splits import make_all_pairs_frame, add_hallucination_examples 
from packages.fairseq.fairseq_format import produce_fairseq_data, reformat_from_frame
from packages.fairseq.utils import extract_hypotheses, extract_hypotheses_mbr
from packages.utils.gitksan_table_utils import get_target_to_paradigm_mapping, extract_non_empty_paradigms, convert_inflection_file_to_frame, get_paradigm_inds, get_all_suppletions, get_all_reduplications
from packages.utils.utils import map_list 
from packages.eval.eval import eval_accuracy_oracle, eval_paradigm_accuracy_random, eval_paradigm_accuracy_max, eval_accuracy_majority_vote 
from packages.fairseq.parse_results_files import parse_results_w_logit_file, expand_to_char_frame
from packages.calibration.temperature_scale import ModelWithTemperature

def eval_fairseq_1_source(inflection_fname, prediction_fname, num_hypotheses):
    """Conducts PCFP evaluation on the 1-source model.
    """
    frame = pd.read_csv(inflection_fname)
    # frame = convert_inflection_file_to_frame(inflection_fname)

    predictions, confidences = extract_hypotheses(prediction_fname, num_hypotheses)
    frame['predictions'] = predictions
    frame['confidences'] = confidences

    # all_paradigms = extract_non_empty_paradigms("whitespace-inflection-tables-gitksan-productive.txt")
    # target_to_paradigm_mapping = get_target_to_paradigm_mapping(all_paradigms)

    # frame['paradigm_i'] = frame.apply(lambda row: target_to_paradigm_mapping[f'{row.target}_{row.target_msd}'], axis=1)

    print("==================Random accuracy===============")
    eval_paradigm_accuracy_random(frame)
    print("==================Oracle accuracy===============")
    eval_accuracy_oracle(frame)
    print("==================Max accuracy===============")
    eval_paradigm_accuracy_max(frame)
    print("==================Majority vote accuracy===========")
    eval_accuracy_majority_vote(frame)
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

    train_frame = split_type_to_frame["train"]
    dev_frame = split_type_to_frame["dev"]
    standard_test_frame = split_type_to_frame["standard_test"]
    challenge_test_frame = split_type_to_frame["challenge_test"]

    for split_type in split_type_to_frame:
        if split_type == "train":
            reinflection_frame = make_reinflection_frame(train_frame, train_frame)
            if aug_strategy == "cross_product_hallucination":
                reinflection_frame = add_hallucination_examples(reinflection_frame)
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
    generate_fairseq_files(split_type_to_frame, "cross_product_w_hallucination", STD_CHL_SPLIT_PATH)

def pull_onesource_results():
    date = datetime.today().strftime('%Y-%m-%d')
    folder = f"results/{date}/onesource"
    try:
        makedirs(folder)
    except FileExistsError:
        pass

    for fname in RESULTS_FNAMES:
        system(f"scp {ONESOURCE_RESULTS_PATH}/{fname} {folder}/{fname}")
    
def pull_hallucination_results():
    date = datetime.today().strftime('%Y-%m-%d')
    folder = f"results/{date}/onesource"
    try:
        makedirs(folder)
    except FileExistsError:
        pass
    for fname in RESULTS_FNAMES:
        system(f"scp {HALLUCINATION_RESULTS_PATH}/{fname} {folder}/{fname}")

def eval_fairseq_1_source_supp_and_red(results_frame: pd.DataFrame, type_split_fnames: List[str], train_type: str):
    print("==================Random accuracy===============")
    type_split_frames = []
    for type_split_fname in type_split_fnames:
        type_split_frame = pd.read_csv(type_split_fname)[["paradigm_i", "gitksan_gloss", "MSD", "eng_gloss", "morph"]]
        type_split_frame = type_split_frame.rename(columns={"MSD": "MSD_tgt"})
        type_split_frames.append(type_split_frame)
    type_split_frame = pd.concat(type_split_frames)

    merge_frame = results_frame.merge(type_split_frame, on=["paradigm_i", "MSD_tgt"])

    supp_frame = get_all_suppletions(merge_frame) 
    eval_paradigm_accuracy_max(supp_frame, True)
    # eval_accuracy_oracle(supp_frame, True)

    # Test
    supp_frame_test = get_all_suppletions(type_split_frame)
    assert (len(supp_frame_test[['MSD_tgt', 'paradigm_i']].drop_duplicates())) == (len(supp_frame[['MSD_tgt','paradigm_i']].drop_duplicates()))
    # Test

    print(supp_frame[['form_src', 'form_tgt', ]])

    redup_frame = get_all_reduplications(merge_frame)
    # eval_paradigm_accuracy_max(redup_frame, True)
    eval_accuracy_oracle(redup_frame, True)

    # Test
    redup_frame_test = get_all_reduplications(type_split_frame)
    assert (len(redup_frame_test[['MSD_tgt', 'paradigm_i']].drop_duplicates())) == (len(redup_frame[['MSD_tgt','paradigm_i']].drop_duplicates()))
    # Test

    print_supp_frame = supp_frame[["form_tgt", "predictions", "form_src", "MSD_src", "MSD_tgt", "confidences", "paradigm_i"]]
    print_supp_frame.groupby('paradigm_i').apply(pd.DataFrame.sort_values, 'confidences', ascending=False)
    store_csv_dynamic(print_supp_frame, f"supp_frame_{train_type}")

    print_redup_frame = redup_frame[["form_tgt", "predictions", "form_src", "MSD_src", "MSD_tgt", "confidences", "paradigm_i"]]
    print_redup_frame.groupby('paradigm_i').apply(pd.DataFrame.sort_values, 'confidences', ascending=False)
    store_csv_dynamic(print_redup_frame, f"redup_frame_{train_type}")

    # eval_accuracy_oracle(frame)
    # print("==================Max accuracy===============")
    # eval_paradigm_accuracy_max(frame)
    # print("==================Majority vote accuracy===========")
    # eval_accuracy_majority_vote(frame)
    return results_frame

def _get_results_frame(inflection_fname: str, prediction_fname: str, num_hypotheses: str) -> pd.DataFrame:
    results_frame = pd.read_csv(inflection_fname)
    # frame = convert_inflection_file_to_frame(inflection_fname)

    predictions, confidences = extract_hypotheses(prediction_fname, num_hypotheses)
    results_frame['predictions'] = predictions
    results_frame['confidences'] = confidences
    return results_frame

def main(args):
    if args.eval_fairseq_1_source:
        inflection_fname_prefix = f"{STD_CHL_SPLIT_PATH}/cross_product_source"
        prediction_fname_prefix = "results/2022-01-30/onesource"
        eval_fairseq_1_source(f"{inflection_fname_prefix}/challenge_test_frame.csv", f"{prediction_fname_prefix}/results_challenge_test.txt", 5)
        eval_fairseq_1_source(f"{inflection_fname_prefix}/standard_test_frame.csv", f"{prediction_fname_prefix}/results_standard_test.txt", 5)
    elif args.eval_fairseq_hallucination:
        inflection_fname_prefix = f"{STD_CHL_SPLIT_PATH}/cross_product_hallucination"
        prediction_fname_prefix = "results/2022-02-03/onesource"
        eval_fairseq_1_source(f"{inflection_fname_prefix}/challenge_test_frame.csv", f"{prediction_fname_prefix}/results_challenge_test.txt", 5)
        eval_fairseq_1_source(f"{inflection_fname_prefix}/standard_test_frame.csv", f"{prediction_fname_prefix}/results_standard_test.txt", 5)
    elif args.eval_fairseq_1_source_supp_and_red_1_source:
        inflection_fname_prefix = f"{STD_CHL_SPLIT_PATH}/cross_product_source"
        prediction_fname_prefix = "results/2022-01-30/onesource"
        challenge_results_frame = _get_results_frame(f"{inflection_fname_prefix}/challenge_test_frame.csv", f"{prediction_fname_prefix}/results_challenge_test.txt", 5)
        standard_results_frame = _get_results_frame(f"{inflection_fname_prefix}/standard_test_frame.csv", f"{prediction_fname_prefix}/results_standard_test.txt", 5)
        # all_results_frame = pd.concat([challenge_results_frame, standard_results_frame])
        all_results_frame = pd.concat([standard_results_frame])
        # eval_fairseq_1_source_supp_and_red(all_results_frame, [f"{STD_CHL_SPLIT_PATH}/standard_test_frame.csv",f"{STD_CHL_SPLIT_PATH}/challenge_test_frame.csv"], "1_source")
        eval_fairseq_1_source_supp_and_red(all_results_frame, [f"{STD_CHL_SPLIT_PATH}/standard_test_frame.csv"], "1_source")
    elif args.eval_fairseq_1_source_supp_and_red_hall:
        inflection_fname_prefix = f"{STD_CHL_SPLIT_PATH}/cross_product_hallucination"
        prediction_fname_prefix = "results/2022-02-03/onesource"
        challenge_results_frame = _get_results_frame(f"{inflection_fname_prefix}/challenge_test_frame.csv", f"{prediction_fname_prefix}/results_challenge_test.txt", 5)
        standard_results_frame = _get_results_frame(f"{inflection_fname_prefix}/standard_test_frame.csv", f"{prediction_fname_prefix}/results_standard_test.txt", 5)
        all_results_frame = pd.concat([ standard_results_frame])
        eval_fairseq_1_source_supp_and_red(all_results_frame, [f"{STD_CHL_SPLIT_PATH}/standard_test_frame.csv"], "hall")
    elif args.generate_fairseq_files_all_pairs:
        generate_fairseq_files_all_pairs()
    elif args.generate_fairseq_files_hallucination:
        generate_fairseq_files_hallucination()
    elif args.pull_onesource_results:
        pull_onesource_results()
    elif args.pull_hallucination_results:
        pull_hallucination_results()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pull_onesource_results', action='store_true')
    parser.add_argument('--pull_hallucination_results', action='store_true')
    parser.add_argument('--produce_fairseq_data_cross_table', action='store_true')
    parser.add_argument('--generate_fairseq_files_all_pairs', action='store_true')
    parser.add_argument('--generate_fairseq_files_hallucination', action='store_true')
    parser.add_argument('--produce_fairseq_data_regular', action='store_true')
    parser.add_argument('--produce_fairseq_data_hall', action='store_true')
    parser.add_argument('--eval_fairseq_cross_table_random', action='store_true')
    parser.add_argument('--eval_fairseq_hallucination', action='store_true')
    parser.add_argument('--eval_fairseq_1_source', action='store_true')
    parser.add_argument('--eval_fairseq_hallucination_max', action='store_true')
    parser.add_argument('--eval_fairseq_1_source_max', action='store_true')
    parser.add_argument('--eval_fairseq_1_source_supp_and_red_1_source', action='store_true')
    parser.add_argument('--eval_fairseq_1_source_supp_and_red_hall', action='store_true')

    main(parser.parse_args())