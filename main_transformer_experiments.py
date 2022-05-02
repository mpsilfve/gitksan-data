from os import system 
from os import makedirs
import pandas as pd
import argparse
from datetime import datetime
import subprocess
from scipy.stats import ttest_ind
from typing import List
from packages.eval.error_analysis import perform_suppletion_analysis

from packages.utils.constants import *
from packages.fairseq.utils import distance
from packages.pkl_operations.pkl_io import store_csv_dynamic
from packages.utils.create_data_splits import make_all_pairs_frame, add_hallucination_examples 
from packages.fairseq.fairseq_format import combine_tags, produce_fairseq_data, reformat_from_frame
from packages.fairseq.utils import extract_hypotheses, extract_hypotheses_mbr
from packages.utils.utils import map_list, try_makedirs
from packages.utils.gitksan_table_utils import get_target_to_paradigm_mapping, extract_non_empty_paradigms, convert_inflection_file_to_frame, get_paradigm_inds, get_all_suppletions, get_all_reduplications
from packages.eval.eval import eval_accuracy_oracle, eval_paradigm_accuracy_random, eval_paradigm_accuracy_max, eval_accuracy_majority_vote 
from packages.fairseq.parse_results_files import parse_results_w_logit_file, expand_to_char_frame
from packages.calibration.temperature_scale import ModelWithTemperature
from packages.visualizations.plot_summary_distributions import plot_edit_distance_jitter
from packages.visualizations.visualize_results import visualize_max_results

def convert_to_prediction_form(inflection_fname: str, prediction_fname: str, num_hypotheses: int, csv_fname:str):
    frame = pd.read_csv(inflection_fname)
    predictions, confidences = extract_hypotheses(prediction_fname, num_hypotheses)
    frame['predictions'] = predictions
    new_frame = frame[['form_src', 'form_tgt', 'MSD_src', 'MSD_tgt', 'predictions', 'paradigm_i']]
    new_frame['tag'] = new_frame.apply(lambda row: combine_tags(row.MSD_src, row.MSD_tgt), axis=1)

    new_frame[['form_src', 'tag', 'form_tgt', 'predictions', 'paradigm_i']].to_csv(f'{csv_fname}_predictions.csv')

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

    # print("==================Random accuracy===============")
    # eval_paradigm_accuracy_random(frame)
    # print("==================Oracle accuracy===============")
    # eval_accuracy_oracle(frame)
    # print("==================Max accuracy===============")
    # eval_paradigm_accuracy_max(frame)
    # print("==================Majority vote accuracy===========")
    # eval_accuracy_majority_vote(frame)

    print("==================Random accuracy===============")
    eval_paradigm_accuracy_random(frame, False)
    print("==================Oracle accuracy===============")
    eval_accuracy_oracle(frame, False)
    print("==================Max accuracy===============")
    eval_paradigm_accuracy_max(frame, False)
    print("==================Majority vote accuracy===========")
    eval_accuracy_majority_vote(frame, False)
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
    
def pull_hallucination_results():
    date = datetime.today().strftime('%Y-%m-%d')
    folder = f"results/{date}/onesource"
    try:
        makedirs(folder)
    except FileExistsError:
        pass
    for fname in RESULTS_FNAMES:
        system(f"scp {HALLUCINATION_RESULTS_PATH}/{fname} {folder}/{fname}")

def _collect_type_splits_frame(type_split_fnames: List[str]) -> pd.DataFrame:
    """Read in and return type split frames in a single DataFrame.

    Args:
        type_split_fnames (List[str]) 

    Returns:
        pd.DataFrame 
    """
    type_split_frames = []
    for type_split_fname in type_split_fnames:
        type_split_frame = pd.read_csv(type_split_fname)[["paradigm_i", "gitksan_gloss", "MSD", "eng_gloss", "morph"]]
        type_split_frame = type_split_frame.rename(columns={"MSD": "MSD_tgt"})
        type_split_frames.append(type_split_frame)
    type_split_frame = pd.concat(type_split_frames)
    return type_split_frame

def get_suppletive_results(results_frame: pd.DataFrame, type_split_fnames: List[str]) -> pd.DataFrame:
    """Get all suppletive predictions.

    Args:
        results_frame (pd.DataFrame): All predictions on Gitksan dataset for an inflection model.
        type_split_fnames ([str]): Type split filenames for the standard and challenge split.

    Returns:
        pd.DataFrame: DataFrame where all targets are suppletive plurals
    """
    type_split_frame = _collect_type_splits_frame(type_split_fnames)
    merge_frame = results_frame.merge(type_split_frame, on=["paradigm_i", "MSD_tgt"])
    supp_frame = get_all_suppletions(merge_frame) 
    return supp_frame

def get_reduplication_results(results_frame: pd.DataFrame, type_split_fnames: List[str]) -> pd.DataFrame: 
    """Get all reduplicative predictions.

    Args:
        results_frame (pd.DataFrame): 
        type_split_fnames (List[str]): 

    Returns:
        pd.DataFrame 
    """
    type_split_frame = _collect_type_splits_frame(type_split_fnames)
    merge_frame = results_frame.merge(type_split_frame, on=["paradigm_i", "MSD_tgt"])
    redup_frame = get_all_reduplications(merge_frame) 
    return redup_frame


def eval_fairseq_1_source_supp_and_red(results_frame: pd.DataFrame, type_split_fnames: List[str], train_type: str):

    print("================Suppletive Analysis=====================")

    # supp_frame = get_suppletive_results(results_frame, type_split_fnames) 
    # eval_paradigm_accuracy_max(supp_frame, True)

    # eval_accuracy_oracle(supp_frame, True)

    # Test
    # supp_frame_test = get_all_suppletions(type_split_frame)
    # assert (len(supp_frame_test[['MSD_tgt', 'paradigm_i']].drop_duplicates())) == (len(supp_frame[['MSD_tgt','paradigm_i']].drop_duplicates()))
    # Test

    print("================Reduplicative Analysis=====================")
    redup_frame = get_reduplication_results(results_frame, type_split_fnames)
    # eval_paradigm_accuracy_max(redup_frame, True)
    eval_accuracy_oracle(redup_frame, True)

    # # Test
    # redup_frame_test = get_all_reduplications(type_split_frame)
    # assert (len(redup_frame_test[['MSD_tgt', 'paradigm_i']].drop_duplicates())) == (len(redup_frame[['MSD_tgt','paradigm_i']].drop_duplicates()))
    # # Test

    # print_supp_frame = supp_frame[["form_tgt", "predictions", "form_src", "MSD_src", "MSD_tgt", "confidences", "paradigm_i"]]
    # print_supp_frame.groupby('paradigm_i').apply(pd.DataFrame.sort_values, 'confidences', ascending=False)
    # store_csv_dynamic(print_supp_frame, f"supp_frame_{train_type}")

    # print_redup_frame = redup_frame[["form_tgt", "predictions", "form_src", "MSD_src", "MSD_tgt", "confidences", "paradigm_i"]]
    # print_redup_frame.groupby('paradigm_i').apply(pd.DataFrame.sort_values, 'confidences', ascending=False)
    # store_csv_dynamic(print_redup_frame, f"redup_frame_{train_type}")

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

def pull_hallucination_no_supp_results():
    date = datetime.today().strftime('%Y-%m-%d')
    folder = f"results/{date}/onesource"
    try:
        makedirs(folder)
    except FileExistsError:
        pass
    for fname in RESULTS_FNAMES:
        system(f"scp {HALLUCINATION_NO_SUPP_RESULTS_PATH}/{fname} {folder}/{fname}")
    
def perform_error_analysis(results_frame: pd.DataFrame):
    # eval_fairseq_1_source_supp_and_red(all_results_frame, [f"{STD_CHL_SPLIT_PATH}/standard_test_frame.csv",f"{STD_CHL_SPLIT_PATH}/challenge_test_frame.csv"], "1_source")

    # print("===Dialectal variation error analysis===")
    # off_by_one_predictions_frame = find_off_by_one_predictions_max(results_frame)
    # print(off_by_one_predictions_frame[['form_tgt', 'predictions', 'form_src']])

    print("====Edit distance difference=====")
    results_frame = results_frame.sort_values(by='confidences', ascending=False)
    eval_frame = results_frame.drop_duplicates(subset=['MSD_tgt', 'paradigm_i'], keep='first') 
    source_gold_edit_distance_avg = eval_frame[['form_src', 'form_tgt']].apply(lambda row: distance(row.form_src, row.form_tgt), axis=1).mean()
    # source_gold_edit_distance_avg = results_frame[['form_src', 'form_tgt']].apply(lambda row: distance(row.form_src, row.form_tgt)).mean()

def _get_hall_results_frame():
    inflection_fname_prefix = f"{STD_CHL_SPLIT_PATH}/cross_product_hallucination"
    prediction_fname_prefix = "results/2022-02-03/onesource"
    challenge_results_frame = _get_results_frame(f"{inflection_fname_prefix}/challenge_test_frame.csv", f"{prediction_fname_prefix}/results_challenge_test.txt", 5)
    standard_results_frame = _get_results_frame(f"{inflection_fname_prefix}/standard_test_frame.csv", f"{prediction_fname_prefix}/results_standard_test.txt", 5)
    challenge_results_frame['test_condition_type'] = 'challenge'
    standard_results_frame['test_condition_type'] = 'standard'
    return pd.concat([challenge_results_frame, standard_results_frame])

def _get_one_source_results_frame():
    inflection_fname_prefix = f"{STD_CHL_SPLIT_PATH}/cross_product_source"
    prediction_fname_prefix = "results/2022-01-30/onesource"
    challenge_results_frame = _get_results_frame(f"{inflection_fname_prefix}/challenge_test_frame.csv", f"{prediction_fname_prefix}/results_challenge_test.txt", 5)
    standard_results_frame = _get_results_frame(f"{inflection_fname_prefix}/standard_test_frame.csv", f"{prediction_fname_prefix}/results_standard_test.txt", 5)
    challenge_results_frame['test_condition_type'] = 'challenge'
    standard_results_frame['test_condition_type'] = 'standard'
    return pd.concat([challenge_results_frame, standard_results_frame])

def _get_max_predictions(results_frame: pd.DataFrame) -> pd.DataFrame:
    results_frame = results_frame.sort_values(by='confidences', ascending=False)
    eval_frame = results_frame.drop_duplicates(subset=['MSD_tgt', 'paradigm_i'], keep='first') 
    return eval_frame

def perform_error_analysis_1_source():
    # print("===Suppletive and reduplication error analysis===")
    all_results_frame = _get_one_source_results_frame()
    perform_error_analysis(all_results_frame)

def perform_error_analysis_hall():
    # print("===Suppletive and reduplication error analysis===")
    all_results_frame = _get_hall_results_frame()
    perform_error_analysis(all_results_frame)
    # eval_fairseq_1_source_supp_and_red(all_results_frame, [f"{STD_CHL_SPLIT_PATH}/standard_test_frame.csv",f"{STD_CHL_SPLIT_PATH}/challenge_test_frame.csv"], "1_source")

def perform_combined_error_analysis():
    hall_results_frame = _get_max_predictions(_get_hall_results_frame())
    one_source_results_frame = _get_max_predictions(_get_one_source_results_frame())
    hall_results_frame['model_type'] = 'hall'
    one_source_results_frame['model_type'] = 'one_source'

    all_results_frame = pd.concat([hall_results_frame, one_source_results_frame])
    all_results_frame['source_gold_dist'] = all_results_frame[['form_src', 'form_tgt']].apply(lambda row: distance(row.form_src, row.form_tgt), axis=1)
    plot_edit_distance_jitter(all_results_frame)

def get_suppletive_ap():
    hall_results_frame = (_get_hall_results_frame())
    print(len(hall_results_frame))
    hall_supp_challenge_frame = get_suppletive_results(hall_results_frame, [f"{STD_CHL_SPLIT_PATH}/challenge_test_frame.csv", f"{STD_CHL_SPLIT_PATH}/standard_test_frame.csv"])
    print(hall_supp_challenge_frame)
    perform_suppletion_analysis(hall_supp_challenge_frame)

    # one_source_results_frame = (_get_one_source_results_frame())
    # one_source_supp_challenge_frame = get_suppletive_results(one_source_results_frame, [f"{STD_CHL_SPLIT_PATH}/challenge_test_frame.csv", f"{STD_CHL_SPLIT_PATH}/standard_test_frame.csv"])
    # # print(one_source_supp_challenge_frame)
    # perform_suppletion_analysis(one_source_supp_challenge_frame)

# TODO: fill in.
def count_num_suppletions():
    STD_CHL_SPLIT_PATH
    test_fnames = []

def visualize_final_results():
    visualize_max_results()

def count_num_paradigms():
    type_split_fnames = ["train_frame.csv", "standard_test_frame.csv", "challenge_test_frame.csv", "dev_frame.csv"]
    type_split_fnames = map_list(lambda x: f"{STD_CHL_SPLIT_PATH}/{x}", type_split_fnames)
    print(type_split_fnames)
    paradigm_i_lst = []
    num_forms = 0
    for fname in type_split_fnames:
        frame = pd.read_csv(fname)
        paradigm_i_lst.extend(frame["paradigm_i"].values)
        num_forms += len(frame)
    print(len(set(paradigm_i_lst)))
    print(num_forms)

def main(args):
    if args.eval_fairseq_1_source:
        inflection_fname_prefix = f"{STD_CHL_SPLIT_PATH}/cross_product_source"
        prediction_fname_prefix = "results/2022-02-26/onesource"
        eval_fairseq_1_source(f"{inflection_fname_prefix}/challenge_test_frame.csv", f"{prediction_fname_prefix}/results_challenge_test.txt", 5)
        eval_fairseq_1_source(f"{inflection_fname_prefix}/standard_test_frame.csv", f"{prediction_fname_prefix}/results_standard_test.txt", 5)
    elif args.eval_fairseq_hallucination:
        inflection_fname_prefix = f"{STD_CHL_SPLIT_PATH}/cross_product_hallucination"
        prediction_fname_prefix = "results/2022-02-03/onesource"
        eval_fairseq_1_source(f"{inflection_fname_prefix}/challenge_test_frame.csv", f"{prediction_fname_prefix}/results_challenge_test.txt", 5)
        eval_fairseq_1_source(f"{inflection_fname_prefix}/standard_test_frame.csv", f"{prediction_fname_prefix}/results_standard_test.txt", 5)
    elif args.eval_fairseq_hallucination_no_supp:
        inflection_fname_prefix = f"{STD_CHL_SPLIT_PATH}/cross_product_hallucination_no_supp"
        prediction_fname_prefix = "results/2022-02-10/onesource"
        eval_fairseq_1_source(f"{inflection_fname_prefix}/challenge_test_frame.csv", f"{prediction_fname_prefix}/results_challenge_test.txt", 5)
        eval_fairseq_1_source(f"{inflection_fname_prefix}/standard_test_frame.csv", f"{prediction_fname_prefix}/results_standard_test.txt", 5)
    elif args.eval_fairseq_1_source_supp_and_red_hall:
        inflection_fname_prefix = f"{STD_CHL_SPLIT_PATH}/cross_product_hallucination"
        prediction_fname_prefix = "results/2022-02-03/onesource"
        challenge_results_frame = _get_results_frame(f"{inflection_fname_prefix}/challenge_test_frame.csv", f"{prediction_fname_prefix}/results_challenge_test.txt", 5)
        standard_results_frame = _get_results_frame(f"{inflection_fname_prefix}/standard_test_frame.csv", f"{prediction_fname_prefix}/results_standard_test.txt", 5)
        all_results_frame = pd.concat([ standard_results_frame])
        eval_fairseq_1_source_supp_and_red(all_results_frame, [f"{STD_CHL_SPLIT_PATH}/standard_test_frame.csv"], "hall")
    # elif args.eval_fairseq_1_source_supp_and_red:
    #     inflection_fname_prefix = f"{STD_CHL_SPLIT_PATH}/cross_product_hallucination"
    #     prediction_fname_prefix = "results/2022-02-03/onesource"
    #     challenge_results_frame = _get_results_frame(f"{inflection_fname_prefix}/challenge_test_frame.csv", f"{prediction_fname_prefix}/results_challenge_test.txt", 5)
    #     standard_results_frame = _get_results_frame(f"{inflection_fname_prefix}/standard_test_frame.csv", f"{prediction_fname_prefix}/results_standard_test.txt", 5)
    #     all_results_frame = pd.concat([ standard_results_frame])
    #     eval_fairseq_1_source_supp_and_red(all_results_frame, [f"{STD_CHL_SPLIT_PATH}/standard_test_frame.csv"], "hall")
    elif args.convert_to_prediction_form:
        # inflection_fname_prefix = f"{STD_CHL_SPLIT_PATH}/cross_product_source"
        # prediction_fname_prefix = "results/2022-01-30/onesource"
        # convert_to_prediction_form(f"{inflection_fname_prefix}/challenge_test_frame.csv", f"{prediction_fname_prefix}/results_challenge_test.txt", 5, "challenge")
        inflection_fname_prefix = f"{STD_CHL_SPLIT_PATH}/cross_product_hallucination"
        prediction_fname_prefix = "results/2022-02-03/onesource"
        convert_to_prediction_form(f"{inflection_fname_prefix}/standard_test_frame.csv", f"{prediction_fname_prefix}/results_standard_test.txt", 5, "standard")
    elif args.generate_fairseq_files_all_pairs:
        generate_fairseq_files_all_pairs()
    elif args.generate_fairseq_files_hallucination:
        generate_fairseq_files_hallucination()
    elif args.pull_onesource_results:
        pull_onesource_results()
    elif args.pull_hallucination_results:
        pull_hallucination_results()
    elif args.pull_hallucination_no_supp_results:
        pull_hallucination_no_supp_results()
    elif args.perform_error_analysis_1_source:
        perform_error_analysis_1_source()
    elif args.perform_error_analysis_hall:
        perform_error_analysis_hall()
    elif args.perform_combined_error_analysis:
        perform_combined_error_analysis()
    elif args.get_suppletive_ap:
        get_suppletive_ap()
    elif args.visualize_final_results:
        visualize_final_results()
    elif args.count_num_paradigms:
        count_num_paradigms()

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pull_onesource_results', action='store_true')
    parser.add_argument('--pull_hallucination_results', action='store_true')
    parser.add_argument('--pull_hallucination_no_supp_results', action='store_true')
    parser.add_argument('--produce_fairseq_data_cross_table', action='store_true')
    parser.add_argument('--generate_fairseq_files_all_pairs', action='store_true')
    parser.add_argument('--get_suppletive_ap', action='store_true')
    parser.add_argument('--generate_fairseq_files_hallucination', action='store_true')
    parser.add_argument('--produce_fairseq_data_regular', action='store_true')
    parser.add_argument('--convert_to_prediction_form', action='store_true')
    parser.add_argument('--produce_fairseq_data_hall', action='store_true')
    parser.add_argument('--eval_fairseq_cross_table_random', action='store_true')
    parser.add_argument('--eval_fairseq_hallucination', action='store_true')
    parser.add_argument('--eval_fairseq_hallucination_no_supp', action='store_true')
    parser.add_argument('--eval_fairseq_1_source', action='store_true')
    parser.add_argument('--eval_fairseq_hallucination_max', action='store_true')
    parser.add_argument('--eval_fairseq_1_source_max', action='store_true')
    parser.add_argument('--eval_fairseq_1_source_supp_and_red_1_source', action='store_true')
    parser.add_argument('--eval_fairseq_1_source_supp_and_red_hall', action='store_true')
    parser.add_argument('--perform_error_analysis_1_source', action='store_true')
    parser.add_argument('--perform_error_analysis_hall', action='store_true')
    parser.add_argument('--perform_combined_error_analysis', action='store_true')
    parser.add_argument('--visualize_final_results', action='store_true')
    parser.add_argument('--count_num_paradigms', action='store_true')

    main(parser.parse_args())