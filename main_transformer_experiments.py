import pandas as pd
import argparse
import subprocess

from packages.fairseq.fairseq_format import produce_fairseq_data
from packages.fairseq.utils import extract_hypotheses, extract_hypotheses_mbr
from packages.utils.gitksan_table_utils import get_target_to_paradigm_mapping, extract_non_empty_paradigms, convert_inflection_file_to_frame, get_paradigm_inds
from packages.utils.utils import map_list
from packages.eval.eval import eval_paradigm_accuracy_random, eval_paradigm_accuracy_max
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

def eval_fairseq_cross_table(inflection_fname, prediction_fname, num_hypotheses):
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
    
def eval_fairseq_1_source_random(inflection_fname, prediction_fname, num_hypotheses):
    """Conducts PCFP evaluation on the cross-table data augmentation format =)
    """
    # inflection_fname = "data/spreadsheets/seen_unseen_split_w_root/gitksan_productive_seen.test"
    # inflection_fname = "data/spreadsheets/seen_unseen_split_w_root_cross_table/gitksan_productive_seen.test"
    frame = convert_inflection_file_to_frame(inflection_fname)

    # prediction_fname = "results/2021-11-19/1_source/results_seen_test.txt"
    predictions, confidences = extract_hypotheses(prediction_fname, num_hypotheses)
    frame['predictions'] = predictions
    frame['confidences'] = confidences

    # all_paradigms = extract_non_empty_paradigms("whitespace-inflection-tables-gitksan-productive.txt")
    # target_to_paradigm_mapping = get_target_to_paradigm_mapping(all_paradigms)

    # frame['paradigm_i'] = frame.apply(lambda row: target_to_paradigm_mapping[f'{row.target}_{row.target_msd}'], axis=1)

    eval_paradigm_accuracy(frame)
    return frame

def eval_fairseq_1_source_max(inflection_fname, prediction_fname, num_hypotheses):
    frame = convert_inflection_file_to_frame(inflection_fname)

    predictions, confidences = extract_hypotheses(prediction_fname, num_hypotheses)
    frame['predictions'] = predictions
    frame['confidences'] = confidences

    eval_paradigm_accuracy_max(frame)
    return frame

def eval_fairseq_1_source_random_mbr(inflection_fname, prediction_fname, num_hypotheses):
    frame = convert_inflection_file_to_frame(inflection_fname)
    predictions, confidences = extract_hypotheses_mbr(prediction_fname, num_hypotheses)
    # print(predictions)
    frame['predictions'] = predictions
    frame['confidences'] = confidences
    print(frame)
    eval_paradigm_accuracy_random(frame)
    # return frame

def eval_fairseq_1_source_max_mbr(inflection_fname, prediction_fname, num_hypotheses):
    frame = convert_inflection_file_to_frame(inflection_fname)
    predictions, confidences = extract_hypotheses_mbr(prediction_fname, num_hypotheses)
    # print(predictions)
    frame['predictions'] = predictions
    frame['confidences'] = confidences
    print(frame)
    eval_paradigm_accuracy_max(frame)
    # return frame

def optimize_temperature_global():
    results_mini_fname = 'results/2021-12-10/results_logits_train_dev.txt'
    dev_fname = 'data/spreadsheets/seen_unseen_split_w_root/gitksan_productive.dev'
    with open(results_mini_fname) as results_f:
        char_to_i, inp_to_frame = parse_results_w_logit_file(results_mini_fname, dev_fname)
    hyp_char_frame = expand_to_char_frame(inp_to_frame)
    model = ModelWithTemperature(hyp_char_frame, char_to_i)
    model.set_temperature()

def main(args):
    if args.produce_fairseq_data_cross_table:
        produce_fairseq_data_cross_table()
    elif args.produce_fairseq_data_regular:
        produce_fairseq_data_regular()
    elif args.produce_fairseq_data_hall:
        produce_fairseq_data_hall()
    elif args.eval_fairseq_cross_table:
        inflection_fname_prefix = "data/spreadsheets/seen_unseen_split_w_root_cross_table"
        prediction_fname_prefix = "results/2021-11-22/cross_table"
        eval_fairseq_cross_table(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test.txt", 5)
        eval_fairseq_cross_table(f"{inflection_fname_prefix}/gitksan_productive_unseen.test", f"{prediction_fname_prefix}/results_unseen_test.txt", 5)
    elif args.eval_fairseq_1_source_random:
        inflection_fname_prefix = "data/spreadsheets/seen_unseen_split_w_root"
        prediction_fname_prefix = "results/2021-11-19/1_source"
        eval_fairseq_1_source_random(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test.txt", 5)
        eval_fairseq_1_source_random(f"{inflection_fname_prefix}/gitksan_productive_unseen.test", f"{prediction_fname_prefix}/results_unseen_test.txt", 4)
    elif args.eval_fairseq_1_source_max:
        inflection_fname_prefix = "data/spreadsheets/seen_unseen_split_w_root"
        prediction_fname_prefix = "results/2021-11-19/1_source"
        eval_fairseq_1_source_max(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test_sampling.txt", 5)
        eval_fairseq_1_source_max(f"{inflection_fname_prefix}/gitksan_productive_unseen.test", f"{prediction_fname_prefix}/results_unseen_test_sampling.txt", 4)
    elif args.eval_fairseq_1_source_random_mbr:
        inflection_fname_prefix = "data/spreadsheets/seen_unseen_split_w_root"
        prediction_fname_prefix = "results/2021-12-03"
        # eval_fairseq_1_source_random_mbr(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test_sampling.txt", 20)
        eval_fairseq_1_source_random_mbr(f"{inflection_fname_prefix}/gitksan_productive_unseen.test", f"{prediction_fname_prefix}/results_unseen_test_sampling.txt", 20)
    elif args.eval_fairseq_1_source_random_mbr_platt_scaled:
        inflection_fname_prefix = "data/spreadsheets/seen_unseen_split_w_root"
        prediction_fname_prefix = "results/2021-12-10"
        # eval_fairseq_1_source_random_mbr(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test_sampling.txt", 20)
        # eval_fairseq_1_source_random_mbr(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test_sampling_platt_scaled.txt", 20)
        eval_fairseq_1_source_random_mbr(f"{inflection_fname_prefix}/gitksan_productive_unseen.test", f"{prediction_fname_prefix}/results_unseen_test_sampling_platt_scaled.txt", 20)
    elif args.eval_fairseq_1_source_max_mbr:
        inflection_fname_prefix = "data/spreadsheets/seen_unseen_split_w_root"
        prediction_fname_prefix = "results/2021-12-03"
        # eval_fairseq_1_source_max_mbr(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test_sampling.txt", 20)
        # eval_fairseq_1_source_max_mbr(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test_sampling.txt", 20)
        eval_fairseq_1_source_max_mbr(f"{inflection_fname_prefix}/gitksan_productive_unseen.test", f"{prediction_fname_prefix}/results_unseen_test_sampling.txt", 20)
        # eval_fairseq_1_source_random_mbr(f"{inflection_fname_prefix}/gitksan_productive_unseen.test", f"{prediction_fname_prefix}/results_unseen_test_sampling.txt", 20)
        # eval_fairseq_1_source_random(f"{inflection_fname_prefix}/gitksan_productive_unseen.test", f"{prediction_fname_prefix}/results_unseen_test.txt", 4)
    elif args.eval_fairseq_1_source_max_mbr_platt_scaled:
        inflection_fname_prefix = "data/spreadsheets/seen_unseen_split_w_root"
        prediction_fname_prefix = "results/2021-12-10"
        # eval_fairseq_1_source_max_mbr(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test_sampling.txt", 20)
        # eval_fairseq_1_source_max_mbr(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test_sampling.txt", 20)
        # eval_fairseq_1_source_max_mbr(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test_sampling.txt", 20)
        # eval_fairseq_1_source_max_mbr(f"{inflection_fname_prefix}/gitksan_productive_seen.test", f"{prediction_fname_prefix}/results_seen_test_sampling_platt_scaled.txt", 20)
        eval_fairseq_1_source_max_mbr(f"{inflection_fname_prefix}/gitksan_productive_unseen.test", f"{prediction_fname_prefix}/results_unseen_test_sampling_platt_scaled.txt", 20)
    elif args.optimize_temperature_global:
        optimize_temperature_global()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--produce_fairseq_data_cross_table', action='store_true')
    parser.add_argument('--produce_fairseq_data_regular', action='store_true')
    parser.add_argument('--produce_fairseq_data_hall', action='store_true')
    parser.add_argument('--eval_fairseq_cross_table', action='store_true')
    parser.add_argument('--eval_fairseq_1_source_random', action='store_true')
    parser.add_argument('--eval_fairseq_1_source_max', action='store_true')
    parser.add_argument('--extract_hypotheses_mbr', action='store_true')
    parser.add_argument('--eval_fairseq_1_source_random_mbr', action='store_true')
    parser.add_argument('--eval_fairseq_1_source_random_mbr_platt_scaled', action='store_true')
    parser.add_argument('--eval_fairseq_1_source_max_mbr', action='store_true')
    parser.add_argument('--eval_fairseq_1_source_max_mbr_platt_scaled', action='store_true')
    parser.add_argument('--optimize_temperature_global', action='store_true')
    main(parser.parse_args())