import pandas as pd
import argparse

from packages.fairseq.fairseq_format import produce_fairseq_data
from packages.fairseq.utils import extract_hypotheses
from packages.utils.gitksan_table_utils import get_target_to_paradigm_mapping, extract_non_empty_paradigms, convert_inflection_file_to_frame, get_paradigm_inds
from packages.utils.utils import map_list
from packages.eval.eval import eval_paradigm_accuracy

def produce_fairseq_data_cross_table():
    """Saves fairseq train/validation/test data
    """
    produce_fairseq_data("gitksan_productive.train", "gitksan_productive.dev", "gitksan_productive_seen.test", "gitksan_productive_unseen.test", "data/spreadsheets/seen_unseen_split_w_root_cross_table")

def produce_fairseq_data_regular():
    """Produces fairseq format data for the `seen_unseen_split_w_root`.
    """
    produce_fairseq_data("gitksan_productive.train", "gitksan_productive.dev", "gitksan_productive_seen.test", "gitksan_productive_unseen.test", "data/spreadsheets/seen_unseen_split_w_root")

def eval_fairseq_cross_table():
    """Conducts PCFP evaluation on the cross-table data augmentation format =)
    """
    inflection_fname = "data/spreadsheets/seen_unseen_split_w_root_cross_table/gitksan_productive_seen.test"
    frame = convert_inflection_file_to_frame(inflection_fname)
    print(frame)

    # prediction_fname = "results/2021-10-28/results_unseen_test.txt"
    prediction_fname = "results/2021-10-28/results_seen_test.txt"
    predictions, confidences = extract_hypotheses(prediction_fname, 4)
    frame['predictions'] = predictions
    frame['confidences'] = confidences

    cross_tab_paradigm_inds = frame['paradigm_i'].values.copy()
    print(cross_tab_paradigm_inds)
    frame['paradigm_i'] = map_list( lambda x: int(x.split('_')[0]), cross_tab_paradigm_inds)
    frame['cross_table_i']= map_list( lambda x: int(x.split('_')[1]), cross_tab_paradigm_inds)

    # all_paradigms = extract_non_empty_paradigms("whitespace-inflection-tables-gitksan-productive.txt")
    # target_to_paradigm_mapping = get_target_to_paradigm_mapping(all_paradigms)

    # frame['paradigm_i'] = frame.apply(lambda row: target_to_paradigm_mapping[f'{row.target}_{row.target_msd}'], axis=1)

    eval_paradigm_accuracy(frame)


    return frame

def main(args):
    if args.produce_fairseq_data_cross_table:
        produce_fairseq_data_cross_table()
    elif args.produce_fairseq_data_regular:
        produce_fairseq_data_regular()
    elif args.eval_fairseq_cross_table:
        eval_fairseq_cross_table()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--produce_fairseq_data_cross_table', action='store_true')
    parser.add_argument('--produce_fairseq_data_regular', action='store_true')
    parser.add_argument('--eval_fairseq_cross_table', action='store_true')
    main(parser.parse_args())