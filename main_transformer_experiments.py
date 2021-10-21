from packages.fairseq.fairseq_format import produce_fairseq_data
import argparse

def produce_fairseq_data_cross_table():
    """Saves fairseq train/validation/test data
    """
    produce_fairseq_data("gitksan_productive.train", "gitksan_productive.dev", "gitksan_productive_seen.test", "gitksan_productive_unseen.test", "data/spreadsheets/seen_unseen_split_w_root_cross_table")

def main(args):
    if args.produce_fairseq_data_cross_table:
        produce_fairseq_data_cross_table()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--produce_fairseq_data_cross_table', action='store_true')
    main(parser.parse_args())