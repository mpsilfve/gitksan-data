import pandas as pd
from ..utils.utils import map_list

def combine_tags(source_tag, target_tag):
    source_feats = source_tag.split(";")
    target_feats = target_tag.split(";")
    source_feats = list(map(lambda feat: f"IN:{feat}", source_feats))
    target_feats = list(map(lambda feat: f"OUT:{feat}", target_feats))

    source_tag_str = ";".join(source_feats)
    target_tag_str = ";".join(target_feats)
    return f"{source_tag_str};{target_tag_str}"

def reformat(fname, finputname, foutputname):
    """will turn all English train, dev, (test if there is test data) into the format Fairseq requires,
       and store the reformatted data in the current directory.

    Args:
        fname (str): SIGMORPHON 2020 shared task 0 data format.
        finputname (str): [description]
        foutputname (str): [description]
    """
    with open(fname) as f, \
        open(finputname, 'w') as finput, \
        open(foutputname, 'w') as foutput:
        for line in f:
            lines = line.strip().split('\t')
            lemma = lines[0].strip()
            form = lines[1].strip()
            msd = lines[2].strip()
            input = [letter for letter in lemma] + [tag for tag in msd.split(';')[1:]] # NOTE: I don't include POS (position 0 in msd) for Gitksan. Should be changed for other languages.
            output = [letter for letter in form]
            finput.write(' '.join(input) + '\n')
            foutput.write(' '.join(output) + '\n')

def reformat_from_frame(inflection_frame: pd.DataFrame, finputname: str, foutputname: str):
    with open(finputname, 'w') as finput, \
        open(foutputname, 'w') as foutput:
        # for line in f:
        for _, entry in inflection_frame.iterrows(): # TODO: needs to be fixed
            source = entry.form_src 
            tgt = entry.form_tgt
            msd_src = entry.MSD_src
            msd_tgt = entry.MSD_tgt
            msd = combine_tags(msd_src, msd_tgt)
            input = [letter for letter in source] + [tag for tag in msd.split(';')[1:]] 
            output = [letter for letter in tgt] 
            finput.write(' '.join(input) + '\n')
            foutput.write(' '.join(output) + '\n')

def produce_fairseq_data(train_fname, dev_fname, test_seen_fname, test_unseen_fname, dir_prefix):
    add_dir_prefix = lambda s: f"{dir_prefix}/{s}"
    add_dir_prefix_fairseq = lambda s: f"{dir_prefix}/fairseq/{s}"
    input_fnames = map_list(add_dir_prefix_fairseq, ["gitksan-train.src", "gitksan-dev.src", "gitksan-seen-test.src", "gitksan-unseen-test.src"])
    output_fnames = map_list(add_dir_prefix_fairseq, ["gitksan-train.tgt", "gitksan-dev.tgt", "gitksan-seen-test.tgt", "gitksan-unseen-test.tgt"])
    fnames = map_list(add_dir_prefix, [train_fname, dev_fname, test_seen_fname, test_unseen_fname])
    for i in range(len(input_fnames)):
        input_fname = input_fnames[i]
        output_fname = output_fnames[i]
        fname = fnames[i]
        reformat(fname, input_fname, output_fname)