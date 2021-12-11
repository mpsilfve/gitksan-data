from ..utils.utils import map_list

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