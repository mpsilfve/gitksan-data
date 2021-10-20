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
            lemma = lines[0].strip().replace(' ', '_')
            msd = lines[-1].strip().replace(' ', '_')
            if len(lines) == 3:
                form = lines[1].strip().replace(' ', '_')
            elif len(lines) == 2:
                form = '-'
            else:
                print('Please make sure each line in your file is a tab separated 3-column entry.')
            pos = msd.split(';')[0]
            if '.' in pos:
                pos = pos.split('.')[0]
            #input = [letter for letter in lemma] + [pos, 'CANONICAL'] + ['#'] + [tag for tag in msd.split(';')]
            input = [letter for letter in lemma] + [tag for tag in msd.split(';')]
            output = [letter for letter in form]
            finput.write(' '.join(input) + '\n')
            foutput.write(' '.join(output) + '\n')