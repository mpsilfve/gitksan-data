from sklearn.model_selection import train_test_split

class Paradigm():
    def __init__(self, paradigm_info_list):
        self.roots = []
        self.entries = []
        for i in range(len(paradigm_info_list)):
            line = paradigm_info_list[i]
            if line.startswith("ROOT\t"):
                self.roots.append(line)
            else:
                self.entries.append(line)
    
    def is_empty(self):
        return all(list(map(lambda x: "\t_\t_\t_\t_" in x, self.entries)))
    
    def count_num_forms(self):
        return sum(list(map(lambda x: 0 if "\t_\t_\t_\t_" in x else 1, self.entries)))

def stream_all_paradigms():
    fname = "whitespace-inflection-tables-gitksan-productive.txt"
    with open(fname, 'r') as gp_f:
        gp_f.readline()
        line_num = 0
        for paradigm_block in gp_f:
            line = paradigm_block
            paradigm = []
            while line != "\n" and line != "": # (start of new paradigm) 
                paradigm.append(line)
                line = gp_f.readline()
            yield Paradigm(paradigm)

def extract_non_empty_paradigms():
    num_paradigms = 0
    non_empty_paradigms = []
    for paradigm in stream_all_paradigms():
        num_paradigms += 1
        if not paradigm.is_empty():
            non_empty_paradigms.append(paradigm)
    print(f"Read {num_paradigms} paradigms!")
    print(f"There are {len(non_empty_paradigms)} non-empty paradigms")
    return non_empty_paradigms

def obtain_train_dev_test_split(non_empty_paradigms, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    x_train, x_test = train_test_split(non_empty_paradigms, test_size=1 - train_ratio)
    x_val, x_test = train_test_split(x_test, test_size=test_ratio/(test_ratio + dev_ratio), shuffle=False)
    return x_train, x_val, x_test

def count_num_forms(paradigms):
    return sum([paradigm.count_num_forms() for paradigm in paradigms])

def test_read_paradigms():
    paradigms = extract_non_empty_paradigms()
    x_train, x_val, x_test = obtain_train_dev_test_split(paradigms)
    print(f"There are {len(x_train)} paradigms in the train set")
    print(f"There are {len(x_val)} paradigms in the dev set")
    print(f"There are {len(x_test)} paradigms in the test set")

    print(f"There are {count_num_forms(x_train)} forms in the train set")
    print(f"There are {count_num_forms(x_val)} forms in the dev set")
    print(f"There are {count_num_forms(x_test)} forms in the test set")