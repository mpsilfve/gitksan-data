import re

def get_root_orthographic(root_entry):
    values = root_entry.split('\t')
    return values[3].strip()

def obtain_orthographic_value(entry):
    values = entry.split('\t') 
    return values[3].strip()

def obtain_tag(entry):
    values = entry.split('\t')
    return values[0].strip().replace("-", ";")

def is_empty_entry(entry):
    return "\t_\t_\t_\t_" in entry

class Paradigm():
    def __init__(self, paradigm_info_list, paradigm_index):
        # TODO: add unique identifier (index in the original data file).
        self.roots = []
        self.paradigm_index = paradigm_index
        self.entries = []
        for i in range(len(paradigm_info_list)):
            line = paradigm_info_list[i]
            if line.startswith("ROOT\t"):
                self.roots.append(line)
            else:
                self.entries.append(line)
    
    def __eq__(self, other):
        return self.roots == other.roots and self.entries == other.entries

    def __contains__(self, msd):
        msd_reg = re.compile(f"{msd}\t")
        for line in self.entries + self.roots:
            if not re.match(msd_reg, line) is None:
                return not "\t_\t_\t_\t_" in line

    def __getitem__(self, msd): 
        msd_reg = re.compile(f"{msd}\t")
        for line in self.entries + self.roots:
            if not re.match(msd_reg, line) is None:
                return obtain_orthographic_value(line)
        # for line in self.entries:
        #     if msd in 
    
    def is_empty(self):
        return all(list(map(lambda x: "\t_\t_\t_\t_" in x, self.entries)))

    def is_empty_roots(self):
        return all(list(map(lambda x: "\t_\t_\t_\t_" in x, self.roots)))
    
    def get_all_msds(self):
        return set([line.split('\t')[0] for line in (self.entries + self.roots)])
    
    def get_roots(self):
        return ":".join([get_root_orthographic(root) for root in self.roots])
    
    def count_num_forms(self):
        entries_count = sum(list(map(lambda x: 0 if "\t_\t_\t_\t_" in x else 1, self.entries)))
        # roots_count = sum(list())
        return entries_count 
    
    def count_num_roots(self):
        roots_count = sum(list(map(lambda x: 0 if "\t_\t_\t_\t_" in x else 1, self.roots)))
        return roots_count
    
    # TODO: test this
    def stream_form_tag_pairs(self, include_root=True):
        all_lines = (self.roots + self.entries) if include_root else self.roots
        for i in range(len(all_lines)):
            entry = all_lines[i]
            if not is_empty_entry(entry):
                form = obtain_orthographic_value(entry)
                tag = obtain_tag(entry)
                yield form, tag