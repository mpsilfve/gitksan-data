
def get_root_orthographic(root_entry):
    values = root_entry.split('\t')
    return values[3].strip()

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
    
    def is_empty_roots(self):
        return all(list(map(lambda x: "\t_\t_\t_\t_" in x, self.roots)))

    
    def get_roots(self):
        return ":".join([get_root_orthographic(root) for root in self.roots])
    
    def count_num_forms(self):
        entries_count = sum(list(map(lambda x: 0 if "\t_\t_\t_\t_" in x else 1, self.entries)))
        # roots_count = sum(list())
        return entries_count 
    
    def count_num_roots(self):
        roots_count = sum(list(map(lambda x: 0 if "\t_\t_\t_\t_" in x else 1, self.roots)))
        return roots_count

    
