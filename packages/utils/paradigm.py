import re
import pandas as pd
import unicodedata as ud
from .utils import map_list

def get_root_orthographic(root_entry):
    values = root_entry.split('\t')
    return values[3].strip()

def obtain_orthographic_value(entry):
    values = entry.split('\t') 
    return values[3].strip()

def obtain_tag(entry):
    values = entry.split('\t')
    return values[0].strip().replace("-", ";")

def obtain_eng_gloss_value(entry):
    return entry.split('\t')[1].strip()

def obtain_gitksan_gloss_value(entry):
    return entry.split('\t')[-1].strip()

def obtain_morph_value(entry):
    return entry.split('\t')[2].strip()

def is_empty_entry(entry):
    return "\t_\t_\t_\t_" in entry

def strip_accents(s):
    """Replace x̲ with X
               k̲ with K
               g̲ with G

    Important for application of the MNC tool, which doesn't handle diacritics well.

    Args:
        s (str): Gitksan word

    Returns:
        [str]: String with diacritics substituted for capitals.
    """
    clean_s = ""
    # return ''.join(c for c in unicodedata.normalize('NFD', s)
    #                 if unicodedata.category(c) != 'Mn')
    i = 0
    while i < len(s):
        ud_name_cur = ud.name(s[i])
        if ud_name_cur in ["latin small letter k".upper(), "latin small letter g".upper(), "latin small letter x".upper()]:
            if (i+1) < len(s):
                ud_name_next = ud.name(s[i+1])
                if ud_name_next == "combining low line".upper():
                    clean_s += s[i].upper() # replace diacritic with uppercase
                else:
                    clean_s += s[i] # k, g, x
        else:
            if ud_name_cur != "combining low line".upper():
                clean_s += s[i] # all other characters (e.g., letters and apostrophes)
        i += 1
    return clean_s

class Paradigm():
    def __init__(self, paradigm_info_list, paradigm_index):
        self.roots = []
        self.paradigm_index = paradigm_index
        self.entries = []
        for i in range(len(paradigm_info_list)):
            line = paradigm_info_list[i]
            if line.startswith("ROOT\t"):
                self.roots.append(line)
            else:
                self.entries.append(line)
        self.frame = self._to_dataframe()
    
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
    
    def get_msds_forms_sequence(self):
        msds_forms_lst = []
        for msd, group_frame in self.frame.groupby('MSD'): 
            group_forms = group_frame['form'].values
            msds_forms_lst.append((msd, group_forms))
        return msds_forms_lst
    
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
    
    # TODO: need to test this
    def has_root_pl(self):
        return any(map(lambda x: re.match(r"ROOT-PL\s[a-zA-Z]", x) is not None, self.entries))
    
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

    def _stream_msd_form_gloss_morph(self):
        all_lines = (self.roots + self.entries) 
        for i in range(len(all_lines)):
            entry = all_lines[i]
            if not is_empty_entry(entry):
                form = obtain_orthographic_value(entry)
                msd = obtain_tag(entry)
                eng_gloss = obtain_eng_gloss_value(entry) 
                gitksan_gloss = obtain_gitksan_gloss_value(entry)
                morph = obtain_morph_value(entry)
                yield msd, form, eng_gloss, gitksan_gloss, morph  
    
    def _to_dataframe(self):
        """Convert paradigm to DataFrame
        """
        msds = []
        forms = []
        eng_glosses = []
        gitksan_glosses = [] 
        morphs = []
        for msd, form, eng_gloss, gitksan_gloss, morph in self._stream_msd_form_gloss_morph(): 
            msds.append(msd)
            forms.append(form)
            eng_glosses.append(eng_gloss)
            gitksan_glosses.append(gitksan_gloss)
            morphs.append(morph)
        paradigm_i_copies = [self.paradigm_index] * len(msds)
        frame = pd.DataFrame(
            data={
                "MSD": msds,
                "form": map_list(strip_accents,forms ),
                "eng_gloss": map_list( strip_accents, eng_glosses),
                "gitksan_gloss": map_list(strip_accents, gitksan_glosses),
                "morph": morphs,
                "paradigm_i": paradigm_i_copies  
            }
        )
        frame = frame.drop_duplicates(subset=['MSD', 'form'], keep='first')
        return frame

    def has_multiple_entries_for_msd(self):
        return len(self.frame['MSD'].values) > len(set(self.frame['MSD'].values))
    
def generate_permutations(forms_lst):
    """

    >>> generate_permutations([['ayook'], ['ayookt'], ['ayoogam'], ['ayookdiit']])
    [['ayook', 'ayookt', 'ayoogam', 'ayookdiit']]
    >>> generate_permutations([['ayook', 'ayookt']])
    [[ayook], [ayookt]]

    Args:
        forms_lst ([[str]]): each sublist represents the possible options for an MSD.
    """
    all_results = []
    if len(forms_lst) == 1:
        forms_for_msd = forms_lst[0]
        for form in forms_for_msd:
            all_results.append([form])
    else:
        forms_for_msd = forms_lst[0]
        for form in forms_for_msd:
            for rest_forms in generate_permutations(forms_lst[1:]):
                all_results.append([form] + rest_forms)
    return all_results 