from packages.utils.gitksan_table_utils import stream_all_paradigms
def count_num_paradigms_with_no_root_multiple_wfs():
    paradigms = []
    count = 0
    for paradigm in stream_all_paradigms('whitespace-inflection-tables-gitksan-productive.txt'):
        num_roots = paradigm.count_num_roots()
        num_wfs = paradigm.count_num_forms()
        if num_roots == 0 and num_wfs >= 2:
            print(paradigm.entries)
            count += 1
    print(count)

def count_num_paradigms_with_multiple_roots():
    paradigms = []
    count = 0
    for paradigm in stream_all_paradigms('whitespace-inflection-tables-gitksan-productive.txt'):
        num_roots = paradigm.count_num_roots()
        if num_roots >= 2:
            count += 1
    print(count)

def inspect_root_distribution():
    num_paradigms_w_root = 0
    num_paradigms_w_root_pl = 0
    num_paradigms_w_both = 0
    for paradigm in stream_all_paradigms('whitespace-inflection-tables-gitksan-productive.txt'):
        num_roots = paradigm.count_num_roots()
        has_root_pl = paradigm.has_root_pl()
        if num_roots >= 1:
            num_paradigms_w_root += 1
        if has_root_pl:
            num_paradigms_w_root_pl += 1
        if num_roots >= 1 and has_root_pl:
            num_paradigms_w_both += 1
        if has_root_pl and not num_roots >= 1:
            print(paradigm["ROOT-PL"])
    print(f"{num_paradigms_w_root} paradigms have a root")
    print(f"{num_paradigms_w_root_pl} paradigms have a plural root")
    print(f"{num_paradigms_w_both} paradigms have both a root and a plural root")

        
        