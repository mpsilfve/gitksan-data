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