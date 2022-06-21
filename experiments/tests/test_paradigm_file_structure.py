from packages.utils.gitksan_table_utils import stream_all_paradigms

def test_no_consecutive_blank_lines():
    with open('whitespace-inflection-tables-gitksan-productive.txt') as inflection_tables_raw_f:
        is_newline = False 
        for line in inflection_tables_raw_f:
            if is_newline:
                if line == '\n':
                    assert False, 'There are two consecutive newlines in the raw inflection table file. This shouldnt happen'
                else:
                    is_newline = False
            if line == '\n':
                is_newline = True
            




