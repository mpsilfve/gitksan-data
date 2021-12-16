import pytest 
from packages.calibration.temperature_scale import _index_logits
from packages.fairseq.parse_results_files import parse_results_w_logit_file, expand_to_char_frame

def test_index_logits(hyp_char_frame):
    logits = _index_logits(hyp_char_frame, (0, 0))
    assert len(logits) == 32

@pytest.fixture()
def hyp_char_frame(): 
    results_mini_fname = 'results/2021-12-06/results_mini.txt'
    dev_fname = 'data/spreadsheets/seen_unseen_split_w_root/gitksan_productive.dev'
    with open(results_mini_fname) as results_f:
        inp_to_frame = parse_results_w_logit_file(results_mini_fname, dev_fname)
    return expand_to_char_frame(inp_to_frame)