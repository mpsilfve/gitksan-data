from torch import Size, Tensor
from packages.fairseq.parse_results_files import obtain_logit_block, _obtain_beam_tensors, extract_hypothesis, obtain_hypotheses, parse_results_w_logit_file, expand_to_char_frame

def test_obtain_logit_block():
    logit_block_fname = 'tests/fairseq/resources/logit_block.txt'
    with open(logit_block_fname) as logit_block_f:
        line = logit_block_f.readline()
        block = obtain_logit_block(logit_block_f, line)
        tens = eval(block)
        assert tens.shape == Size([5, 1, 32])
        assert block.startswith('T')

def test_obtain_beam_tensors():
    tensor_blocks_fname = 'tests/fairseq/resources/tensor_and_hypothesis_block_start.txt'
    with open(tensor_blocks_fname) as tensor_blocks_f:
        line = tensor_blocks_f.readline()
        tensors, _ = _obtain_beam_tensors(tensor_blocks_f, line)
        assert len(tensors) == 8
        for tens in tensors:
            assert tens.shape == Size([5, 1, 32])

def test_extract_hypothesis():
    hypothesis_fname = 'tests/fairseq/resources/hypothesis.txt'
    with open(hypothesis_fname) as hypothesis_f:
        line = hypothesis_f.readline()
        hypothesis, line = extract_hypothesis(hypothesis_f, line)
        assert hypothesis == 'jahlit'

def test_obtain_hypotheses():
    hypotheses_fname = 'tests/fairseq/resources/hypotheses.txt'
    with open(hypotheses_fname) as hypotheses_f:
        line = hypotheses_f.readline()
        inp, hypotheses, line = obtain_hypotheses(hypotheses_f, line)
        assert inp == 'jisjahlit IN:ROOT IN:PL IN:TR IN:3.II OUT:ROOT OUT:TR OUT:3.II'
        assert len(hypotheses) == 5
        assert hypotheses[0] == 'jahlit'     
        assert hypotheses[-1] == 'jabit'     

def test_parse_results_w_logit_file():
    results_mini_fname = 'results/2021-12-06/results_mini.txt'
    dev_fname = 'data/spreadsheets/seen_unseen_split_w_root/gitksan_productive.dev'
    with open(results_mini_fname) as results_f:
        inp_to_frame = parse_results_w_logit_file(results_mini_fname, dev_fname)
        assert len(inp_to_frame) == 10
        print(inp_to_frame)

def test_expand_to_char():
    results_mini_fname = 'results/2021-12-06/results_mini.txt'
    dev_fname = 'data/spreadsheets/seen_unseen_split_w_root/gitksan_productive.dev'
    with open(results_mini_fname) as results_f:
        inp_to_frame = parse_results_w_logit_file(results_mini_fname, dev_fname)
    expanded_frame = expand_to_char_frame(inp_to_frame)
    assert len(expanded_frame) == 6 + 6 
    print(expanded_frame)