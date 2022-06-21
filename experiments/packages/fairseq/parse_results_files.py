import pandas as pd
import doctest
import re
from torch import Tensor, Size, cat, stack

def _match_size_line(line):
    """

    >>> _match_size_line("torch.Size([5, 1, 32])\\n")
    True
    >>> _match_size_line("-0.1640, -0.1886, -0.0604, -0.2151, -0.1832, -0.0928, -0.1047\\n")
    False

    Args:
        line ([str]): Line in fairseq results file.
    

    Returns:
        bool: Whether line is of the form `torch.Size([5, 1, 32])`
    """
    return re.search(r"torch\.Size", line) is not None

def _match_end_logit_block(line):
    """Match the end of a block of logits.

    >>> _match_end_logit_block("          -0.1094, -0.1201, -0.0794, -0.0997]]])\\n")
    True
    >>> _match_end_logit_block("          -0.1840,  0.3588, -0.1264, -0.0902, -0.1629,  0.0911, -0.1380,\\n")
    False
    """
    return re.search(r"\]\]\]", line) is not None

def _skip_line(prediction_f):
    prediction_f.readline()

def _match_start_logit_block(line):
    """Match the start of a block of logits

    >>> _match_start_logit_block("tensor([[[-0.0817, -0.0810, -0.1352, -0.0812, -0.1763, -0.2682, -0.1134,\\n")
    True
    >>> _match_start_logit_block("          -0.1840,  0.3588, -0.1264, -0.0902, -0.1629,  0.0911, -0.1380,\\n")
    False
    """
    return line.startswith("tensor([[[")

def obtain_logit_block(pred_f, start_line):
    """[summary]

    Args:
        pred_f (str): Prediction/result file (opened)

    Returns:
        [torch.Tensor]: 3D tensor containing logits.
    """
    block_str = start_line
    line = pred_f.readline()
    while not _match_end_logit_block(line): 
        block_str += line
        line = pred_f.readline()
        pass 
    block_str += line
    return 'T' + block_str[1:]
    # TODO: substitute the t in tensor with T. 
    # return eval(block_str)

def _match_hypothesis_block_start(block):
    """Returns True if line (block) is the first line of fairseq hypothesis 
    generation. 

    >>> _match_hypothesis_block_start('S-0	n e k s IN:ROOT OUT:ROOT OUT:ATTR')
    True
    >>> _match_hypothesis_block_start('           4.1284e-03, -7.9074e-02, -1.2347e-01, -2.4562e-02, -1.3447e-01,\\n')
    False

    Args:
        block ([str]): line in fairseq results file.
    """
    return block.startswith('S-') 

def _reshape_logit_tensors(arr_tensors):
    """Reshapes fairseq output to align with beam size. 

    Args:
        arr_tensors ([torch.Tensor]): {max_len_sequence} length array with {beam size} x {vocab size}  
    """
    beam_size = arr_tensors[0].shape[0]
    max_len = len(arr_tensors)
    hypothesis_tensors = []
    for i in range(beam_size):
        hypothesis_tensor = []
        for j in range(max_len):
            i_char_j_pos_logits = arr_tensors[j][i] 
            hypothesis_tensor.append(i_char_j_pos_logits)
        hypothesis_tensor = stack(hypothesis_tensor)
        # assert hypothesis_tensor.shape == Size([8,32]), f"hypothesis tensor shape is {hypothesis_tensor.shape}"
        hypothesis_tensors.append(hypothesis_tensor)
    return hypothesis_tensors

def _obtain_beam_tensors(prediction_f, cur_line):
    beam_tensors = []
    while not _match_hypothesis_block_start(cur_line): 
        if _match_start_logit_block(cur_line):
            beam_tens = obtain_logit_block(prediction_f, cur_line) # going to load a (BeamSize x batch size x dictionary size)
            beam_tens = eval(beam_tens)
            shape_line = prediction_f.readline().strip()
            shape = eval(shape_line[shape_line.index('Size'): ])
            assert shape == beam_tens.shape
            beam_tensors.append(beam_tens)
            cur_line = prediction_f.readline()
    return beam_tensors, cur_line

def extract_hypothesis(prediction_f, line):
    hypothesis = ''.join(line.split('\t')[2].strip().split(' '))
    _skip_line(prediction_f) # skip D-# line
    _skip_line(prediction_f) # skip likelihood line
    return hypothesis, prediction_f.readline()

def _parse_input_line(line):
    """[summary]

    >>> _parse_input_line("S-1\tj i s j a h l i t IN:ROOT IN:PL IN:TR IN:3.II OUT:ROOT OUT:TR OUT:3.II\\n")
    'jisjahlit IN:ROOT IN:PL IN:TR IN:3.II OUT:ROOT OUT:TR OUT:3.II'

    Args:
        line (str): input line tokenized on characters on tags by fairseq
    """
    input_line = line.split('\t')[1]
    input_tokens = input_line.strip().split(' ')
    i = 0
    inp_form = ''
    inp_tag = ''
    while not 'IN' in input_tokens[i]: # checking for IN; assuming that no other char that is I in vocab.
        inp_form += input_tokens[i]
        i += 1  
    for j in range(i, len(input_tokens)):
        inp_tag += input_tokens[j] + ' '

    return f"{inp_form} {inp_tag[0:-1]}"

def obtain_hypotheses(prediction_f, cur_line):
    inp = _parse_input_line(cur_line) 
    _skip_line(prediction_f) # skip timing 
    line = prediction_f.readline()
    hypotheses = []
    while not (_match_start_logit_block(line) or line == ''):
        hypothesis, line  = extract_hypothesis(prediction_f, line)
        hypotheses.append(hypothesis)
    return inp, hypotheses, line

def parse_results_w_logit_file(results_fname, input_fname):
    """
    Args:
        results_fname (str): filename for results produced from fairseq
        input_fname (str): input filename produced from fairseq

    Returns:
        pd.DataFrame 
    """


    with open(results_fname) as prediction_f, open(input_fname) as input_f:
        cur_line = prediction_f.readline()# NOTE: skip first line; just says "Printing tgt dict!"
        char_to_i = eval(prediction_f.readline().strip()) # obtains token to index mapping

        cur_line = prediction_f.readline()
        input_logits_hypotheses_frames = []
        pred_line = 0
        while cur_line != '':
            print(f"Processing example number: {pred_line}")
            beam_tensors, cur_line = _obtain_beam_tensors(prediction_f, cur_line)
            beam_tensors = [tensor.squeeze(dim=1) for tensor in beam_tensors]
            beam_tensors = _reshape_logit_tensors(beam_tensors)
            inp, hypotheses, cur_line = obtain_hypotheses(prediction_f, cur_line)
            print(len(beam_tensors))
            print(len(hypotheses))

            gold_form = input_f.readline().strip().split('\t')[1]
            input_logits_hypotheses_frames.append(
                pd.DataFrame({
                    'pred_line': [pred_line] * len(hypotheses),
                    'input': [inp] * len(hypotheses),
                    'hypothesis': hypotheses, 
                    'logits': beam_tensors, 
                    'gold': [gold_form] * len(hypotheses)
                })
            )
            pred_line += 1
            # input_to_logits_hypotheses_frame[inp] = pd.DataFrame({
            #     'hypothesis': hypotheses, 
            #     'logits': beam_tensors
            # })
        return char_to_i, pd.concat(input_logits_hypotheses_frames)

def expand_to_char_frame(input_to_hyp_logit_frame):
    """
    Args:
        input_hyp_logit_frame (pd.DataFrame): DataFrame with columns: pred_line|input|hypothesis|logits|gold

    Returns:
        pd.DataFrame: DataFrame ready for use in calibrating temperature.
    """
    input_tophyp_logit_frame = input_to_hyp_logit_frame.drop_duplicates(subset='pred_line', keep='first')
    expanded_frames = []
    def _expand_to_char(row):
        hyp_chars = []
        gold_chars = []
        pos_inds = []
        pred_line_nums = []
        less_len = min(len(row.hypothesis), len(row.gold)) # TODO: should go into the report. 
        input_strs = []
        logits = []
        # TODO: This could be improved, especially if we're overgenerating
        for i in range(less_len):
            hyp_chars.append(row.hypothesis[i])
            gold_chars.append(row.gold[i])
            pos_inds.append(i)
            input_strs.append(row.input)
            logits.append(row.logits[i])
            pred_line_nums.append(row.pred_line)
        expanded_frame = pd.DataFrame(
            {
                'input': input_strs, 
                'pos_i': pos_inds, 
                'pred_char': hyp_chars, 
                'gold_char': gold_chars,
                'logits': logits,
                'pred_line_num': pred_line_nums
            }
        )
        expanded_frames.append(expanded_frame)
    input_tophyp_logit_frame.apply(_expand_to_char, axis=1)
    return pd.concat(expanded_frames)

if __name__ == "__main__":
    doctest.testmod()