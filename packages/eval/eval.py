from ..fairseq.utils import distance

def eval_paradigm_accuracy_random(results_frame):
    print(len(results_frame))
    results_frame = results_frame.sample(frac=1, random_state=42)
    eval_frame = results_frame.drop_duplicates(subset=['MSD_tgt', 'paradigm_i'], keep='first') # NOTE: this assumes the data was properly shuffled before
    total = 0
    num_correct = 0
    correct_paradigms = []
    for paradigm_i in set(eval_frame['paradigm_i'].values):
        paradigm = eval_frame.loc[eval_frame['paradigm_i'] == paradigm_i]
        if (paradigm['form_tgt'] == paradigm['predictions']).all():
            num_correct += 1
            correct_paradigms.append(paradigm_i)
        total += 1
    print(f"Correct paradigms: {sorted(correct_paradigms)}")
    print(f"PCFP accuracy: {num_correct/total}")

def eval_paradigm_accuracy_max(results_frame):
    results_frame = results_frame.sort_values(by='confidences', ascending=False)
    print(results_frame)
    # TODO: shouldn't this be target MSD?
    eval_frame = results_frame.drop_duplicates(subset=['MSD_tgt', 'paradigm_i'], keep='first') 
    total = 0
    num_correct = 0
    correct_paradigms = []
    for paradigm_i in set(eval_frame['paradigm_i'].values):
        paradigm = eval_frame.loc[eval_frame['paradigm_i'] == paradigm_i]
        if (paradigm['form_tgt'] == paradigm['predictions']).all():
            num_correct += 1
            correct_paradigms.append(paradigm_i)
        total += 1
    print(f"PCFP accuracy: {num_correct/total}")
    print(f"Correct paradigms: {sorted(correct_paradigms)}")

def eval_accuracy_oracle(results_frame): 
    """Evaluates PCFP accuracy using an oracle method: as long as any of the predictions are correct, for a 
    paradigm/target combination, we consider all of them correct.
    """
    results_frame = results_frame.copy()
    results_frame['bleu'] = results_frame[['predictions', 'form_tgt']].apply(lambda row: distance(row.predictions, row.form_tgt), axis=1)
    results_frame = results_frame.sort_values(by='bleu', ascending=True)
    eval_frame = results_frame.drop_duplicates(subset=['MSD_tgt', 'paradigm_i'], keep='first') 
    total = 0
    num_correct = 0
    correct_paradigms = []
    for paradigm_i in set(eval_frame['paradigm_i'].values):
        paradigm = eval_frame.loc[eval_frame['paradigm_i'] == paradigm_i]
        if (paradigm['form_tgt'] == paradigm['predictions']).all():
            num_correct += 1
            correct_paradigms.append(paradigm_i)
        total += 1
    print(f"Oracle PCFP accuracy: {num_correct/total}")
    print(f"Oracle correct paradigms: {sorted(correct_paradigms)}")
    return results_frame

def eval_accuracy_per_row(results_frame):
    """Evaluates morphological reinflection accuracy. 

    Args:
        results_frame (pd.DataFrame): DataFrame with inputs (source form and msd, target msd) and gold target and predictions. 
    """
    total = len(results_frame)
    correct = sum(results_frame['predictions'] == results_frame['form_tgt'])
    print(f"Morphological reinflection accuracy: {correct/total}")