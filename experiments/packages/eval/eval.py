import pandas as pd
from ..fairseq.utils import distance

def eval_paradigm_accuracy_random(results_frame: pd.DataFrame, use_wf_accuracy: bool = False):
    results_frame = results_frame.sample(frac=1, random_state=42)
    eval_frame = results_frame.drop_duplicates(subset=['MSD_tgt', 'paradigm_i'], keep='first') # NOTE: this assumes the data was properly shuffled before
    total = 0
    num_correct = 0
    correct_paradigms = []
    for paradigm_i in set(eval_frame['paradigm_i'].values):
        paradigm = eval_frame.loc[eval_frame['paradigm_i'] == paradigm_i]
        if not use_wf_accuracy:
            if (paradigm['form_tgt'] == paradigm['predictions']).all():
                num_correct += 1
                correct_paradigms.append(paradigm_i)
        else:
            num_correct += sum(paradigm['form_tgt'] == paradigm['predictions'])

        if not use_wf_accuracy:
            total += 1
        else:
            total += len(paradigm)
    if not use_wf_accuracy:
        print(f"Correct paradigms: {sorted(correct_paradigms)}")
        print(f"PCFP accuracy: {num_correct/total}")
    else:
        print(f"Word form accuracy: {num_correct/total:.2f} out of {total} total word forms")

def eval_paradigm_accuracy_max(results_frame: pd.DataFrame, use_wf_accuracy: bool = False):
    results_frame = results_frame.sort_values(by='confidences', ascending=False)
    eval_frame = results_frame.drop_duplicates(subset=['MSD_tgt', 'paradigm_i'], keep='first') 
    total = 0
    num_correct = 0
    correct_paradigms = []
    for paradigm_i in set(eval_frame['paradigm_i'].values):
        paradigm = eval_frame.loc[eval_frame['paradigm_i'] == paradigm_i]
        
        if not use_wf_accuracy:
            if (paradigm['form_tgt'] == paradigm['predictions']).all():
                num_correct += 1
                correct_paradigms.append(paradigm_i)
        else:
            num_correct += sum(paradigm['form_tgt'] == paradigm['predictions'])

        if not use_wf_accuracy:
            total += 1
        else:
            total += len(paradigm)
    if not use_wf_accuracy:
        print(f"Correct paradigms: {sorted(correct_paradigms)}")
        print(f"PCFP accuracy: {num_correct/total}")
    else:
        print(f"Word form accuracy: {num_correct/total:.2f} out of {total} total word forms")

def eval_accuracy_oracle(results_frame: pd.DataFrame, use_wf_accuracy: bool = False): 
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
        if not use_wf_accuracy:
            if (paradigm['form_tgt'] == paradigm['predictions']).all():
                num_correct += 1
                correct_paradigms.append(paradigm_i)
        else:
            num_correct += sum(paradigm['form_tgt'] == paradigm['predictions'])

        if not use_wf_accuracy:
            total += 1
        else:
            total += len(paradigm)
    if not use_wf_accuracy:
        print(f"Correct paradigms: {sorted(correct_paradigms)}")
        print(f"PCFP accuracy: {num_correct/total}")
    else:
        print(f"Word form accuracy: {num_correct/total:.2f} out of {total} total word forms")
    return results_frame

def eval_accuracy_majority_vote(results_frame: pd.DataFrame, use_wf_accuracy: bool = False):
    """Evaluates morphological reinflection accuracy using a majority vote decision.

    Args: 
        results_frame: DataFrame with inputs (source form and msd, target msd) and gold target and predictions.
    """
    total = 0
    num_correct = 0
    correct_paradigms = []
    for paradigm_i in set(results_frame['paradigm_i'].values):
        paradigm_frame = results_frame.loc[results_frame['paradigm_i'] == paradigm_i]
        slot_grouped_paradigm_frame = paradigm_frame.groupby('MSD_tgt')
        most_common_predictions = slot_grouped_paradigm_frame['predictions'].agg(lambda s: s.mode().iloc[0]).values
        gold_targets = slot_grouped_paradigm_frame['form_tgt'].agg(pd.Series.sample).values
        if (gold_targets == most_common_predictions).all():
            num_correct += 1
            correct_paradigms.append(paradigm_i)
        total += 1
    print(f"Majority vote PCFP accuracy: {num_correct/total}")
    print(f"Majority vote correct paradigms: {sorted(correct_paradigms)}")

def find_off_by_one_predictions_max(results_frame: pd.DataFrame) -> pd.DataFrame:
    results_frame = results_frame.sort_values(by='confidences', ascending=False)
    eval_frame = results_frame.drop_duplicates(subset=['MSD_tgt', 'paradigm_i'], keep='first') 
    eval_frame['off_by_1'] = eval_frame[['predictions', 'form_tgt']].apply(lambda row: distance(row.predictions, row.form_tgt) == 1, axis=1)
    return eval_frame[eval_frame['off_by_1']]
    

        