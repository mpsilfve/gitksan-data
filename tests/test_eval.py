import pandas as pd
def test_eval_paradigm_accuracy_max():
    frame = pd.DataFrame({
        "MSD_tgt": ["a", "a", "a", "b", "b", "b", "c", "c"],
        "prediction": ['run', 'run', 'running', 'try', 'try', 'tried', "beg", 'begged'],
        "form_tgt": ['ran', 'ran', 'ran', 'try', 'try', 'try', "beg", 'beg']
    })
    maj_preds = frame.groupby(["MSD_tgt"])['prediction'].agg(lambda s: s.mode().iloc[0]).values
    gold_tgts = frame.groupby(["MSD_tgt"])['form_tgt'].agg(pd.Series.sample).values
    print(maj_preds)
    print(gold_tgts)
    assert not (maj_preds == gold_tgts).all()