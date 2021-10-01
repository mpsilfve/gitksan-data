import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ..utils.gitksan_table_utils import get_char_counts, get_tag_feat_counts
from ..pkl_operations.pkl_io import store_pic_dynamic

def plot_character_distribution(frame):
    """Plot the (normalized) frequencies of characters in the paradigm data
    """
    char_counter = get_char_counts(frame, 'source_form')
    item_cnt_pairs = char_counter.items()
    items, counts = zip(*item_cnt_pairs)
    count_frame = pd.DataFrame({'char': items, 'count': counts})
    count_frame['count'] = count_frame['count'] / (count_frame['count'].sum())
    print(count_frame)
    count_frame = count_frame.sort_values(by='count', ascending=False)
    sns.barplot(x='char', y='count',  ci=None, data=count_frame)
    store_pic_dynamic(plt, 'paradigm_norm_char_freq')

def plot_feat_distribution(frame):
    """Plot the (normalized) frequencies of characters in the paradigm data
    """
    tag_feat_counter = get_tag_feat_counts(frame, 'source_tag')
    item_cnt_pairs = tag_feat_counter.items()
    items, counts = zip(*item_cnt_pairs)
    count_frame = pd.DataFrame({'tag_feat': items, 'count': counts})
    count_frame['count'] = count_frame['count'] / (count_frame['count'].sum())
    print(count_frame)
    count_frame = count_frame.sort_values(by='count', ascending=False)
    sns.barplot(x='tag_feat', y='count',  ci=None, data=count_frame)
    store_pic_dynamic(plt, 'paradigm_feat_freq')