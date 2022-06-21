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

def plot_fullness_dist(paradigms):
    """Plot histogram with number of forms in x axis and number of paradigms with that many
    forms in the y-axis.

    Args:
        paradigms ([Paradigm]): Stream of (non-empty) paradigms
    """
    paradigm_sizes = []
    for paradigm in paradigms:
        paradigm_sizes.append(paradigm.count_num_forms())
    sns.histplot(data=paradigm_sizes)
    plt.xlabel("Number of entries in paradigm", fontsize='large')
    plt.ylabel("Number of paradigms", fontsize='large')
    plt.title("Distribution of fullness of paradigms", fontsize='large')
    plt.tick_params(axis='both', labelsize='large')
    store_pic_dynamic(plt, 'paradigm_num_forms_hist', 'results', True)


def plot_msd_distribution(frame):
    """Plot the frequency with which cells (msds) are filled in the paradigm data
    Args:
        frame (pd.DataFrame): |tag|form| 
    """
    sns.histplot(data=frame, x="tag")
    plt.title("Frequency of morphosyntactic descriptions (MSDs)", fontsize='large')
    plt.xlabel("MSD", fontsize='large')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("MSD count", fontsize='large')
    plt.tick_params(axis='both', labelsize='large')
    plt.tight_layout()
    store_pic_dynamic(plt, 'paradigm_msd_dist', 'results', True)

def plot_edit_distance_jitter(results_frame):
    sns.countplot(x='source_gold_dist', hue='model_type', data=results_frame)
    store_pic_dynamic(plt, 'source_gold_dist', 'results')