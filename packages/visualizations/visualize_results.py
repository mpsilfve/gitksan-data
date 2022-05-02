import numpy as np
import matplotlib.pyplot as plt

from ..pkl_operations.pkl_io import store_pic_dynamic

def prettify_graph(ax):
    ax.autoscale_view()
    plt.grid(True, c='lightgray', linewidth=1, alpha=0.9)
    plt.box(False)
    plt.tight_layout()

def visualize_max_results():
    fig, ax = plt.subplots(1,1)

    labels = [ "observed", "wug"]
    width = 0.3
    x = np.arange(len(labels))
    ax.bar(x - width/2,  [0.83, 0.44], width, label='standard')
    ax.bar(x + width/2,  [0.65, 0.55], width, label='augmented')
    ax.set_ylabel('Accuracy')
    ax.set_title('Scores by test condition and model type')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    prettify_graph(ax)
    store_pic_dynamic(plt, 'max_performance', 'results', True)
