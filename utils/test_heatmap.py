import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import numpy as np

def save_heatmap_fig(xlabels, ylabels, values, filename):
    fig, ax = plt.subplots()
#     a = np.random.uniform(0, 1, size=(10, 10))
    sns.heatmap(a, cmap='Blues', ax=ax)
    # sns.heatmap(a, cmap='Blues', linewidth=0.5)
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    plt.savefig(filename, bbox_inches='tight')
#     plt.show()

save_heatmap_fig(['1']*10, ['12']*10, np.random.uniform(0, 1, size=(10, 10)), "test2.pdf")