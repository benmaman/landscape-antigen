import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.spatial import cKDTree
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from tools import *
from scipy.interpolate import splprep, splev
import numpy as np
import plotly.graph_objects as go

color_map = {
    'HK68': 'purple',         # hard purple
    'EN72': 'mediumseagreen', # blue-green
    'VI75': 'khaki',          # dirty yellow
    'TX77': 'saddlebrown',    # brown
    'BK79': 'darkgreen',      # green bottle
    'SI87': 'navy',           # blue navy
    'BE89': 'red',            # red
    'BE92': 'pink',           # pink lady
    'WU95': 'limegreen',      # green grass
    'SY97': 'cyan',           # cyan
    'FU02': 'gold',           # yellow
    'CA04': 'lawngreen',      # glare green
    'EI05': 'blue',           # blue
    'PE09': 'mediumorchid'    # purple
}


def clean_titer(t):
    if t == '*' or pd.isna(t):
        return np.nan  # or -1 if you prefer explicit undetectable marking
    if t=="<10":
        return -1
    elif isinstance(t, str):
        t = t.replace("=", "").replace(">", "").strip()
        try:
            return int(t)
        except:
            print(f"{t} couldnt have changed to int")
            return np.nan
    else:
        return int(t)
    

#merge data and normalziation
def convert_titer(titer):
    if titer==-1:
        return -1
    return np.log2(titer / 10)


def antibody_map(df_antigenic_map):
    # # Create a color palette with one unique color per cluster
    unique_clusters = df_antigenic_map['cluster'].unique()
    # palette = sns.color_palette("hsv", len(unique_clusters))  # or "tab10", "Set2", etc.
    # color_map = dict(zip(unique_clusters, palette))
    color_map = {
        'HK68': 'purple',         # hard purple
        'EN72': 'mediumseagreen', # blue-green
        'VI75': 'khaki',          # dirty yellow
        'TX77': 'saddlebrown',    # brown
        'BK79': 'darkgreen',      # green bottle
        'SI87': 'navy',           # blue navy
        'BE89': 'red',            # red
        'BE92': 'pink',           # pink lady
        'WU95': 'limegreen',      # green grass
        'SY97': 'cyan',           # cyan
        'FU02': 'gold',           # yellow
        'CA04': 'lawngreen',      # glare green
        'EI05': 'blue',           # blue
        'PE09': 'mediumorchid'    # purple
    }

    # Plot
    plt.figure(figsize=(10, 6))
    for cluster in unique_clusters:
        cluster_df = df_antigenic_map[df_antigenic_map['cluster'] == cluster]
        plt.scatter(cluster_df['AG coordinate 2'], cluster_df['AG coordinate 1'],
                    s=60, color=color_map[cluster], label=cluster)
        
        # Add cluster name text at the mean coordinate position
        x_mean = cluster_df['AG coordinate 2'].mean()
        y_mean = cluster_df['AG coordinate 1'].mean()-0.6
        plt.text(x_mean, y_mean, str(cluster), fontsize=11, weight='bold', 
                color=color_map[cluster], ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.6))

    # Optional: Add virus names (can be uncommented if needed)
    # for i in range(len(df)):
    #     plt.text(df['AG coordinate 2'][i], df['AG coordinate 1'][i], df['viruses'][i],
    #              fontsize=8, ha='right', va='bottom')

    # Customize the grid and axis
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Antigenic Coordinate 2')
    plt.ylabel('Antigenic Coordinate 1')
    plt.title('Antigenic Map of Viruses (Colored by Cluster)')
    plt.tight_layout()
    plt.show()


def plot_error_scatter(Z,predicted_z,cluster_label,unique_clusters):

    plt.figure(figsize=(6, 6))

    # Plot clusters
    for cluster in unique_clusters:
        cluster_idx = cluster_label == cluster
        plt.scatter(Z[cluster_idx], predicted_z[cluster_idx],
                    s=60, color=color_map[cluster], label=str(cluster),
                    edgecolors='k', alpha=0.7)


    # Add perfect match line y = x
    min_val = min(min(Z), min(predicted_z))
    max_val = max(max(Z), max(predicted_z))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match (y = x)')

    # Labels and final touches
    plt.xlabel('True Titer Values (Z)', fontsize=12)
    plt.ylabel('Predicted Titer Values', fontsize=12)
    plt.title('True vs Predicted Values by Cluster', fontsize=14)
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(Z.min()-0.5, Z.max()+0.5)
    plt.ylim(Z.min()-0.5, Z.max()+0.5)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=8)
    plt.tight_layout()
    plt.show()
