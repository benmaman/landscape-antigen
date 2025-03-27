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
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
from typing import List

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


def plot_3d_antibody(X,Y,Z,C):

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Vertical gray impulses
    for x, y, z in zip(X, Y, Z):
        ax.plot([x, x], [y, y], [-1, z], color='gray', linewidth=1)

    # Colored points at the top
    ax.scatter(X, Y, Z, c=C, s=60, edgecolors='black', linewidths=0.5, depthshade=False)

    # Axes labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('log₂(HI titer / 10)', labelpad=30, fontsize=14)

    # Axis limits
    ax.set_xlim(-0.11, 1.2)
    ax.set_ylim(-0.1, 1.5)
    ax.set_zlim(-1, Z.max())

    # View and aspect
    ax.view_init(elev=15, azim=235)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1.01, 1, 1]))
    ax.set_box_aspect(aspect=(3, 1, 1.5), zoom=1.2)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_zticks([])

    # Remove grid
    # ax.grid(False)

    # Turn off background panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Turn off axis lines
    ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0)
    # ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0)

    plt.tight_layout()
    plt.show()


def smooth(scalars: List[float], weight: float) -> List[float]:
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_ema_smooth_landscape_with_clusters(
    x_path_valid, y_path_valid, z_path_pre, z_path_post,
    df_centroids, color_map, weight=0.9,
    label_post="Post", label_pre="Pre",
    color_post="navy", color_pre="gray", fill_color="green"
):
    # Step 1: Compute path distance (x-axis)
    path_deltas = np.sqrt(np.diff(x_path_valid)**2 + np.diff(y_path_valid)**2)
    path_distance = np.concatenate([[0], np.cumsum(path_deltas)])

    # Step 2: EMA smoothing
    z_pre_smooth = smooth(z_path_pre, weight)
    z_post_smooth = smooth(z_path_post, weight)

    # Step 3: Compute cluster tick positions
    tick_positions = []
    tick_labels = []
    tick_colors = []

    for _, row in df_centroids.iterrows():
        cluster_name = row['cluster']
        cx, cy = row['x'], row['y']
        dists = np.sqrt((x_path_valid - cx)**2 + (y_path_valid - cy)**2)
        closest_idx = np.argmin(dists)
        tick_positions.append(path_distance[closest_idx])
        tick_labels.append(str(cluster_name))
        tick_colors.append(color_map[cluster_name])

    # Step 4: Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(path_distance, 0, z_pre_smooth, color=color_pre, alpha=0.4)
    ax.fill_between(path_distance, z_pre_smooth, z_post_smooth,
                    where=np.array(z_post_smooth) > np.array(z_pre_smooth),
                    color=fill_color, alpha=0.5)

    ax.plot(path_distance, z_post_smooth, color=color_post, linewidth=2.5, label=label_post)
    ax.plot(path_distance, z_pre_smooth, color=color_pre, linewidth=2.5, label=label_pre)

    ax.set_xlabel('Summary Path (Cluster Order)', fontsize=12)
    ax.set_ylabel('log₂(HI titer / 10)', fontsize=12)
    ax.set_title('Smoothed Antibody Landscape with Clusters', fontsize=14)

    # Add colored x-axis cluster labels
    ax.set_xticks(tick_positions)
    tick_texts = ax.set_xticklabels(tick_labels, rotation=45)
    for label, color in zip(tick_texts, tick_colors):
        label.set_color(color)
        label.set_fontweight('bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.legend()
    plt.tight_layout()
    plt.show()