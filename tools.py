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
from scipy.optimize import minimize
import os

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
        return 5
    if t==">=1280":
        return 1280

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



def plot_antigenic_map(df_antigenic_map,color_map,unique_clusters):
    # # Create a color palette with one unique color per cluster
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


    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Antigenic Coordinate 2')
    plt.ylabel('Antigenic Coordinate 1')
    plt.title('Antigenic Map of Viruses (Colored by Cluster)')
    plt.tight_layout()
    plt.show()



def find_summary_path(df_landscape,SMOOTH_FACTOR):

    # Get unique clusters
    unique_clusters = df_landscape['cluster'].unique()

    # Compute centroids and sort from left to right (based on x)
    df_centroids = (
        df_landscape
        .groupby('cluster')[['AG coordinate 1', 'AG coordinate 2']]
        .mean()
        .reset_index()
        .rename(columns={'AG coordinate 2': 'x', 'AG coordinate 1': 'y'})
        .sort_values(by='x')
    )

    # Fit a smooth spline through centroids
    points = df_centroids[['x', 'y']].values
    tck, u = splprep([points[:, 0], points[:, 1]], s=SMOOTH_FACTOR)
    unew = np.linspace(0, 1, 200)
    x_smooth, y_smooth = splev(unew, tck)


    # Extend last segment forward (same direction)
    # Get direction of final segment
    dx = x_smooth[-1] - x_smooth[-2]
    dy = y_smooth[-1] - y_smooth[-2]

    # Extend by N steps forward
    N_extend = 3
    step_size = 1.0  # you can adjust to control how far it extends

    x_extension = [x_smooth[-1] + i * dx * step_size for i in range(1, N_extend + 1)]
    y_extension = [y_smooth[-1] + i * dy * step_size for i in range(1, N_extend + 1)]

    # Combine with original path
    x_smooth = np.concatenate([x_smooth, x_extension])
    y_smooth = np.concatenate([y_smooth, y_extension])

    summary_path = np.column_stack((x_smooth, y_smooth))



    #plot summary_path
    plt.figure(figsize=(8, 6))

    # Plot clusters
    for cluster in unique_clusters:
        cluster_df = df_landscape[df_landscape['cluster'] == cluster]
        plt.scatter(cluster_df['AG coordinate 2'], cluster_df['AG coordinate 1'],
                    s=60, color=color_map[cluster], label=cluster)

        x_mean = cluster_df['AG coordinate 2'].mean()
        y_mean = cluster_df['AG coordinate 1'].mean() - 0.05
        plt.text(x_mean, y_mean, str(cluster), fontsize=11, weight='bold',
                color=color_map[cluster], ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.6))

    # Plot summary path
    plt.plot(x_smooth, y_smooth, color='black', linewidth=2, label='Summary Path')


    plt.title('Antigenic Map with Summary Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    return x_smooth,y_smooth,df_centroids


def compute_surface_regression(X,Y,Z,GRID_SIZE,A,CLOSE_NEIGH,cluster_label,unique_clusters,OPTIMIZATION=True):
    # Step 1: Create grid over antigenic map
    x_range = np.linspace(np.min(X) , np.max(X) , GRID_SIZE)
    y_range = np.linspace(np.min(Y) , np.max(Y) , GRID_SIZE)
    xj, yj = np.meshgrid(x_range, y_range)


    # Step 2: Compute distances and weights (tricubic)
    aij = np.sqrt((X[:, None, None] - xj) ** 2 + (Y[:, None, None] - yj) ** 2)
    weights = np.where(aij <= A, (1 - (aij / A) ** 3) ** 3, 0)

    #closest entigen in the map
    min_dist_to_grid = np.min(aij, axis=0)
    landscape_z = np.zeros_like(xj)

    #Step 3 : looop over each point on the surface and predict the z vlaue by weighted linear regression
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):

            # Condition1 : If the closest antigenic point is too far, skip
            if min_dist_to_grid[i, j] > CLOSE_NEIGH:
                landscape_z[i, j] = np.nan
                continue
            
            # Condition 2: if the sum of the weights are 0 - it means very far point
            w = weights[:, i, j]
            if np.sum(w) == 0:
                landscape_z[i, j] = np.nan
                print("null becasue sum weight is 0")
                continue

            # compute  predicted surface
            if OPTIMIZATION:

                z_val = fit_weighted_surface_at_point(X, Y, Z, w, xj[i, j], yj[i, j])
                landscape_z[i, j] = z_val

            else:
                X_design = sm.add_constant(np.column_stack((X, Y)))
                try:
                    model = sm.WLS(Z, X_design, weights=w,)
                    results = model.fit()
                    beta = results.params
                    landscape_z[i, j] = beta[0] + beta[1] * xj[i, j] + beta[2] * yj[i, j]
                except:
                    landscape_z[i, j] = np.nan
                    print("null becasue couldnt reach linera regression")

    # Step 4: Find corresponding surface height at each (X, Y)
    # Locate closest grid point index
    x_idx = np.searchsorted(x_range, X) - 1
    y_idx = np.searchsorted(y_range, Y) - 1

    # Clip indices to stay within grid bounds
    x_idx = np.clip(x_idx, 0, GRID_SIZE - 1)
    y_idx = np.clip(y_idx, 0, GRID_SIZE - 1)


    # Get surface height at that grid point
    predicted_z = landscape_z[y_idx, x_idx]
    rmse = np.sqrt(mean_squared_error(Z,predicted_z))
    print(f"the RMSE is :{rmse}")

    plot_error_scatter(Z,predicted_z,cluster_label,unique_clusters)
    return x_range,y_range,landscape_z,predicted_z




def fit_weighted_surface_at_point(X, Y, Z, w, xj, yj):
    """
    Fits a local linear regression model at a given grid point (xj, yj)
    using weighted least squares. Censored titers are handled via L-BFGS-B optimization.

    Parameters:
    -----------
    X, Y : np.ndarray
        Coordinates of the antigenic points.
    Z : np.ndarray
        Log-transformed titer values with censored values marked.
    w : np.ndarray
        Weights for each point based on tricubic kernel.
    xj, yj : float
        Coordinates of the grid point being fitted.

    Returns:
    --------
    float
        The predicted landscape height (z-value) at grid point (xj, yj).
    """
     # Step 1: Build design matrix for linear regression

    X_design = sm.add_constant(np.column_stack((X, Y)))

    #assign undtectable and high titer values
    detectable_mask = (Z > -1) & (Z < 7)
    undetectable_mask = Z <= -1
    high_mask = Z >= 7

    detectable_idx = np.where(detectable_mask)[0]
    undetectable_idx = np.where(undetectable_mask)[0]
    high_idx = np.where(high_mask)[0]

    # Step 3: Define custom RMSE loss function that handles censoring behavior

    def rmse_loss_local(censored_vals):
        Z_full = Z.copy()
        Z_full[undetectable_idx] = censored_vals[:len(undetectable_idx)]
        Z_full[high_idx] = censored_vals[len(undetectable_idx):]

        try:
            model = sm.WLS(Z_full, X_design, weights=w)
            results = model.fit()
            beta = results.params
            z_pred = beta[0] + beta[1] * xj + beta[2] * yj
        except:
            return np.inf

        errors = []
        for idx in range(len(Z)):
            if idx in undetectable_idx:
                if z_pred <= 0:
                    continue #no penalty
                else:
                    errors.append((z_pred - (-1)) ** 2)
            elif idx in high_idx:
                if z_pred >= 7:
                    continue #no penalty
                else:
                    errors.append((z_pred - 7) ** 2)
            else:
                errors.append((z_pred - Z[idx]) ** 2)

        return np.sqrt(np.mean(errors)) if errors else 0
    
    # Step 4: Optimize censored values using L-BFGS-B to minimize RMSE

    if len(undetectable_idx) + len(high_idx) > 0:
        x0 = np.concatenate([
            np.full(len(undetectable_idx), -1),
            np.full(len(high_idx), 7)
        ])
        bounds = [(-10, -1)] * len(undetectable_idx) + [(7, 10)] * len(high_idx)

        res = minimize(rmse_loss_local, x0=x0, bounds=bounds, method='L-BFGS-B')

        Z_full = Z.copy()
        if res.success:
            Z_full[undetectable_idx] = res.x[:len(undetectable_idx)]
            Z_full[high_idx] = res.x[len(undetectable_idx):]
        else:
            return np.nan
    else:
        Z_full = Z

    try:
        model = sm.WLS(Z_full, X_design, weights=w)
        results = model.fit()
        beta = results.params
        return beta[0] + beta[1] * xj + beta[2] * yj
    except:
        return np.nan


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


def plot_3d_antibody(X,Y,Z,C,save_path=None,SAMPLE2=None):

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
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_ylim(np.min(Y), np.max(X))
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
    if save_path:
        #check folder exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the figure
        if SAMPLE2:
            save_path=os.path.join(save_path, "3d_antigen_response_sample2.png")
        else:
            save_path=os.path.join(save_path, "3d_antigen_response_sample1.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")



def plot3d_antibody_surface(X,Y,Z,C,GRID_SIZE,landscape_z,x_range,y_range,x_smooth,y_smooth,unique_clusters,cluster_label):

    ### Step 1: Calculate Z values for the surface and the summary path

    # -- Surface Z values (masked where Z <= -1)
    masked_z = np.where(landscape_z <= -1, np.nan, landscape_z)

    # Create facecolors from the colormap, handling NaNs with transparency
    facecolors = plt.cm.Blues(
        (masked_z - np.nanmin(masked_z)) / (np.nanmax(masked_z) - np.nanmin(masked_z))
    )
    facecolors[np.isnan(masked_z)] = (1, 1, 1, 0)  # Transparent for invalid values

    # -- Summary path Z values (interpolated from grid)
    x_idx_path = np.searchsorted(x_range, x_smooth) - 1
    y_idx_path = np.searchsorted(y_range, y_smooth) - 1

    # Check if each path point is inside the grid
    inside_x = (x_smooth >= x_range[0]) & (x_smooth <= x_range[-1])
    inside_y = (y_smooth >= y_range[0]) & (y_smooth <= y_range[-1])
    inside_mask = inside_x & inside_y

    # Clip to valid index bounds
    x_idx_path = np.clip(x_idx_path, 0, landscape_z.shape[1] - 1)
    y_idx_path = np.clip(y_idx_path, 0, landscape_z.shape[0] - 1)

    # Create z_path and assign only inside values
    # Create z_path and assign default -1 for out-of-bounds points
    z_path = np.full_like(x_smooth, -1, dtype=float)
    z_path[inside_mask] = landscape_z[y_idx_path[inside_mask], x_idx_path[inside_mask]]

    # Filter out invalid path points (those still equal to -1)
    valid_mask = z_path > -1

    # Filter out invalid path points
    valid_mask = ~np.isnan(z_path)
    x_path_valid = x_smooth[valid_mask]
    y_path_valid = y_smooth[valid_mask]
    z_path_valid = z_path[valid_mask]

    ### Step 2: Calculate surface Z values at data points and transparency

    # Grid indices of (X, Y) sample locations
    x_idx = np.searchsorted(x_range, X) - 1
    y_idx = np.searchsorted(y_range, Y) - 1
    
    
    x_idx = np.clip(x_idx, 0, GRID_SIZE - 1)
    y_idx = np.clip(y_idx, 0, GRID_SIZE - 1)

    # Get Z values from surface at those indices
    Z_surf_at_data = landscape_z[y_idx, x_idx]  # Note: row first (Y), column second (X)

    # Compute alpha values: fade if below surface
    alpha_vals = np.where(Z < Z_surf_at_data, 0.3, 1)
    
    xj, yj = np.meshgrid(x_range, y_range)


    # --- Matplotlib 3D surface plot ---
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(xj, yj, masked_z, color='lightgray', rstride=1, cstride=1,
                    linewidth=0.1, antialiased=True, edgecolor='k', alpha=0.6, zorder=1)

    for x, y, z in zip(X, Y, Z):
        ax.plot([x, x], [y, y], [-1, z], color='gray', linewidth=1, alpha=1, zorder=2)

    for xi, yi, zi, ci, ai in zip(X, Y, Z, C, alpha_vals):
        ax.scatter(xi, yi, zi, color=ci, edgecolors='black', linewidths=0.6, alpha=ai,
                   s=60, depthshade=False, zorder=3)

    ax.plot(x_path_valid, y_path_valid, z_path_valid, color='black', linewidth=3,
            label='Summary Path', zorder=4)

    ax.set_zlabel('log₂(HI titer / 10)', labelpad=15, fontsize=12)
    ax.set_zlim(-1, Z.max())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.view_init(elev=30, azim=255)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1.01, 1, 1]))
    ax.set_box_aspect(aspect=(3, 1, 1.5), zoom=1.2)

    plt.tight_layout()
    plt.show()

    # --- Plotly Interactive 3D Surface ---
    z_path_valid_3d = np.clip(z_path_valid, -1, z_path_valid.max())
    surface = go.Surface(
        x=xj,
        y=yj,
        z=masked_z,
        colorscale='Blues',
        showscale=False,
        opacity=0.8,
        cmin=np.nanmin(masked_z),
        cmax=np.nanmax(masked_z),
        name='Surface'
    )

    summary_path = go.Scatter3d(
        x=x_path_valid,
        y=y_path_valid,
        z=z_path_valid_3d,
        mode='lines',
        line=dict(color='black', width=6),
        name='Summary Path'
    )

    cluster_traces = []
    for cluster in unique_clusters:
        cluster_idx = cluster_label == cluster
        trace = go.Scatter3d(
            x=X[cluster_idx],
            y=Y[cluster_idx],
            z=Z[cluster_idx],
            mode='markers',
            marker=dict(
                size=6,
                color=color_map[cluster],
                line=dict(color='black', width=0.5),
                opacity=0.9
            ),
            name=f'{cluster}'
        )
        cluster_traces.append(trace)

    fig = go.Figure(data=[surface, summary_path] + cluster_traces)
    fig.update_layout(
        title='Interactive Antibody Landscape',
        scene=dict(
            xaxis_title='Antigenic X',
            yaxis_title='Antigenic Y',
            zaxis_title='log₂(HI titer / 10)',
            zaxis=dict(range=[-1, Z.max()])
        ),
        width=1000,
        height=700,
        legend=dict(title='Clusters')
    )

    fig.show()


    return masked_z,x_path_valid,y_path_valid,z_path_valid

def smooth(scalars: List[float], weight: float) -> List[float]:
    # Find index of the first non -1 value
    pos = 0
    for i, val in enumerate(scalars):
        if val > -1:
            pos = i
            break
    else:
        raise ValueError("All values are -1, cannot smooth.")

    smoothed = scalars[:pos]  # keep the initial -1s unchanged
    smoothed = scalars[:pos].tolist()  # Now it's a Python list and supports .append()
    last = scalars[pos]
    for point in scalars[pos:]:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def plot_2d_landscape(x_path_valid, y_path_valid, z_path_valid, df_centroids, weight=0.9,y_lim=10,save_path=None):
    # Step 1: Compute distance along path
    path_deltas = np.sqrt(np.diff(x_path_valid)**2 + np.diff(y_path_valid)**2)
    path_distance = np.concatenate([[0], np.cumsum(path_deltas)])

    # Step 2: EMA smoothing
    z_path_valid = smooth(z_path_valid, weight)

    # Step 3: Find closest point on path for each cluster centroid
    tick_positions = []
    tick_colors = []

    for _, row in df_centroids.iterrows():
        cx, cy = row['AG coordinate 2'], row['AG coordinate 1']
        dists = np.sqrt((x_path_valid - cx)**2 + (y_path_valid - cy)**2)
        closest_idx = np.argmin(dists)

        tick_positions.append(path_distance[closest_idx])
        tick_colors.append(row['color'])

    # Step 4: Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(path_distance, z_path_valid, color='black', linewidth=2)
    ax.fill_between(path_distance, -1, z_path_valid, color="gray", alpha=0.4)

    ax.set_ylabel('log₂(HI titer / 10)', fontsize=12)
    ax.set_ylim(0, y_lim)
    ax.set_xlim(30.89,path_distance[-1])
    ax.grid(False)

    # Step 5: Plot dots absolutely below the x-axis using axes coordinates
    for xdata, color in zip(tick_positions, tick_colors):
        # Convert data x to axes fraction
        x_axes = ax.transData.transform((xdata, 0))[0]
        x_axes = ax.transAxes.inverted().transform((x_axes, 0))[0]
        ax.scatter(x_axes, -0.05, color=color, s=60,alpha=0.6, transform=ax.transAxes, clip_on=False)
    plt.xticks([])

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.5)
    plt.show()
    if save_path:
        #check folder exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the figure
        save_path=os.path.join(save_path, "2d_landscape.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")

def plot_ema_smooth_landscape_with_clusters(
    x_path_valid, y_path_valid, z_path_pre, z_path_post,
    df_landscape, color_map, weight=0.9,
    label_post="Post", label_pre="Pre",
    color_post="navy", color_pre="gray", fill_color="green",y_lim=10,save_path=None
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

    for _, row in df_landscape.iterrows():
        cx, cy = row['AG coordinate 2'], row['AG coordinate 1']
        dists = np.sqrt((x_path_valid - cx)**2 + (y_path_valid - cy)**2)
        closest_idx = np.argmin(dists)

        tick_positions.append(path_distance[closest_idx])
        tick_colors.append(row['color'])

    # Step 4: Plot
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.fill_between(path_distance, -1, z_pre_smooth, color=color_pre, alpha=0.4)
    ax.fill_between(path_distance, z_pre_smooth, z_post_smooth,
                    where=np.array(z_post_smooth) > np.array(z_pre_smooth),
                    color=fill_color, alpha=0.5)

    ax.plot(path_distance, z_post_smooth, color=color_post, linewidth=2.5, label=label_post)
    ax.plot(path_distance, z_pre_smooth, color=color_pre, linewidth=2.5, label=label_pre)

    ax.set_ylabel('log₂(HI titer / 10)', fontsize=12)

    ax.set_ylim(0, y_lim)
    ax.set_xlim(30.89,path_distance[-1])
    # Optional: keep tick labels or remove
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=10)

    # Step 5: Add colored dots below the x-axis
    for xdata, color in zip(tick_positions, tick_colors):
        x_axes = ax.transData.transform((xdata, 0))[0]
        x_axes = ax.transAxes.inverted().transform((x_axes, 0))[0]
        ax.scatter(x_axes, -0.05, color=color, s=60, transform=ax.transAxes, clip_on=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.legend()

    # Adjust layout to avoid cropping dots
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.5)
    plt.show()
    if save_path:
        #check folder exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the figure
        save_path=os.path.join(save_path, "2d_landscape_comparison.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
