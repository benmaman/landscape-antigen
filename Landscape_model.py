import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tools import *
from matplotlib.lines import Line2D



# ==== USER PARAMETERS ====

SAMPLE = (11101, 'D-28 predose')     # sample number of the data
SAMPLE_2=     (11101, 'D1 predose') #optional : when we want to compare 2 measurment of 2 patients or patient before and after
GRID_SIZE = 200        # Grid resolution for plotting or interpolation
A = 11              # Distance threshold to define a "distant" antigen
CLOSE_NEIGH = 5     # Radius threshold for regression surface - points which far form this point will not be clacualted
NORMALIZE = True       # Whether to normalize the X and Y coordinates
SMOOTH_FACTOR =50    # Spline smoothing factor for the summary path
OPTIMIZATION=False
DATA_DIR='patients_bt_years.csv'
FLUGEN=True




##1.import data
df_antigenic_map=pd.read_csv("antigentic_map.csv")
df_antigenic_map_by_gal=pd.read_csv("antigen_map/antigen_coordinates.csv")
df=pd.read_csv(DATA_DIR)
df=df.iloc[:,2:]
#fluegen
if FLUGEN:
    #data preprocessing
    df=pd.read_excel("flugen_lilach/flugen_002_sera_IgA_for_Gal.xlsx",index_col='ptid')
    df=df.groupby(by=['ptid', 'timepoint']).mean(numeric_only=True)

    hertz_titer=pd.read_csv("fluegen/list_of_identical_seqs.csv")
    rename_dict = dict(zip(hertz_titer['name'], hertz_titer['similiar_seq_name']))
    df=df[list(rename_dict.keys())].rename(columns=rename_dict)



## 2. data preprocessing
#preprocess- 
df=df.applymap(clean_titer)
df=df.applymap(convert_titer)
unique_clusters = df_antigenic_map['cluster'].unique()

    

#patient1 preprocess
sample1=df.transpose()[SAMPLE]
sample1=sample1.rename("log_titers")
sample1 = sample1.dropna()  # drop missing titers just in case


# Merge coordinates and titers for a single ferret sample
# Merge antigenic coordinates and titer data
df_antigenic_map["color"]=df_antigenic_map['cluster'].map(color_map)
df_landscape = df_antigenic_map.merge(sample1, left_on='viruses', right_index=True)

# scaler = MinMaxScaler()
# df_landscape['AG coordinate 1']=scaler.fit_transform(df_landscape['AG coordinate 1'].values.reshape(-1, 1)).flatten()
# df_landscape['AG coordinate 2']=scaler.fit_transform(df_landscape['AG coordinate 2'].values.reshape(-1, 1)).flatten()

# Extract coordinates and colors
X = df_landscape['AG coordinate 2'].values
Y = df_landscape['AG coordinate 1'].values
Z = df_landscape['log_titers'].values
C = df_landscape['color']
cluster_label1=df_landscape["cluster"].values

if SAMPLE_2:
    #patient2 preprocess
    sample2=df.transpose()[SAMPLE_2]
    sample2=sample2.rename("log_titers")
    sample2 = sample2.dropna()  # drop missing titers just in case

    #fliter only antigen which has shown in sample1
    sample2 = sample2[sample2.index.intersection(sample1.index)]
    
    # Merge coordinates and titers for a single ferret sample
    # Merge antigenic coordinates and titer data
    df_landscape_2 = df_antigenic_map.merge(sample2, left_on='viruses', right_index=True)
    df_landscape_2['color'] = df_landscape_2['cluster'].map(color_map)

    # scaler = MinMaxScaler()
    # df_landscape_2['AG coordinate 1']=scaler.fit_transform(df_landscape_2['AG coordinate 1'].values.reshape(-1, 1)).flatten()
    # df_landscape_2['AG coordinate 2']=scaler.fit_transform(df_landscape_2['AG coordinate 2'].values.reshape(-1, 1)).flatten()

    # Extract coordinates and colors
    X_2 = df_landscape_2['AG coordinate 2'].values
    Y_2 = df_landscape_2['AG coordinate 1'].values
    Z_2 = df_landscape_2['log_titers'].values
    C_2 = df_landscape_2['color']
    cluster_label2=df_landscape_2["cluster"].values


x_smooth,y_smooth,df_centroids=find_summary_path(df_antigenic_map,SMOOTH_FACTOR)
x_smooth2,y_smooth2,df_centroids2=find_summary_path(df_antigenic_map,SMOOTH_FACTOR)


#sample1
x_range,y_range,landscape_z,predicted_z =compute_surface_regression(X,Y,Z,GRID_SIZE,A,CLOSE_NEIGH,cluster_label1,unique_clusters,OPTIMIZATION)
masked_z,x_path_valid,y_path_valid,z_path_valid=plot3d_antibody_surface(X,Y,Z,C,GRID_SIZE,landscape_z,x_range,y_range,x_smooth,y_smooth,unique_clusters,cluster_label1)


#sample2
x_range,y_range,landscape_z2,predicted_z2 =compute_surface_regression(X_2,Y_2,Z_2,GRID_SIZE,A,CLOSE_NEIGH,cluster_label2,unique_clusters,OPTIMIZATION)
masked_z2,x2_path_valid,y2_path_valid,z2_path_valid=plot3d_antibody_surface(X_2,Y_2,Z_2,C,GRID_SIZE,landscape_z2,x_range,y_range,x_smooth2,y_smooth2,unique_clusters,cluster_label2)


#create antibody landscape
plot_2d_landscape(x_path_valid,y_path_valid,z_path_valid,df_landscape,weight=0.9,y_lim=10)
plot_2d_landscape(x2_path_valid,y2_path_valid,z2_path_valid,df_landscape_2,weight=0.9,y_lim=10)
plot_ema_smooth_landscape_with_clusters(x_path_valid, y_path_valid, z_path_valid, z2_path_valid, df_landscape,color_map=color_map ,label_pre="Pre", label_post="Post",y_lim=10)
