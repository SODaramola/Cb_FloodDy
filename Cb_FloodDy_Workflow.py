#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install -U Cb-FloodDy')


# ## Voronoi Workflow Parameters

# In[3]:


import os
from pyproj import CRS
import matplotlib.pyplot as plt
from Cb_FloodDy import voronoi_clusters

plt.rcParams["font.family"] = "Times New Roman"

# Define source CRS (WGS84)
src_crs = CRS.from_epsg(4326)

# Custom ticks for longitude and latitude
longitudes = [-95.5, -95.0, -94.5]
latitudes = [29.0, 29.4, 29.8]

# Range of station files to include (inclusive)
start_idx, end_idx = 1, 21  # e.g., station_1.csv through station_21.csv

# Directory containing station CSV files
station_dir = 'training_water_level'

# Load the floodmap shapefile
shapefile_path = 'GBay_cells_polygon.shp'

# Whether to combine polygons and which to combine (1-based indices)
# combine_pairs = [(1, 19), (12, 21), (3, 18)]  # set [] or None to skip combining

# Whether to reorder polygons so index matches station number
reorder_by_station = True

# Output paths (change as needed)
out_shapefile = 'voronoi_clusters.shp'
out_fig = os.path.join('figures', 'voronoi_map.png')

summary = voronoi_clusters.run_workflow(
    src_crs=src_crs,
    station_dir=station_dir,
    station_range=(start_idx, end_idx),
    shapefile_path=shapefile_path,
    x_ticks=longitudes,
    y_ticks=latitudes,
    out_shapefile=out_shapefile,
    out_fig=out_fig,
    reorder_by_station=True,
#     lon_name="Longitude_dd",   # optional if auto-detect fails
#     lat_name="Latitude_dd",    # optional if auto-detect fails
)

# summary  # show a summary dict


# In[ ]:





# ## Bayesian Optimization Workflow

# In[ ]:


from Cb_FloodDy import bayesian_opt_tuning as bo
import os

# === Data directories ===
train_atm_pressure_dir = os.path.join(os.getcwd(), 'atm_pressure')
train_wind_speed_dir   = os.path.join(os.getcwd(), 'wind_speed')
train_precipitation_dir= os.path.join(os.getcwd(), 'precipitation')
train_water_depth_dir  = os.path.join(os.getcwd(), 'water_depth')
train_river_discharge_dir = os.path.join(os.getcwd(), 'river_discharge')
water_level_dir = os.path.join(os.getcwd(), 'training_water_level')
polygon_clusters_path  = os.path.join(os.getcwd(), 'voronoi_clusters.shp')

# === DEMs (explicit lengths per DEM; must sum to total T) ===
dem_files = [
    os.path.join(os.getcwd(), 'DEM/dem_1.tif'),
    os.path.join(os.getcwd(), 'DEM/dem_2.tif'),
#     os.path.join(os.getcwd(), 'DEM/dem_3.tif'),
#     os.path.join(os.getcwd(), 'DEM/dem_4.tif'),
]
dem_timesteps = [217, 410]  # first 217 use DEM1, next 100 use DEM2, next 300 DEM3, next 151 DEM4

# === Temporal ===
sequence_length = 6

# === Hyperparameter spaces ===
convlstm_filters = [16, 32, 48, 64]
lstm_units       = [32, 48, 64]
dense_units      = [32, 48, 64]
l2_reg_range     = (1e-6, 1e-3)
lr_range         = (1e-5, 1e-3)
dropout_range    = (0.2, 0.5)

# === BO / training ===
n_trials   = 2
epochs     = 10
batch_size = 2
val_split  = 0.2
checkpoint_dir_BO = "checkpoint_BO"
study_name = "Flood_Depth_Prediction_BO"

# === Seed & EarlyStopping ===
seed_value = 42
es_monitor = "val_loss"
early_stopping = 5
es_restore_best = True

summary = bo.run_optimization(
    train_atm_pressure_dir=train_atm_pressure_dir,
    train_wind_speed_dir=train_wind_speed_dir,
    train_precipitation_dir=train_precipitation_dir,
    train_water_depth_dir=train_water_depth_dir,
    train_river_discharge_dir=train_river_discharge_dir,
    water_level_dir=water_level_dir,
    polygon_clusters_path=polygon_clusters_path,
    sequence_length=sequence_length,
    n_trials=n_trials,
    study_name=study_name,
    checkpoint_dir_BO=checkpoint_dir_BO,
    seed_value=seed_value,
    convlstm_filters=convlstm_filters,
    lstm_units=lstm_units,
    dense_units=dense_units,
    l2_reg_range=l2_reg_range,
    lr_range=lr_range,
    dropout_range=dropout_range,
    es_monitor=es_monitor,
    early_stopping=early_stopping,
    es_restore_best=es_restore_best,
    epochs=epochs,
    batch_size=batch_size,
    val_split=val_split,
    # Dynamic multi-DEM
    dem_files=dem_files,
    dem_timesteps=dem_timesteps,
    visualize=True
)

summary


# In[ ]:





# ## Flood Depth Prediction

# In[3]:


from Cb_FloodDy import model_prediction as mp
import os

# === TEST/EVAL SETTINGS ===
checkpoint_dir_BO     = "checkpoint_BO"  # folder with trial_*/ and artifacts/
polygon_clusters_path = os.path.join(os.getcwd(), 'voronoi_clusters.shp')  # fallback if cluster_masks.npy missing
sequence_length       = 6
seed_value            = 42

# this MUST match your actual folder name on disk
test_name = "testingharvey"

# everything lives under this directory:
base_test_dir = os.path.join(os.getcwd(), test_name)

mp.run_predictions(
    # Input rasters (TIFFs), all inside testingharvey/
    test_atm_pressure_dir    = os.path.join(base_test_dir, 'atm_pressure'),
    test_wind_speed_dir      = os.path.join(base_test_dir, 'wind_speed'),
    test_precipitation_dir   = os.path.join(base_test_dir, 'precipitation'),
    test_river_discharge_dir = os.path.join(base_test_dir, 'river_discharge'),
    test_water_depth_dir     = os.path.join(base_test_dir, 'water_depth'),

    # Water level CSVs are in testingharvey/original_water_level
    test_water_level_dir     = os.path.join(base_test_dir, 'original_water_level'),

    # DEM used for testing (single raster broadcast across timesteps)
    test_dem_file            = os.path.join(base_test_dir, 'DEM', 'dem_idw.tif'),

    # BO output from training
    checkpoint_dir_BO        = checkpoint_dir_BO,

    # Cluster polygons (only used if cluster_masks.npy wasn't saved in artifacts/)
    polygon_clusters_path    = polygon_clusters_path,

    # Temporal setup
    sequence_length          = sequence_length,

    # Output layout
    output_root              = "predictions",    # parent folder where results go
    test_name                = test_name,        # this becomes predictions/testingharvey/...

    # Reproducibility
    seed_value               = seed_value,
)


# In[ ]:




