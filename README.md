# Cb_FloodDy

_A cluster-based temporal-attention framework  with utilities for Voronoi cluster generation and Optuna-driven hyperparameter tuning._

---

## Highlights

- **End-to-end training pipeline** built on ConvLSTM + CBAM (channel & spatial attention) with a custom **temporal attention** layer and **cluster-aware spatial modulation**.
- **Voronoi clustering toolkit** to partition a floodplain into station-informed regions, save shapefiles, and produce publication-ready plots.
- **Lazy module loading** at package import time to keep interactive workflows snappy (heavy modules are only loaded when needed).
- Packaged for PyPI; standard build metadata included.

---

## Installation

```bash
# (Optional) create a clean env
conda create -n cb_flooddy -y
conda activate cb_flooddy

# install from PyPI
pip install Cb-FloodDy
```

---

## Model Workflow

### 1) Voronoi clusters (create station-informed polygons)

```python
from pyproj import CRS
from Cb_FloodDy.voronoi_clusters import run_workflow

clusters = run_workflow(
    src_crs=CRS.from_epsg(4326),                  # lon/lat
    station_dir="path/to/water_level_stations",   # files like station_1.csv, station_2.csv, ...
    station_range=(1, 21),                        # i.e., 21 stations available, should be set to the available number of stations
    shapefile_path="GBay_cells_polygon.shp",      # domain/flood extent polygon(s)
    combine_pairs=[(1, 19), (12, 21), (3, 18)],   # optional unions
    x_ticks=[-95.5, -95.0, -94.5],                # optional map ticks
    y_ticks=[29.0, 29.4, 29.8],
    out_shapefile="voronoi_clusters.shp",         # optional outputs
    out_fig="voronoi_map.png",
    reorder_by_station=True,                      # ensure polygon i matches station i
)
```

- Under the hood: station CSVs are parsed (with robust lon/lat detection), a bounded Voronoi tessellation is built and clipped to your floodplain, optional polygons get unioned, and outputs can be saved/visualized.

### 2) Train the flood-depth model with Optuna

```python
from Cb_FloodDy import bayesian_opt_tuning as bo

summary = bo.run_optimization(
    train_atm_pressure_dir="data/atm_pressure_tifs/",
    train_wind_speed_dir="data/wind_speed_tifs/",
    train_precipitation_dir="data/precip_tifs/",
    train_water_depth_dir="data/water_depth_tifs/",  # y
    train_river_discharge_dir="data/river_discharge_tifs/",
    water_level_dir="data/water_levels_csvs/",
    polygon_clusters_path="voronoi_clusters.shp",                # from step 1
    sequence_length=6,
    n_trials=30,
    study_name="cb_flooddy_study",
    checkpoint_dir_BO="checkpoints/optuna",
    seed_value=3,
    convlstm_filters=[16, 32, 48],                               # search grids/ranges
    lstm_units=[32, 48],
    dense_units=[64, 128],
    l2_reg_range=(1e-7, 1e-4),
    lr_range=(1e-4, 5e-3),
    dropout_range=(0.1, 0.5),
    es_monitor="val_loss",
    early_stopping=10,
    es_restore_best=True,
    epochs=100,
    batch_size=2,
    val_split=0.2,
    dem_files=["data/dem_t0.tif","data/dem_t1.tif"],              # tiled across time
    dem_timesteps=[120, 240],
    visualize=True
)
print(summary)
```

- The pipeline stacks multi-source rasters (atm pressure, wind, precip, discharge, DEM) into sequences, normalizes with NaN-aware masks, aligns water-level histories per station, ensures **#clusters == #stations**, and launches Optuna trials.
- The model: 3×ConvLSTM → CBAM blocks (masked channel+spatial attention), shared LSTMs on water-level sequences → custom temporal attention → **ClusterBasedApplication** to project station context back into the spatial domain → modulation + dense head to predict flood depth rasters.
- Artifacts written per trial (e.g., `best_model.h5`, `best_val_loss.txt`, `viz/` with prediction vs. truth and **spatial attention** maps; study-level `study_summary.csv`). Temporal attention weights for the best epoch are also exported.

---

Outputs saved after training:
```
checkpoint_BO/
├─ trial_000/
│  ├─ best_model.keras
│  └─ ...
├─ trial_001/
│  ├─ best_model.keras
│  └─ ...
└─ artifacts/
   ├─ normalization_params.npz
   └─ cluster_masks.npy            # if created during training
```
---

### 3) Generate flood‑depth predictions on a test event

After training finishes, you’ll have a directory like:
- `checkpoint_BO/`
  - `trial_000/best_model.keras`
  - `trial_001/best_model.keras`
  - ...
  - `artifacts/normalization_params.npz`
  - `artifacts/cluster_masks.npy`

Those are all you need (plus your test rasters and station CSVs) to run inference.

```python
from Cb_FloodDy import model_prediction as mp
import os

# === paths from training ===
checkpoint_dir_BO     = "checkpoint_BO"             # same folder you passed into run_optimization(...)
polygon_clusters_path = os.path.join(os.getcwd(),
                                     "voronoi_clusters.shp")  # fallback if cluster_masks.npy is missing
sequence_length       = 6                            # MUST match what you trained with
seed_value            = 42                           # for reproducibility

# name of the test case (e.g., a specific storm)
test_name = "testingharvey"

# everything for this storm/test lives under this folder:
#   testingharvey/
#       atm_pressure/            *.tif (1 per timestep)
#       wind_speed/              *.tif
#       precipitation/           *.tif
#       river_discharge/         *.tif
#       water_depth/             *.tif  (this is the “truth” flood depth; used for metrics)
#       original_water_level/    *.csv  (one CSV per gauge/station)
#       DEM/dem_idw.tif          DEM raster (broadcast across time)
base_test_dir = os.path.join(os.getcwd(), test_name)

mp.run_predictions(
    # Required raster time series
    test_atm_pressure_dir    = os.path.join(base_test_dir, "atm_pressure"),
    test_wind_speed_dir      = os.path.join(base_test_dir, "wind_speed"),
    test_precipitation_dir   = os.path.join(base_test_dir, "precipitation"),
    test_river_discharge_dir = os.path.join(base_test_dir, "river_discharge"),
    test_water_depth_dir     = os.path.join(base_test_dir, "water_depth"),

    # Required water-level histories (CSV per station)
    test_water_level_dir     = os.path.join(base_test_dir, "original_water_level"),

    # DEM used for testing (single raster, broadcast internally)
    test_dem_file            = os.path.join(base_test_dir, "DEM", "dem_idw.tif"),

    # Training artifacts
    checkpoint_dir_BO        = checkpoint_dir_BO,
    polygon_clusters_path    = polygon_clusters_path,   # only used if cluster_masks.npy is missing

    # Temporal setup (must match training)
    sequence_length          = sequence_length,

    # Output layout
    output_root              = "predictions",           # parent folder for all outputs
    test_name                = test_name,               # subfolder under output_root

    # Reproducibility
    seed_value               = seed_value,
)
```

### What `run_predictions(...)` does

- Rebuilds the exact input tensors the model saw during training:
  - Stacks atmospheric pressure, wind speed, precipitation, discharge, and DEM into `(T, H, W, C)`.
  - Cuts them into sliding windows of length `sequence_length` (e.g., 6 timesteps).
  - Aligns those windows with:
    - observed flood depth rasters (`water_depth/`) for evaluation,
    - per‑station water‑level CSVs (`original_water_level/`) for the hydrodynamic signal.
- Loads `normalization_params.npz` and `cluster_masks.npy` from `checkpoint_dir_BO/artifacts/` so test data use **the same normalization and masks** as training.
  - If `cluster_masks.npy` isn’t there, it will rasterize `voronoi_clusters.shp` as a fallback.
- Loops over every trial folder inside `checkpoint_BO` (e.g., `trial_000`, `trial_001`, …), loads the trial’s `best_model.keras`, and runs inference.

For each trial it writes:
- Per‑timestep predicted depth GeoTIFFs to  
  `predictions/<test_name>/trial_XXX/<matching_original_filename>.tif`
- A `summary_metrics.csv` in that same `trial_XXX/` folder with average MSE / RMSE / R² for that trial
  (metrics use only valid pixels; NaN pixels in the truth are ignored).
- An overall `all_trials_summary.csv` in  
  `predictions/<test_name>/` comparing every trial’s aggregate skill.


## Data Expectations

- **Raster inputs (.tif):** Each meteorological/hydrologic variable is a time-stack (one file per timestep), same shape & transform. The DEM can change by regime; provide `dem_files` + `dem_timesteps` whose counts sum to the total number of timesteps. Shape checks and tiling are handled for you.
- **Water levels (CSV):** One CSV per station (naturally sorted), with a `water_level` column; sequences are normalized per global min/max and aligned to the raster sequence length.
- **Cluster polygons (SHP):** Produced by `voronoi_clusters.run_workflow(...)`. Each pixel is assigned to **at most one** cluster; overlaps are checked and rejected.

---

## Key APIs 

### `Cb_FloodDy.voronoi_clusters`
- `load_station_points(station_dir, start_idx, end_idx, lon_name=None, lat_name=None) -> list[(lon, lat)]`
- `load_floodmap(shapefile_path) -> (gdf, boundary_union)`
- `build_voronoi(stations, boundary_union) -> list[Polygon]`
- `combine_specified_polygons(polygons, pairs) -> list[Polygon]`
- `plot_voronoi_on_floodmap(...) -> (fig, ax)`
- `save_polygons_as_shapefile(polygons, crs, out_path)`
- `run_workflow(...) -> dict`

### `Cb_FloodDy.bayesian_opt_tuning`
- Data utilities: TIFF loaders, NaN-aware normalization, mask verification/visualization, natural sort, water-level ingestion.
- Attention: **StandardCBAM** (masked), **CustomAttentionLayer** (top-k emphasis), **ClusterBasedApplication** (station-to-grid projection).
- Loss/metrics: `masked_mse`, `TrueLoss` (averaged over valid pixels).
- Model factory: `build_model_with_cbam_weighted(...)` returns a compiled Keras model.
- Training & search: `run_optimization(...)` orchestrates Optuna trials, callbacks (EarlyStopping/LR-plateau, custom checkpoint that also extracts attention), and result logging/visualization.

---

## Outputs & Artifacts

- `checkpoints/optuna/trial_###/best_model.h5` — best epoch per trial.
- `.../best_val_loss.txt` — scalar.
- `.../params_table.csv` — single-trial hyperparams; `study_summary.csv` — all trials.
- `.../viz/pred_vs_actual_val0.png`, `.../viz/spatial_attention_val0.png` — qualitative inspection.
- `.../artifacts/cluster_masks.npy`, `.../artifacts/normalization_params.npz` — reproducibility.

---

## Tips & Gotchas

- **GPU & precision:** TensorFlow GPU memory growth is enabled; global precision set to `float32` for stability.
- **Valid-pixel masking:** Loss/metrics and CBAM attention paths respect masked/invalid pixels (NaNs in inputs become zeros; a complementary mask is carried through).
- **Clusters ↔ stations:** The model asserts `num_clusters == num_stations`. Ensure your Voronoi workflow (possibly after `combine_pairs`) yields a 1:1 mapping.

---

## References

Refer to these papers for a detailed explanation:

- Daramola, S., et al. (2025). **A Cluster-based Temporal Attention Approach for Predicting Cyclone-induced Compound Flood Dynamics**. *Environmental Modelling & Software* 191, 106499. https://doi.org/10.1016/j.envsoft.2025.106499

