### 2. The Core Prototype Script (`hk_tree_mapper.py`)
This script shows that you understand how to use their model and combine it with your knowledge of spatial data and arboriculture. Save this as `hk_tree_mapper.py` in your repository.

```python
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from deepforest import main
from deepforest import get_data
import os

def load_municipal_inventory(csv_path):
    """
    Loads a simulated Hong Kong municipal street tree inventory.
    (e.g., Data from HyD or ArchSD open datasets).
    """
    # Simulated data representing what HK open data provides
    df = pd.read_csv(csv_path)
    
    # Convert lat/lon points to a GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    inventory_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Project to a local metric CRS (e.g., Hong Kong 1980 Grid - EPSG:2326) for accurate overlap
    inventory_gdf = inventory_gdf.to_crs(epsG=2326)
    return inventory_gdf

def run_deepforest_prediction(image_path):
    """
    Uses the Weecology DeepForest model to predict tree crowns.
    """
    print("Initializing DeepForest model...")
    model = main.deepforest()
    model.use_release() # Load the pre-trained NEON model
    
    print(f"Predicting crowns for {image_path}...")
    # Returns a pandas dataframe with bounding box coordinates (xmin, ymin, xmax, ymax)
    predictions = model.predict_image(path=image_path, return_plot=False)
    return predictions

def create_crown_polygons(predictions_df):
    """
    Converts DeepForest bounding box predictions into spatial polygons.
    """
    geometries = []
    for idx, row in predictions_df.iterrows():
        # Create a spatial box from the predicted coordinates
        geom = box(row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        geometries.append(geom)
        
    # Assume image is georeferenced to HK 1980 Grid (EPSG:2326)
    crowns_gdf = gpd.GeoDataFrame(predictions_df, geometry=geometries, crs="EPSG:2326")
    return crowns_gdf

def map_inventory_to_crowns(crowns_gdf, inventory_gdf):
    """
    Performs a spatial join to link ground-truth municipal tree data 
    (species, risk level) to the DeepForest predicted crown polygons.
    """
    print("Performing spatial join between predicted crowns and municipal inventory...")
    # Join the point data (inventory) into the polygon data (predicted crowns)
    mapped_trees = gpd.sjoin(crowns_gdf, inventory_gdf, how="inner", predicate="contains")
    return mapped_trees

if __name__ == "__main__":
    # 1. Provide paths to your simulated/downloaded Hong Kong data
    # Note: Replace these with sample files in your actual github repo
    IMAGE_PATH = get_data("OSBS_029.png") # Using sample image for prototype
    INVENTORY_CSV = "hk_street_trees_sample.csv" 
    
    # 2. Run the pipeline (Requires dummy CSV to execute fully)
    if os.path.exists(INVENTORY_CSV):
        inventory = load_municipal_inventory(INVENTORY_CSV)
        raw_predictions = run_deepforest_prediction(IMAGE_PATH)
        crown_polygons = create_crown_polygons(raw_predictions)
        
        # 3. Output the fused Digital Twin dataset
        final_mapped_canopy = map_inventory_to_crowns(crown_polygons, inventory)
        print(f"Successfully mapped {len(final_mapped_canopy)} trees.")
        print(final_mapped_canopy[['score', 'Species', 'Risk_Level', 'geometry']].head())
    else:
        print("Please provide a sample HK inventory CSV to complete the spatial join.")
