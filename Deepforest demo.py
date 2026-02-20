
import os
import cv2
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from deepforest import main
from PIL import Image
import warnings

# Suppress warnings for a clean terminal output
warnings.filterwarnings('ignore')

# Define base directory and change working directory to it
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ==========================================
MUNICIPAL_CSV = os.path.join(BASE_DIR, "Trees_Major_Parks.csv")
METADATA_CSV = os.path.join(BASE_DIR, "photo_metadata.csv")

def run_batch_pipeline():
    # ==========================================
    # 2. LOAD AI MODEL & MUNICIPAL DATA
    # ==========================================
    print("Loading DeepForest Model and Municipal Tree Data...")
    model = main.deepforest()
    model.use_release()

    model.config["score_thresh"] = 0.5

   # Load Municipal Data
    try:
        raw_csv = pd.read_csv(MUNICIPAL_CSV)
        csv_geom = [Point(xy) for xy in zip(raw_csv['Longitude'], raw_csv['Latitude'])]
        inventory_gdf = gpd.GeoDataFrame(raw_csv, geometry=csv_geom, crs="EPSG:4326")
        
        # Create the Spatial Mask (The "Invisible Fence" around park trees)
        park_mask = inventory_gdf.unary_union.convex_hull.buffer(0.0005)
    except Exception as e:
        print(f"❌ Error loading municipal data: {e}")
        return

    # ==========================================
    # 3. LOOP THROUGH YOUR PHOTO METADATA
    # ==========================================
    metadata_df = pd.read_csv(METADATA_CSV)
    for _, row in metadata_df.iterrows():
        img_file = row['image_name']
        # Add .jpg extension if missing
        if not os.path.splitext(img_file)[1]:
            img_file = img_file + '.jpg'
        
        # Get scale for this specific image
        with Image.open(img_file) as img:
            img_w, img_h = img.size
        
        lon_per_px = (row['tr_lon'] - row['bl_lon']) / img_w
        lat_per_px = (row['tr_lat'] - row['bl_lat']) / img_h
        tl_lat, tl_lon = row['tr_lat'], row['bl_lon']

        print(f"\n--- Analyzing: {img_file} ---")
        
        # predict_tile using 'raster_path' (required for v1.4.0)
        predictions = model.predict_tile(
            raster_path=img_file, 
            return_plot=False, 
            patch_size=400, 
            patch_overlap=0.1
        )

        if predictions is None or predictions.empty:
            continue

        # Convert pixels to GPS
        predictions['geometry'] = predictions.apply(
            lambda r: box(
                tl_lon + (r['xmin'] * lon_per_px),
                tl_lat - (r['ymax'] * lat_per_px),
                tl_lon + (r['xmax'] * lon_per_px),
                tl_lat - (r['ymin'] * lat_per_px)
            ), axis=1
        )
        crowns_gdf = gpd.GeoDataFrame(predictions, geometry='geometry', crs="EPSG:4326")

        # APPLY THE MASK: Keep only trees inside the park boundary
        masked_crowns = crowns_gdf[crowns_gdf.geometry.intersects(park_mask)]
        
        # Save Visual Output
        img_cv = cv2.imread(img_file)
        for _, m_row in masked_crowns.iterrows():
            # Draw green boxes only for masked trees
            p = m_row['geometry'].bounds
            px_min = int((p[0] - tl_lon) / lon_per_px)
            px_max = int((p[2] - tl_lon) / lon_per_px)
            py_min = int((tl_lat - p[3]) / lat_per_px)
            py_max = int((tl_lat - p[1]) / lat_per_px)
            cv2.rectangle(img_cv, (px_min, py_min), (px_max, py_max), (0, 255, 0), 2)
            
        cv2.imwrite(f"MASKED_{img_file}", img_cv)
        print(f"✅ Success: Saved MASKED_{img_file} with {len(masked_crowns)} trees.")

if __name__ == "__main__":
    run_batch_pipeline()