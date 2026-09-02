import os
import sys
import argparse
from pathlib import Path

try:
    import pycolmap
except ImportError:
    print("Error: pycolmap is not installed. Please install it using 'pip install pycolmap'")
    sys.exit(1)

try:
    import open3d as o3d
    import numpy as np
except ImportError:
    print("Warning: open3d or numpy not installed. Visualizations will be skipped.")
    o3d = None

def run_pipeline(image_dir_str, output_dir_str, log_callback=print, show_viz=False):
    image_dir = Path(image_dir_str)
    output_dir = Path(output_dir_str)

    if not image_dir.exists():
        log_callback(f"Error: Image directory '{image_dir}' does not exist.")
        return None

    image_files = list(image_dir.glob("*.[jJ][pP][gG]")) + list(image_dir.glob("*.[pP][nN][gG]"))
    if not image_files:
        log_callback(f"Error: No images found in '{image_dir}'.")
        return None

    log_callback(f"Found {len(image_files)} images. Starting reconstruction pipeline...")
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "database.db"

    # Step 1: Feature Extraction
    log_callback("\n--- Extracting Features ---")
    if database_path.exists():
        database_path.unlink() # Start fresh

    # Pycolmap writes to C++ stdout which is hard to capture in Python sys.stdout.
    # But it will print to the console.
    pycolmap.extract_features(database_path, image_dir)

    # Step 2: Feature Matching
    log_callback("\n--- Matching Features ---")
    pycolmap.match_exhaustive(database_path)

    # Step 3: Incremental Mapping (Structure from Motion)
    log_callback("\n--- Incremental Mapping (SfM) ---")
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_dir)

    if not maps:
        log_callback("Error: Failed to reconstruct any models. Try taking more overlapping photos.")
        return None

    # Use the best model (typically the first one)
    best_model = maps[0]
    log_callback(f"\nSuccess! Reconstructed model with {best_model.num_reg_images()} registered images and {best_model.num_points3D()} 3D points.")

    # Export to PLY
    ply_path = output_dir / "sparse_model.ply"
    best_model.export_PLY(str(ply_path))
    log_callback(f"Exported sparse point cloud to: {ply_path}")

    # Step 4: Visualization (if Open3D is available)
    if show_viz and o3d is not None and ply_path.exists():
        log_callback("\n--- Visualizing Point Cloud ---")
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        log_callback("Close the visualization window to exit.")
        o3d.visualization.draw_geometries([pcd], window_name="3D Pipe Reconstruction")
        
    return str(ply_path)

def main():
    parser = argparse.ArgumentParser(description="3D Reconstruction using pycolmap")
    parser.add_argument("--image_dir", type=str, default="images", help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the reconstruction output")
    args = parser.parse_args()
    run_pipeline(args.image_dir, args.output_dir, show_viz=True)

if __name__ == "__main__":
    main()
