import os
import rasterio
from rasterio.windows import Window
import numpy as np

base_path = "path_to/Augsburg_data_4_publication/entire_city" # <-- update this with you path to "Augsburg_data_4_publication/entire_city"

# Funció per retallar i guardar subpatches alineats en la resta d'imatges
def save_aligned_patches(image_path, ref_dst_filename, output_filename):
    with rasterio.open(image_path) as src:
        if os.path.exists(output_filename):
            print(f"{output_filename} ja existeix, saltant...", flush=True)
        else:
            print(f"Processant {image_path}...", flush=True)
            with rasterio.open(ref_dst_filename) as ref_patch:
                ref_transform = ref_patch.transform
                ref_bounds = rasterio.transform.array_bounds(ref_patch.height, ref_patch.width, ref_transform)
                
                # Convertim els límits de la referència en coordenades de la imatge objectiu
                window = rasterio.windows.from_bounds(*ref_bounds, transform=src.transform)
                
                patch = src.read(window=window)
                patch_transform = rasterio.windows.transform(window, src.transform)
                
                with rasterio.open(
                    output_filename, "w", driver="GTiff", height=patch.shape[1], width=patch.shape[2],
                    count=src.count, dtype=src.dtypes[0], crs=src.crs, transform=patch_transform
                ) as dst:
                    dst.write(patch)
                print(f"{output_filename} done!", flush=True)


image_filenames = [
    "3K_DSM.tif", "3K_RGB.tif", "EeteS_EnMAP_10m.tif", "EeteS_EnMAP_30m.tif", 
    "EeteS_Sentinel_2_10m.tif", "HySpex.tif", "Sentinel-1.tif", "Sentinel-2.tif"
]
ref_image_filename = "EeteS_EnMAP_10m.tif"

splits = ("valid", "train")
patch_sizes = ((457, 888), (914, 888))

for split, patch_size in zip(splits, patch_sizes):
    # Crop original EeteS_EnMAP_10m.tif file
    ref_dst_filename = f"data/MDAS/{split}/EeteS_EnMAP_10m.tif"
    image_path = os.path.join(base_path, ref_image_filename)

    with rasterio.open(image_path) as src:
        transform = src.transform
        window = Window(0, 0, patch_size[0], patch_size[1])
        patch = src.read(window=window)
        patch_transform = rasterio.windows.transform(window, transform)
        
        with rasterio.open(
            ref_dst_filename, "w", driver="GTiff", height=patch_size[1], width=patch_size[0],
            count=src.count, dtype=src.dtypes[0], crs=src.crs, transform=patch_transform
        ) as dst:
            dst.write(patch)
    print(f"{ref_dst_filename} created.", flush=True)

    # Crop the rest of the images acording to the reference crop of EeteS_EnMAP_10m.tif
    target_images = [img for img in image_filenames if img != ref_image_filename]
    for img_filename in target_images:
        img_path = os.path.join(base_path, img_filename)
        output_filename = os.path.join(f"data/MDAS/{split}", img_filename)
        save_aligned_patches(img_path, ref_dst_filename, output_filename)