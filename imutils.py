"""
Image Processing Utilities
--------------------------

This module provides a set of utility functions for image processing. These include resampling images, 
rotating bounding boxes, splitting images into tiles, and updating coordinates for tiled images.
"""

from PIL import Image
import numpy as np
import pandas as pd
import os

def resample_image(img_array: np.ndarray, scale: float) -> np.ndarray:
    """
    Resamples the given image array by tripling its size.

    Parameters:
    - img_array (numpy.ndarray): The input image array with shape (w, h, C).
    - scale (float): Scaling factor for image resizing.
    Returns:
    - numpy.ndarray: The resampled image array with shape (3*w, 3*h, C).
    """

    img = Image.fromarray(img_array.astype(np.uint8))

    img_resampled = img.resize((img_array.shape[1] * scale, img_array.shape[0] * scale), Image.LANCZOS)

    return np.array(img_resampled)

def rotate_bbox_90(xmin: int, ymin: int, xmax: int, ymax: int, img_width: int, img_height: int) -> tuple[int, int]:
    return ymin, img_width - xmax, ymax, img_width - xmin

def rotate_bbox_180(xmin: int, ymin: int, xmax: int, ymax: int, img_width: int, img_height: int) -> tuple[int, int]:
    return img_width - xmax, img_height - ymax, img_width - xmin, img_height - ymin

def rotate_bbox_270(xmin: int, ymin: int, xmax: int, ymax: int, img_width: int, img_height: int) -> tuple[int, int]:
    return img_height - ymax, xmin, img_height - ymin, xmax

def rotate_center(x: int, y: int, angle: int, w: int, h: int) -> tuple[int, int]:
    """
    Rotates a point by a given angle around the center of an image.

    Parameters:
        - x, y (int): Coordinates of the point.
        - angle (int): Angle in degrees to rotate the point.
        - w, h (int): Dimensions of the image.

    Returns:
        - tuple[int, int]: New coordinates of the rotated point.
    """
    if angle == 90:
        return h - y, x
    elif angle == 180:
        return w - x, h - y
    elif angle == 270:
        return y, w - x
    else:
        return x, y

def split_and_save_tiles(tiling_directory: str, tile_size: int, image_name: str, image_array: np.ndarray) -> np.ndarray:
    """
    Splits an image into tiles and saves them as local files.

    Parameters:
        - tiling_directory (str): The directory where tiles will be saved.
        - tile_size (int): Side length of each square tile.
        - image_name (str): Original name of the image being tiled.
        - image_array (numpy.ndarray): The image array to be tiled.

    Returns:
        - list: A list of saved tile arrays.
        - list: A list of tile coordinates as tuples (xmin, ymin, xmax, ymax).
    """
    tiles = []
    tile_coordinates = []
    height, width = image_array.shape[:2]
    
    n_tiles_x = (width + tile_size - 1) // tile_size
    n_tiles_y = (height + tile_size - 1) // tile_size
    
    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            xmin = j * tile_size
            ymin = i * tile_size
            xmax = min(xmin + tile_size, width)
            ymax = min(ymin + tile_size, height)
            
            tile = image_array[ymin:ymax, xmin:xmax]
            
            tile_name = os.path.join(tiling_directory, f"tiling_{tile_size}", f"tile_{image_name}_{i}_{j}.png")
            Image.fromarray(tile).save(tile_name)
            
            tiles.append(tile)
            tile_coordinates.append((xmin, ymin, xmax, ymax))
            
    return tiles, tile_coordinates


def find_tile_and_update_coords(row: pd.Series, tile_coordinates: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], str]:
    """
    Finds the tile that maximizes the overlap with a given bounding box and updates the coordinates.

    Parameters:
        - row (pd.Series): A row from a DataFrame containing bounding box coordinates.
        - tile_coordinates (list): A list of tile coordinates as tuples (xmin, ymin, xmax, ymax).

    Returns:
        - tuple: Updated coordinates of the bounding box relative to the best-fitting tile.
        - str: Path to the best-fitting tile.
    """
    xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    
    max_overlap = 0
    best_tile = None
    best_tile_index = None
    
    for i, (tile_xmin, tile_ymin, tile_xmax, tile_ymax) in enumerate(tile_coordinates):
        overlap_xmin = max(xmin, tile_xmin)
        overlap_ymin = max(ymin, tile_ymin)
        overlap_xmax = min(xmax, tile_xmax)
        overlap_ymax = min(ymax, tile_ymax)
        
        if overlap_xmin < overlap_xmax and overlap_ymin < overlap_ymax:
            overlap_area = (overlap_xmax - overlap_xmin) * (overlap_ymax - overlap_ymin)
            if overlap_area > max_overlap:
                max_overlap = overlap_area
                best_tile = (tile_xmin, tile_ymin, tile_xmax, tile_ymax)
                best_tile_index = i
                
    new_xmin = xmin - best_tile[0]
    new_ymin = ymin - best_tile[1]
    new_xmax = xmax - best_tile[0]
    new_ymax = ymax - best_tile[1]
    tile_path = f"tile_{best_tile_index}.png"
    
    return (new_xmin, new_ymin, new_xmax, new_ymax), tile_path