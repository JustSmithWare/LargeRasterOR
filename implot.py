"""
Plotting Utilities for Image Data
---------------------------------

This module provides a set of functions to visualize image data along with any associated annotations.
It includes utilities to plot random tiles, individual image data, and overlays of bounding boxes
on images for object detection tasks.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from ImageData import ImageData

from fileio import random_image_data

def plot_image_with_annotations(image, filtered_annotations):   
    
    fig, ax = plt.subplots(figsize=(10, 10))

    
    ax.imshow(image)

    if len(filtered_annotations) > 0:
        for i, row in filtered_annotations.iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
    else:
        print("No annotations found")

    plt.show()

def plot_image_data(image_data: ImageData):   
    plot_image_with_annotations(image_data.image, image_data.df)

def plot_random_tile(tile_dir: str):
    im_data = random_image_data(tile_dir)
    plot_image_data(im_data)

def plot_image(image_array):
    plt.imshow(image_array)
    plt.show()
