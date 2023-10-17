"""
OS and IO Operations Utilities
------------------------------

This module offers a collection of utility functions that handle file input/output and operating system-related tasks.
These utilities encompass reading raster images, generating DataFrame identifiers from tile names, managing directories,
and various functions that deal with the selection, logging, and validation of images and paths.
"""

import numpy as np
import pandas as pd
import random
import rasterio
import os
import shutil
import tempfile
from ImageData import ImageData
from logging import Logger

def read_raster(filepath: str) -> np.ndarray:
    """
    Reads a raster image from its path or a file-like object.

    Parameters:
    - file (str or file-like): File path or object from which to read the raster data. 

    Returns:
    - numpy.ndarray: The numpy array that results from reading the file.
    """
    with rasterio.open(filepath) as src:
        img_array = src.read()

    img_transposed = np.transpose(img_array, (1, 2, 0))
    return img_transposed

def identifier_from_tilename(string_tilename: str) -> str:
    """
    Returns a Dataframe identifier from a tile's filename.

    Parameters:
    - string_tilename (str): String from which the identifier will be retreived. 

    Returns:
    - str: The DataFrame identifier.
    """
    return string_tilename.split('/')[-1].partition('.')[0]

def recreate_directory(directory_path: str) -> str:
    """
    Parameters:
    - directory_path (str): Path to the directory that will be re-created.

    Returns:
    - str: The newly created directory's path.
    """
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)

    os.makedirs(directory_path, exist_ok=True)
    return directory_path

def recreate_temp_directory(logger: Logger) -> str:
    """
    Returns a temporary directory's path after cleaning '/training' and '/testing' subdirectories.

    Returns:
    - str: The temporary directory's path.
    """
    tmpdir = tempfile.gettempdir()

    dirs_to_remove = [os.path.join(tmpdir, 'training'), os.path.join(tmpdir, 'testing')]

    for dir_path in dirs_to_remove:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        except Exception as e:
            logger.exception(f'Failed to delete {dir_path}. Reason: {e}')

    return tmpdir

def random_image_data(tiling_directory: str) -> ImageData:
    """
    Returns an ImageData object created from a random image in a directory that contains tiles and annotations.

    Parameters:
    - tiling_directory (str): Path to the directory containing the tiles and their annotations.

    Returns:
    - ImageData: An ImageData object created from a random tile and its annotations.
    """
    csv_path = os.path.join(tiling_directory, 'tile.csv')
    png_files = [file for file in os.listdir(tiling_directory) if file.endswith(".png")]
    random_image = random.choice(png_files)
    df = pd.read_csv(csv_path)
    return ImageData(read_raster(os.path.join(tiling_directory, random_image)),
                      random_image.split('.')[0],
                      df[df.tile_path == random_image])

def check_unreferenced(df: pd.DataFrame, directory: str, logger: Logger, column_name: str, remove: bool=False) -> int | pd.DataFrame:
    """
    Debugging utility. Checks for tile images that do not exist but are referenced by an annotation's file.
    
    Given a DataFrame `df` with a column name containing file paths, this function
    compares these paths to the actual files present in a directory. It logs the discrepancies
    and, if requested, removes the dangling references from the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing file paths in one of its columns.
    - directory (str): The directory against which to check the file paths.
    - logger (Logger): The logging object to capture debug information.
    - column_name (str): The name of the DataFrame column containing the file paths.
    - remove (bool, optional): Whether to remove the unreferenced paths from the DataFrame. 
                               Defaults to False.
    
    Returns:
    - int | pd.DataFrame: If `remove` is False, returns the number of unreferenced paths. 
                          If `remove` is True, returns the cleaned DataFrame.
    """

    image_paths = df[column_name].unique()
    files = {f for f in os.listdir(directory)}
    dangling_paths = {path for path in image_paths if str(path).split('/')[-1] not in files}
    
    logger.debug(f'Unique tile paths in DF: {image_paths}')
    logger.debug(f'Files in directory: {files}')
    logger.debug(f'There are {len(dangling_paths)} non-existing files: {dangling_paths}')

    if remove:
        logger.debug(f'Deleted references to {len(dangling_paths)} files.')
        return df[~df[column_name].isin(dangling_paths)]
    
    return len(dangling_paths)