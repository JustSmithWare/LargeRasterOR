import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class ImageData:
    """Class for bundling an Image and corresponding Dataset"""
    image: np.ndarray
    image_identifier: str
    df: pd.DataFrame

@dataclass
class ImageDataPaths:
    """Class for bundling an Image and corresponding Dataset's file paths"""
    image_path: str
    image_identifier: str
    df_path: str