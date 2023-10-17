# Large Raster Object Recognition Model Training
### Summary
The ´main.ipynb´ notebook gathers up the necessary code to train a RetinaNet-based object recognition model using pytorch lightning.

What makes this model and training process stand out, is that it's mainly oriented to train on really large raster images, namely satellite imagery of the earth's surface, astronomical imagery, or any simmilar task where raster data is often in the form of large, high-resolution files.  

In this notebook you will find many utilities made to deal with large image files and datasets, from a pipeling for cleaning and augmenting the data, to a pytorch-lightning based machine learning workflow that performs a grid search for optimal hyperparameters while training an object recognition model on tiles that were generated from the original rasters.
### Main Features
- Use of multiprocessing Pools to deal with the large amount of computing needed to preprocess the datasets and images by code paralellization.
- Use of tensor operations to split images and generate bounding boxes from object's position and diameter.
- Flexible functions that can be fed a variable amount of filters or operations created on the fly.
- Custom implemented common data augmentation techniques for iamges and datasets.
- Many plotting and logging utilities to facilitate visualization and debugging.
- Grid search through different hyperparameters.
- Pytorch lightning module wrapping a RetinaNet based model.
- A ModelConfig class for storing model hyperparameters.
- Custom model logging for training, validation and testing results, supporting checkpointing.
- A simple heatmap plot to review the results of the best performing model after training.
### Disclaimer
The images and data used for model training are not in the public domain. As a result, some of the debugging plots might be off by default.  
This notebook and the adjacent modules serve exclusively as a *code sample*, as such, I do not plan on regularly mantaining it.
### How to use
The required steps for running this code are the following:
1. Fill the data/datasets local directory with csv files each containing one dataset per raster tiff file.

2. Fill the data/images local directory with one raster tiff file per dataset/csv file.

3. Fill the DF_FILENAMES and IMAGE_FILENAMES variables with the names of your dataset and raster files, make sure the i'th dataset corresponds to the i'th raster file for all entries.

4. Run the notebook