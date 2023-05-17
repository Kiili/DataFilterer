# Dataset filterer

## Setup
Tested with: 
- Ubuntu 20.04, Ubuntu 22.04, Arch Linux
- Python 3.8 and 3.10
- CUDA versions 11.4-11.6, 12.1
- pytorch 2.x, pytorch>=1.12.1

### pip
`pip install -r requirements.txt`

In case of “Failed to import CuPy” error, uninstall all cupy versions 
`pip uninstall cupy_cuda11x cupy_cuda12x`
and reinstall the one according to your cuda version

In case of missing `libcusolver.so.11` error, a potential solution is to install a 
[1.x version of torch](https://pytorch.org/get-started/previous-versions/).

## Usage
It is required to create a class which extends the `FilterableInterface` class. See `filterable_interface.py`

### Application
- Run application `python app.py`
- Import your python file that contains a `FilterableInterface` class
- Choose a layer to extract feature maps from
- Optionally get an overview of the filterable dataset. This does an iterative pass over the dataset
- Set the percentages for the amount of outliers and semantically similar images removed (0.0%-100.0%)
- Optionally choose a different dimensionality reduction method. 
This method is used for getting 3D plot points and for downscaling the feature maps
  - `Uniform Manifold Approximation and Projection` is recommended and used by default
  - `Principal Component Analysis` is also recommended
  - Other methods are not recommended due to poor results
- Optionally override the reduced dimensionality of the feature maps that are used while filtering.
  - `overriden_value=0` translates to `overriden_value=inf`
  - The final dimensionality of the feature maps is calculated as `N_dim = min(feature_map_dim, dataset_size, overriden_value)`
- Filter the dataset. The metadata of items in the filtered dataset are written to a text file
- After filtering, it is possible to:
  - View the left-out images 
    - Requires the `get_image()` function to be implemented
  - Interact with a 3D point cloud representation of the whole dataset
    - The click-to-view-image functionality significantly lowers the 
        performance of the plot in case of ~2000+ items in the dataset

### Python code
```python
from data_filterer import DataFilterer
from x import MyModel

# Define the data filterer. Provided model must extend FilterableInterface class
# See print(MyModel().get_model()) for model layer names (or use the application)
# Submodules are separated by "."
filterer = DataFilterer(model=MyModel(), layer="module1.layer2.conv")

# Filter the dataset by removing 5% of similars and outliers
metadata = filterer.get_idxs(semantic_percentage=0.05, outlier_percentage=0.05)

# Metadata of each item that is in the filtered dataset. See filterable_interface.py
print(metadata)
```
See examples folder for additional usage.

## Troubleshooting
### `cuda::OutOfMemoryError`
- If this happens during feature map extraction, choose a layer with less features. 
    Caching feature maps to the disk is currently not supported
- If this happens during filtering, reduce the dimensionality of the feature maps or use another dimensionality reduction method

### Known issues
- Due to unexpected behaviour of HDBSCAN, it is possible for the 
    algorithm to find less outliers than the targeted amount.
    This occurs more frequently when feature maps are forming groups/clusters. 
    To counteract this, it is recommended to use prior layers
- NVIDIA RAPIDS library [has no Windows support](https://developer.nvidia.com/blog/run-rapids-on-microsoft-windows-10-using-wsl-2-the-windows-subsystem-for-linux/).
    It is required to use WSL
- Using the CPU for filtering uses significantly more memory. 
    Also, the results differ compared to using the GPU. 
    Thus, it is experimental and not recommended
- Small memory leaks could occur from specific actions in the application
