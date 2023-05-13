# Data filterer

## Setup
Tested on Python 3.8 and 3.10. 

### pip
`pip install -r requirements.txt`

In case of “Failed to import CuPy” error, uninstall all cupy versions 
`pip uninstall cupy_cuda11x cupy_cuda12x`
and reinstall the one according to your cuda version

In case of missing `libcusolver.so.11` error, a potential solution is to install a 
[1.x version of torch](https://pytorch.org/get-started/previous-versions/).

## Usage
Create a class which extends the FilterableInterface. See example.py

- Run application `python app.py`
- Import your model
- Choose a layer to extract features from
- Set the percentages for the amount of outliers and semantically similar images removed
- Filter the dataset

