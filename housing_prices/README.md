# Project 1: Model Evaluation & Validation
## Predicting Boston Housing Prices

### Description

This project aims to predict housing prices based on:
- `'RM'` is the average number of rooms among homes in the neighborhood.
- `'LSTAT'` is the percentage of homeowners in the neighborhood considered "lower class" (working poor).
- `'PTRATIO'` is the ratio of students to teachers in primary and secondary schools in the neighborhood.

We will evaluate the performance and predictive power of a DecisionTreeRegressor model on data collected from homes in suburbs of Boston, Massachusetts. The aim of the model is to make predictions about the monetary value of a home which would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis.
 
### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

### Code

There is an additional Python file at `visuals.py` that the notebook needs to visualize graphics.

### Run

To open the noteboook, in a terminal or command window, run one of the following commands:

```ipython notebook```  
```jupyter notebook```

then select `'boston_housing.ipynb'` in your directory. This will open the iPython Notebook software and project file in your browser.

### Data

Download the dataset `housing.csv` used in this project.

The dataset is Boston housing data collected in 1978. Each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. 

The following preprocessing steps have been made to the dataset:

- 16 data points have an 'MEDV' value of 50.0. These data points likely contain missing or censored values and have been removed.
- 1 data point has an 'RM' value of 8.78. This data point can be considered an outlier and has been removed.
- The features 'RM', 'LSTAT', 'PTRATIO', and 'MEDV' are essential. The remaining non-relevant features have been excluded.
- The feature 'MEDV' has been multiplicatively scaled to account for 35 years of market inflation.

