README


1. DESCRIPTION
- This project predicts credit card payment default utilizing general client and financial performance information.  
- The dataset used to train the models came from the UCI Machine Learning Repository.  
- It consists of 30,000 observations and 23 features. 
- The prediction of the best model surpasses both the baseline 78% default rate of the dataset and the benchmark model. 
- It is mostly useful to credit card issuers and financial institutions interested in modeling and preventing delinquencies in their consumer accounts.  


2. INSTALLATION
The code is written on Python 2.7. To run it, you need to have Python installed and an iPython Notebook such as Jupyter.  

Download the dataset in the same local directory as all the files included: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#


3. LIBRARIES
The following libraries need to be installed in your machine before running the program:

numpy, pandas, seaborn, matplotlib, sklearn, feature_selection.RFE, scipy.stats.boxcox, MinMaxScaler, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV, 
f1_score, RandomForestClassifier, XGBClassifier, LogisticRegression, MLPClassifier, IPython, jupyter, hide_code


4. CONFIGURATION
No additional setup/startup or initial configuration is needed.  


5. DOCUMENT DESCRIPTION
- default of credit card clients.xls: project dataset
- Capstone_Project_Notebook.ipynb: iPython notebook includes complete report, code, output and results
- Project_Report.pdf: condensed version of the notebook in pdf format
- Image files needed for displaying on notebook: Perceptron1.jpg, Multilayer1.png, GBM.jpg, LRcurve.png