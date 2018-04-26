'''
PREDICTION OF CREDIT CARD PAYMENT DEFAULT 
ALFRED K. HUNG, akh022@gmail.com 
February 9, 2018 
'''

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
f1_scorer = make_scorer(f1_score, average='weighted')


# Import libraries and methods
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.feature_selection import RFECV
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from IPython.display import display
import gc
get_ipython().magic(u'matplotlib inline')
# Reclaims memory occupied by objects that are no longer in use by the program 
gc.enable()

# Load the dataset
# Label columns with abbreviations. For monthly columns: 'D' for Delay, 'B' for Billing, 'P' for Payment
nam = ['Lim', 'Sex', 'Edu', 'Mar', 'Age', 'DSep', 'DAug', 'DJul', 'DJun', 'DMay', 'DApr', 'BSep', 'BAug', 'BJul', 'BJun', 'BMay', 'BApr', 'PSep', 'PAug', 'PJul', 'PJun', 'PMay', 'PApr', 'y']
# Set categorical columns as string type
conv = {"SEX":str, "EDUCATION":str, "MARRIAGE":str, "PAY_0":str, "PAY_2":str, "PAY_3":str, "PAY_4":str, "PAY_5":str,"PAY_6":str}
# Read dataset and do column name conversions
data = pd.read_excel("default of credit card clients.xls", header=0, index_col=0, skiprows=1, names = nam, converters = conv)


# Show the list of columns
print "\nFeature columns:\n\n{}".format(list(data.columns))

# Show the feature information by printing the first five rows
print "\nFeature values:"
display(data.head(3))

# Summary statistics of dataset
print "\nSummary Statistics:"
display(data.describe())

nrows = data.shape[0]
print "\nCredit Dataset:"
print "Total number of observations: {}".format(nrows)

ncols = data.shape[1]-1
print "Total number of features: {}".format(ncols)

# Calculate total defaults
deft = sum(data.y)
print "\nCredit Statistics:"
print "Total Credit Defaults: {}".format(deft)

# Calculate total non-defaults
ndeft = nrows - deft
print "Total Credit Non-Defaults: {}".format(ndeft)

# Calculate percentage of defaults
pct_deft = deft*100./nrows
print "Default percentage: {:.2f}%".format(pct_deft)

# Percentage of males and females
s = data.shape[0]
f = data.Sex.value_counts()[0]*100./s
print "\nSex Feature Breakdown:"
print "Females: {:.2f}%".format(f)
print "Males: {:.2f}%".format(100.-f)

# Percentage of each Educational level
c = data.Edu.value_counts()[0]*100./s
g = data.Edu.value_counts()[1]*100./s
h = data.Edu.value_counts()[2]*100./s
print "\nEducation Feature Breakdown:"
print "Graduate level: {:.2f}%".format(g)
print "College level: {:.2f}%".format(c)
print "High School level: {:.2f}%".format(h)
print "Other educational levels: {:.2f}%".format(100.-(g + c + h))

# Percentage of each Marital Status
sg = data.Mar.value_counts()[0]*100./s
mr = data.Mar.value_counts()[1]*100./s
print "\nMarital Status Feature Breakdown:"
print "Married: {:.2f}%".format(mr)
print "Singles: {:.2f}%".format(sg)
print "Divorce and Other: {:.2f}%".format(100.-(mr + sg))

# Percentage of Payment Delays [-2, -1, 0]
dcols = [ feat for feat in nam if feat.startswith('D') ]
dc = data[dcols] # Get Payment delay data
d1 = pd.DataFrame(data['DSep'].value_counts()) # get number count for Sep
for i in dcols[1:7]: # exclude 'DSep' in loop
    d2 = pd.DataFrame(dc[i].value_counts()) # Get number count for next month
    d1 = pd.merge(d1, d2, left_index=True, right_index=True, how='outer') # put both counted columns together
d1['sum'] = d1.sum(axis=1) # sum all rows and put in 'sum' column
d11 = d1.iloc[:3,:] # create separate table for data of clients who pay on time
d11 = d11.copy()
d11['pct'] = pd.Series.round(d11['sum']*100/d11['sum'].sum(), 2) # calculate percentages of clients who pay on time
print "\nCounts and percentage of clients without payment delays:"
display(d11)

# Percentage of values [-2, -1, 0] in Delay columns
d11['DSep_pct'] = pd.Series.round(d11['DSep']*100/data.shape[0], 2) # calculate percentage of each value category for Sep
d11['DMay_pct'] = pd.Series.round(d11['DMay']*100/data.shape[0], 2) # calculate percentage of each value category for May
d11.loc['Total'] = pd.DataFrame.sum(d11, axis=0) # calculate totals
print "\nPayment Delay Feature Breakdown (sample):"
display(d11[['DSep_pct', 'DMay_pct']])

# Percentage of Payment Delays 1 thru 8 
d12 = d1.iloc[3:11,:] # get table of columns with payment delays
d12 = d12.copy()
d12['pct'] = pd.Series.round(d12['sum']*100/d12['sum'].sum(), 2) # get percentages
print "\nCounts and percentage of clients with payment delays from 1 to 8 months:"
display(d12)

# Numerical Features
print "\nSample values for numerical features:"
col = ['Lim', 'Age', 'BSep', 'PSep']
for cl in col:
    print "Feature {} has {} values".format(cl, len(data[cl].value_counts()))
    
# Range: maximum and Minimum
print "\nSample range for numerical features:"
col = ['Age', 'BJul', 'PMay']
for cl in col:
    dif = max(data[cl])-min(data[cl])
    print "Feature {} has max-min difference of {:,}".format(cl, dif)
       
# Calculation function for percent defaults per feature
def pct_calc2(dfr, feat, arg=""):
    vals = list(np.unique(dfr[feat]))
    print "\nPercent default for{}:".format(arg)
    for v in vals:
        s0 = sum((dfr[feat] == v) & (dfr['y'] == 0))
        s1 = sum((dfr[feat] == v) & (dfr['y'] == 1))
        s01 = float(s1)*100/(s0+s1)
        print "{}={} is {:.2f}%".format(feat, v, s01)

# Calculations for Sex, Edu, Mar
col = ['Sex', 'Edu', 'Mar']
for cl in col:
    pct_calc2(data, cl)

# Sample calculations for Delay Payment feature
txt = ' (sample)'
pct_calc2(data, 'DSep', txt)

# Get default percentages for numerical features
cols = data.columns[data.dtypes=='int64']
datc2 = data.copy()
datc2 = datc2[cols]
# Split continuous columns into quartile ranges
col = ['Age', 'Lim', 'BSep', 'PSep']
for cl in col:
    datc2[cl] = list(pd.qcut(datc2[cl].values, 5, duplicates='drop').codes)

# Sample calculations for numerical features
txt = ' (5 quantiles)'
for cl in col:
    pct_calc2(datc2, cl, txt)

# Calculation function for 3 features:
def pct_calc3(dfr, feat1, feat2, feat3):
    lst = []
    val1 = list(np.unique(dfr[feat1]))
    val2 = list(np.unique(dfr[feat2]))
    val3 = list(np.unique(dfr[feat3]))
    for x in val3:
        for w in val2:
            for v in val1:
                s0 = sum((dfr[feat1] == v) & (dfr[feat2] == w) & (dfr[feat3] == x) & (dfr['y'] == 0))
                s1 = sum((dfr[feat1] == v) & (dfr[feat2] == w) & (dfr[feat3] == x) & (dfr['y'] == 1))
                s01 = (s0+s1)
                if s01 == 0:
                    continue
                sm = float(s1)*100/s01
                lst.append([v, w, x, round(sm, 2)])
    cols = [feat1,  feat2, feat3, '%_default']
    df = pd.DataFrame(lst, columns=cols)
    df = df.sort_values('%_default', ascending=False)
    print "\nTop 3 defaults for three feature combination:"
    return display(df.head(3))

pct_calc3(data, 'Sex', 'Mar', 'Edu')
pct_calc3(data, 'Age', 'Edu', 'Sex')
pct_calc3(data, 'Lim', 'Edu', 'Sex')


### Pair plots with Correlation numbers
# Produce a scatter matrix for each pair of features in the data
axes = pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')


### Categorical plots
# Multiple features plot with Categorical and Continuous features
g = sns.factorplot(x="Edu", y="Age", hue="y",
               col="Sex", data=data, kind="box", size=4, aspect=1, margin_titles=True)
g.fig.suptitle("Graph 2: Age Boxplot by Education & Sex\n")
plt.tight_layout()
plt.subplots_adjust(top=0.75)
plt.show()
# Age vs. Marital by Sex columns
g = sns.factorplot(x="Mar", y="Age", hue="y",
               col="Sex", data=data, kind="box", size=4, aspect=1, margin_titles=True)
g.fig.suptitle("Graph 3: Age Boxplot by Marital Status & Sex\n")
plt.tight_layout()
plt.subplots_adjust(top=0.75)
plt.show()
# Credit limit vs. Education by Sex Columns
g = sns.factorplot(x="Edu", y="Lim", hue="y",
               col="Sex", data=data, kind="box", size=4, aspect=1, margin_titles=True)
g.fig.suptitle("Graph 4: Credit Limit Boxplot by Education & Sex\n")
plt.tight_layout()
plt.subplots_adjust(top=0.75)
plt.show()
# Credit Limit vs. Marital by Sex Columns
g = sns.factorplot(x="Mar", y="Lim", hue="y",
               col="Sex", data=data, kind="box", size=4, aspect=1, margin_titles=True)
g.fig.suptitle("Graph 5: Credit Limit Boxplot by Marital Status & Sex\n")
plt.tight_layout()
plt.subplots_adjust(top=0.75)
plt.show()


### Continuous features plots

# Billings histogram
Bcols = [ feat for feat in nam if feat.startswith('B') ]  # get billing columns only
datc2['Bavg'] = datc2[Bcols].sum(axis=1)/6 # calculate average billings
# focus on densest part of histogram except negative, zero billings; differentiate by default class  
filtered0 = datc2.Bavg[(datc2.Bavg >= 10) & (datc2.Bavg < 5000) & (datc2.y == 0)] 
filtered1 = datc2.Bavg[(datc2.Bavg >= 10) & (datc2.Bavg < 5000) & (datc2.y == 1) ]
plt.figure(figsize=(4,2)) # set graph size
# plot 100-bins histograms with density curve
sns.distplot(filtered0, kde=True, hist=True, bins=100, label='Non-defaults', color="blue")
sns.distplot(filtered1, kde=True, hist=True, bins=100, label='Defaults', color="darkorange")
plt.title("Graph 6: Six-months-average Billings Histogram (truncated)\n")
plt.xlabel('\nSix Month Average Billing Amounts\n')
plt.legend()
plt.show()

# Payments histogram
Pcols = [ feat for feat in nam if feat.startswith('P') ] # get payments columns only
datc2['Pavg'] = datc2[Pcols].sum(axis=1)/6 # calculate average payments
# focus on densest part of histogram excepting zero payments; differentiate by default class
filtered0 = datc2.Pavg[(datc2.Pavg >= 10) & (datc2.Pavg < 10000) & (datc2.y == 0) ]
filtered1 = datc2.Pavg[(datc2.Pavg >= 10) & (datc2.Pavg < 10000) & (datc2.y == 1) ]
plt.figure(figsize=(4,2)) # set graph size
# plot 100-bins histograms with density curve
sns.distplot(filtered0, kde=True, hist=True, bins=100, label='Non-defaults', color="blue")
sns.distplot(filtered1, kde=True, hist=True, bins=100, label='Defaults', color="darkorange")
plt.title("Graph 7: Six-months-average Payments Histogram (truncated)\n")
plt.xlabel('\nSix Month Average Payments Amounts\n')
plt.legend()
plt.show()

# Histogram comparisons
g = sns.factorplot(x="Age", col="Sex", hue="y", data=data, kind="count", size=4, aspect=1, margin_titles=True)
xt = np.append(0, np.arange(9, 60, 10)) 
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels):
        if i not in xt : labels[i] = '' # label every 10 ticks, skip the rest
    ax.set_xticklabels(labels) # set new labels
g.fig.suptitle("Graph 8: Age Histogram by Sex\n")
plt.tight_layout()
plt.subplots_adjust(top=0.75)
plt.show()

# Histogram of credit limit by Sex columns
g = sns.factorplot(x="Lim", col="Sex", hue="y", data=data, kind="count", size=4, aspect=1, margin_titles=True)
xt = np.arange(0, 90, 10)
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels):
        if i not in xt : labels[i] = '' # label every 10 ticks, skip the rest
    ax.set_xticklabels(labels, rotation=30) # set new labels
g.fig.suptitle("Graph 9: Credit Limit Histogram by Sex\n")
plt.tight_layout()
plt.subplots_adjust(top=0.75)
plt.show()


#### General Functions for training and prediction 

# Set up general functions for training, prediction, train scoring and test scoring
def train_classifier(clf, X_train, y_train):
    # Fits a classifier to the training data
    clf.fit(X_train, y_train)
    
def predict_labels(clf, features, target):
    # Makes predictions using a fit classifier based on F1 score
    y_pred = clf.predict(features)
    return f1_score(target.values, y_pred, average='weighted')

def train_predict(clf, X_train, y_train, X_test, y_test):
    # Train and predict using a classifer based on F1 score
    # Indicate the classifier
    print "\nTraining and Prediction using a {} : ".format(clf.__class__.__name__)
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.6f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.6f}.".format(predict_labels(clf, X_test, y_test))


# Separate the data into feature data and target data 
y = data['y']
data = data.drop(labels='y', axis=1)

# Set up general seed for random state to make results reproducible 
rs = 57

# Save copy for the computation of benchmark's sensitivity distribution
data_c1 = data.copy()


#### Partition into training and testing sets

# Shuffle and split the dataset into a 75/25 ratio of training and testing points for the benchmark

sss = StratifiedShuffleSplit(n_splits=20, test_size=0.25, random_state=rs)
for train_index, test_index in sss.split(data, y):
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


#### Benchmark

# Set up Benchmark model
clf = RandomForestClassifier(random_state=rs, class_weight='balanced')
print "\nBenchmark Model:"
# Train and get prediction for benchmark
train_predict(clf, X_train, y_train, X_test, y_test)

# Save test score result for sensitivity distribution (Justification section)
lst2 = []
lst2.append(predict_labels(clf, X_test, y_test))

# Feature Ranking results from Random Forest
# Get order of importance of features
fi = pd.DataFrame({'Feature':data.columns, 'Feat_Imptc':clf.feature_importances_})
# Sort values in descending order
fisv = fi.sort_values(['Feat_Imptc', 'Feature'], ascending=[False, True])
print "\nFeature importances (top ten):"
display(fisv[:10])


#### Categorical Feature Transformation
# Transform categorical data using dummy variables
data = pd.get_dummies(data, prefix_sep='.', drop_first=True)

# View transformed columns
print "\nFeature columns processed with dummy indicators ({} total features):\n\n{}\n".format(len(data.columns), list(data.columns))

# Save copy of data for sensitivity analysis
data_c2 = data.copy()


# ### Partition into training and testing sets (2)
# 
# We apply train/test split to the expanded number of variables to prepare them for RFECV processing. 
# 
# Shuffle and split the expanded dataset after applying dummy vars
sss = StratifiedShuffleSplit(n_splits=20, test_size=0.25, random_state=rs)
for train_index, test_index in sss.split(data, y):
    X_train1, X_test1 = data.iloc[train_index], data.iloc[test_index]
    y_train1, y_test1 = y.iloc[train_index], y.iloc[test_index]


#### Feature Selection: Recursive Feature Elimination with Cross Validation

# Set up the estimator
clf = RandomForestClassifier(random_state=rs, class_weight='balanced')
# Setup Recursive Feature Elimination with Cross Validation
rfecv = RFECV(estimator=clf, step=1, cv=sss, scoring=f1_scorer, n_jobs=-1)
# Train RFECV
rfecv.fit(X_train1, y_train1)  
print("\nOptimal number of features : %d" % rfecv.n_features_)  
# Get and show selected feature columns
sel_cols = data.columns[rfecv.support_]
print "\nSelected Features by RFECV:\n\n{}".format(sel_cols)

# Get dataset with optimized features
data = data[sel_cols].copy()


# Plot number of features vs. cross-validation scores

plt.figure(figsize=(4,2))
plt.title('Graph 10: Feature Selection\nCV Scores vs. No. Features\n')
plt.xlabel("Number of features selected")
plt.ylabel("CV score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


#### Correlations 

# Extract correlations above 0.9
corr = data.corr().unstack().reset_index() # group together pairwise
corr.columns = ['feat1','feat2','corr'] # rename columns
# Show correlation results above 0.9 and exclude self-correlated variables (corr = 1)
print "\n\nSelected Features Correlated above 0.9"
display( corr[ (corr['corr'].abs() > 0.9) & (corr['feat1'] != corr['feat2']) ] ) 


# Correlation Heatmap

# Get correlation matrix
corrmat = data.corr()
# Plot heatmap
f, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.title('Correlation Heatmap of Selected Features\n')
plt.tight_layout()
plt.show()


# ### Partition into training and testing sets (3)
# 
# Before applying boxcox, we use StratifiedShuffleSplit again to split the selected features into train and test sets in preparation for modeling.  Once the original features changed, it's as if we started anew with a new dataset.  
# 
# Shuffle and split the feature-selected dataset

sss = StratifiedShuffleSplit(n_splits=20, test_size=0.25, random_state=rs)
for train_index, test_index in sss.split(data, y):
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


#### Feature Transformation: Boxcox 

# Transform numerical features to a more normal-looking distribution using Boxcox
X_train = X_train.copy()
X_test = X_test.copy()

# Transform Lim and Age features
numcol = ['Lim', 'Age']
for feat in numcol:
    X_train.loc[:, feat], lambda_ = boxcox(X_train[feat])
    X_test.loc[:, feat] = boxcox(X_test[feat], lmbda=lambda_)

# Transform Billing and Payment features that have negative or zero values by shifting by the minimum value + 1
numcol = Bcols + Pcols
for feat in numcol:
    X_train.loc[:, feat], lambda_ = boxcox(X_train[feat] + abs(min(X_train[feat])) + 1)
    X_test.loc[:, feat] = boxcox(X_test[feat] + abs(min(X_test[feat])) + 1, lmbda=lambda_)

# Plot numerical features after boxcox
print "Graph 11: X_train's histograms of numerical features after applying boxcox"
X_train[X_train.dtypes[X_train.dtypes=="float64"].index.values].hist(bins=100, figsize=[10,10])
plt.show()


#### Outliers Detection

outliers = []
print "\nNumber and percentage of outliers per feature:\n"
# For each feature find the data points with extreme high or low values
for feature in X_train.keys():
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(X_train[feature], 25)
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(X_train[feature], 75)
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3-Q1) * 1.5
    # Identify the outliers
    out_list = X_train[~((X_train[feature] >= Q1 - step) & (X_train[feature] <= Q3 + step))]
    # Print outliers
    print "{} is {} or {:.2f}%".format(feature, len(out_list), len(out_list)*100./X_train.shape[0])
    # Save outliers in list
    outliers.extend(list(out_list.index))

# Sort, delete duplicates from outlier index then put back into a list
outliers = list(set(sorted(outliers)))
print "\nTotal number of outliers is {:,} or {:.2f}%\n".format(len(outliers), len(outliers)*100./X_train.shape[0])


#### Min-Max Scale Transformation

# Show first rows of X_train after boxcox
print "\nX_train after boxcox:"
display(X_train.head(3))

# Statistics after boxcox
print "\nStatistics for X_train after boxcox:"
display(X_train.describe())

# Scale range of feature values within [0, 1]
X_scaler = MinMaxScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

# Show first few rows
print "\nX_train after Min-Max Scaling:"
display(X_train[:2])


### Principal Component Analysis

# Set up temporary X_train, X_test to be used only in this section
X_train_pca = X_train.copy()
X_test_pca = X_test.copy()
# Set up PCA with 10 components and fit it to training data
pca = PCA(n_components=10)
pca.fit(X_train_pca)
# Apply PCA transformation to both training and testing sets
pca_train = pca.transform(X_train_pca)
pca_test = pca.transform(X_test_pca)
# Get PCA components and variance information
pca.components_
pca.explained_variance_
pca.explained_variance_ratio_



# Training and Evaluating Models
# Initialize the three models
clf_A = MLPClassifier(random_state=rs)
clf_B = XGBClassifier(seed=rs)
clf_C = LogisticRegression(class_weight='balanced', random_state=rs)

# Set up the models
models = [clf_A, clf_B, clf_C]

# Execute the 'train_predict' function for each classifier
for clf in models:
    print "\n{}:".format(clf.__class__.__name__)
    if clf == clf_B:
        # Predict XGBoost with data transformed with dummy vars only
        train_predict(clf, X_train1, y_train1, X_test1, y_test1)
    else:
         # Predict other models with data transformed with dummy vars, boxcox, min-max scaling
        train_predict(clf, X_train, y_train, X_test, y_test)       

# Choose the Best Model and show its parameters
print "\nParameters of the Best Model: Logistic Regression"
display(clf.get_params(deep=True))

# Save LR's test prediction for sensitivity distribution
LR_pred1 = predict_labels(clf_C, X_test, y_test)


## Approach #1: Using RandomizedSearchCV
# Setup best model's parameters
clf = LogisticRegression(class_weight='balanced', random_state=rs)
# Define parameters and their range of values 
params = {'C': np.logspace(-5,5,11),
          'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']}
# Use randomized search on hyper parameters for fast performance
rand_srch = RandomizedSearchCV(clf, params, scoring = f1_scorer, cv=sss, n_jobs=-1, random_state=rs)
rand_srch.fit(X_train, y_train)
# Get the best estimator
clf = rand_srch.best_estimator_
print "\nThe Parameters for the Tuned Estimator are:"
display(clf)
# Report the final F1 score for training and testing after parameter tuning
print "\n{}'s Tuned Model: \n".format(clf.__class__.__name__)
print "Tuned model has a training F1 score of {:.6f}".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.6f}".format(predict_labels(clf, X_test, y_test))

## Approach #2: Using GridSearchCV
from sklearn.model_selection import GridSearchCV
# Setup best model's parameters
clf = LogisticRegression(class_weight='balanced', random_state=rs)
# Set up hyperparameters and training function
gs = GridSearchCV(clf, params, scoring=f1_scorer, cv=sss, n_jobs=-1)
gs.fit(X_train, y_train)
# Get the best parameters
clf = gs.best_params_
print "\nThe Parameters for the Tuned Estimator are:"
display(clf)
# Report the final F1 score for training and testing after parameter tuning
print "\n{}'s Tuned Model: \n".format(clf.__class__.__name__)
print "Tuned model has a training F1 score of {:.6f}".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.6f}".format(predict_labels(clf, X_test, y_test))


# Define best algorithm to use in refinement
clf = LogisticRegression(class_weight='balanced', random_state=rs)

# Define parameters and their range of values 
params = {'C': np.arange(0.51, 1.11, 0.1),
          'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']}

# Use randomized search on hyper parameters for fast performance
rand_srch = RandomizedSearchCV(clf, params, scoring = f1_scorer, cv=sss, n_jobs=-1, random_state=rs)
rand_srch.fit(X_train, y_train)

# Get the best estimator
clf = rand_srch.best_estimator_
print "\nThe Parameters for the Tuned Estimator are:"
display(clf)

# Report the final F1 score for training and testing after parameter tuning
print "\n{}'s Tuned Model: \n".format(clf.__class__.__name__)
print "Tuned model has a training F1 score of {:.6f}".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.6f}".format(predict_labels(clf, X_test, y_test))


#### Sensitivity Analysis for final model

# Function to get test results of sensitivity analysis. It selects features, does boxcox, scaling, training, prediction 
def robustness_test(ns=20, ts=0.25, rs=57, n=22500):
    # Retrieve dataset after dummy vars was applied
    data = data_c2.copy()
    # Get Selected features
    sss = StratifiedShuffleSplit(n_splits=ns, test_size=ts, random_state=rs)
    for train_index, test_index in sss.split(data, y):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf = RandomForestClassifier(random_state=rs, class_weight='balanced')
    rfecv = RFECV(estimator=clf, step=1, cv=sss, scoring=f1_scorer, n_jobs=-1)
    rfecv.fit(X_train, y_train)
    print("\nOptimal number of features : %d" % rfecv.n_features_) 
    sel_cols = data.columns[rfecv.support_]
    data = data[sel_cols].copy()
    # Perform boxcox
    for train_index, test_index in sss.split(data, y):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_train = X_train.copy()
    X_test = X_test.copy()
    numcol = ['Lim', 'Age']
    for feat in numcol:
        X_train.loc[:, feat], lambda_ = boxcox(X_train[feat])
        X_test.loc[:, feat] = boxcox(X_test[feat], lmbda=lambda_)
    numcol = ['BSep', 'BAug', 'BJul', 'BJun', 'BMay', 'BApr', 'PSep', 'PAug', 'PJul', 'PJun', 'PMay', 'PApr']
    for feat in numcol:
        X_train.loc[:, feat], lambda_ = boxcox(X_train[feat] + abs(min(X_train[feat])) + 1)
        X_test.loc[:, feat] = boxcox(X_test[feat] + abs(min(X_test[feat])) + 1, lmbda=lambda_)
    # Scale to (0, 1) range
    X_scaler = MinMaxScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)   
    # Train, predict LR using default parameters
    clf_C = LogisticRegression(C=1.0, class_weight='balanced', random_state=rs)
    train_predict(clf_C, X_train[:n], y_train[:n], X_test, y_test)
    lst1.append(predict_labels(clf_C, X_test, y_test))

# Initialize list of all sample test scores
lst1 = []
# Random seed sample (StratifiedShuffleSplit and all models)
rnd = [12, 111]
# Test size sample (StratifiedShuffleSplit)
tst = [0.15, 0.3]
# Number of splits sample (StratifiedShuffleSplit)
npl = [10, 30]
# Sample of train set sizes: 30%, 60% of 22500 (total X_train length)
rws = [int(len(X_train)*0.3), int(len(X_train)*0.6)]
print "\nSensitivity analysis for Random State, Test Size, SSS No. Splits and Train Set Size:"
# Get prediction scores for four sensitivity samples  
for i in rnd:
    print "\n***A. Random state = {}***".format(i)
    robustness_test(rs=i)
for i in tst:
    print "\n***B. Test size = {}***".format(i)
    robustness_test(ts=i)
for i in npl:
    print "\n***C. No. splits = {}***".format(i)
    robustness_test(ns=i)
for i in rws:
    print "\n***D. Train set size (no. rows) = {}***".format(i)
    robustness_test(n=i)

# Show summary statistics of 8 test scores
print "\nSensitivity Test Scores have mean {:.6f}, median {:.6f} and standard deviation {:.6f}.".format(np.mean(lst1), np.median(lst1), np.std(lst1))


# Include first LR's prediction in sensitivity distribution
lst1.append(LR_pred1)

# Summary statistics after adding LR's #9 prediction
print "\nLR's Test Score Distribution (9 observations) has mean {:.6f}, median {:.6f} and standard deviation {:.6f}.".format(np.mean(lst1), np.median(lst1), np.std(lst1))


#### Create distribution values for Benchmark model

def benchmark_sensitivity(ns=20, ts=0.25, rs=57, n=22500):
    # Retrieve original dataset used for benchmark
    data = data_c1.copy()
    # Split into train and test sets 
    sss = StratifiedShuffleSplit(n_splits=ns, test_size=ts, random_state=rs)
    for train_index, test_index in sss.split(data, y):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Set up Benchmark model
    clf = RandomForestClassifier(random_state=rs, class_weight='balanced')  
    # Train model
    train_classifier(clf, X_train[:n], y_train[:n])
    # Store prediction results
    lst2.append(predict_labels(clf, X_test, y_test))

# Create distribution data using same sensitivity variables for final model
for i in rnd:
    benchmark_sensitivity(rs=i)
for i in tst:
    benchmark_sensitivity(ts=i)
for i in npl:
    benchmark_sensitivity(ns=i)
for i in rws:
    benchmark_sensitivity(n=i)

print "\nRandom Forest's generated distribution data is:\n\n{}\n".format(lst2)
print "\nThe Random Forest distribution has mean {:6f}, median {:6f} and standard deviation {:6f}.\n".format(np.mean(lst2), np.median(lst2), np.std(lst2))

# Calculation of t-statistics and p-value for statistical significance
from statsmodels.stats.weightstats import ttest_ind as tt

print "\nThe test statistic is {}, the p-value is {} and the degrees of freedom are {}.\n".format(tt(lst1, lst2)[0], tt(lst1, lst2)[1], tt(lst1, lst2)[2])


#### Visualization of Logistic Regression's scores vs. C parameter values

# Initialize temp storage of results
lst3 = []
# Set range of C parameter values
C = np.arange(0.1, 1.11, 0.01)
# Train, predict best model with range of C parameters
for c in C:    
    clf = LogisticRegression(C=c, class_weight='balanced', random_state=rs)
    train_classifier(clf, X_train, y_train)
    lst3.append([c, predict_labels(clf, X_train, y_train), predict_labels(clf, X_test, y_test)])
# Save predictions and C values for plotting
cols = ['c', 'train_score', 'test_score']
df = pd.DataFrame(lst3, columns=cols)
# Plot titles and labels
plt.figure(figsize=(8,4))
plt.suptitle("Logistic Regression: Training and Testing Curves", y=1.05, fontsize=13)
plt.title("Graph 12: F1 Scores vs. C Parameter\n", fontsize=13)
plt.xlabel("C parameter values")
plt.ylabel("F1 Scores")
plt.xticks(np.arange(0.1, 1.11, 0.1))
lw = 2
# Plot both curves with legend
plt.plot(df.c, df.train_score, label="Training score",
             color="navy", lw=lw)
plt.plot(df.c, df.test_score, label="Testing score",
             color="darkorange", lw=lw)
plt.legend(loc='best', fontsize='medium')
plt.show()

# Display C and the best test score
print "The best test score and its C parameter value in {} iterations :".format(len(df))
display(df[df.test_score==max(df.test_score)])


#### Visualization of Logistic Regression's scores vs. C parameter values

# Initialize temp storage of results
lst3 = []
# Set range of C parameter values
C = np.logspace(-5,5,11) 
# Train, predict best model with range of C parameters
for c in C:    
    clf = LogisticRegression(C=c, class_weight='balanced', random_state=rs)
    train_classifier(clf, X_train, y_train)
    lst3.append([c, predict_labels(clf, X_train, y_train), predict_labels(clf, X_test, y_test)])
# Save predictions and C values for plotting
cols = ['c', 'train_score', 'test_score']
df = pd.DataFrame(lst3, columns=cols)

# Display C and the best test score
print "\nThe best test score and its C parameter value in {} iterations :".format(len(df))
display(df[df.test_score==max(df.test_score)])

