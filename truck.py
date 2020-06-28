#*******************#
#   ATA GÜNDÜZALP   #
#*******************#

import pandas as pd
import numpy as np
import math, time, random, datetime
from pandas.api.types import is_numeric_dtype
from pandas.core.dtypes.missing import isna
import missingno as missingno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_regression, f_classif
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score 
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

def changed_neg_pos_to_int(data):
    data.loc[data['class'] == "neg", 'class'] = 0
    data.loc[data['class'] == "pos", 'class'] = 1
    return data

def decide_drop_and_normalized_columns(data):
    cols_to_drop = []
    cols_to_norm = []
    for column in data.columns[1:]:
        data[column] = pd.to_numeric(data[column],errors='coerce')
        total_na = sum(isna(data[column]))
        na_rate = total_na / 60000
        if na_rate > 0.5:
            cols_to_drop.append(column)
        else:
            cols_to_norm.append(column)
    return data, cols_to_norm, cols_to_drop
    

def feature_engineering(data, cols_to_norm, cols_to_drop):
    positive_df = data[(data['class'] == 1)]
    negative_df = data[(data['class'] == 0)]
    
    na_rate_per_column = 0
    for column in positive_df.columns[1:]:
        positive_df[column] = pd.to_numeric(positive_df[column],errors='coerce')
        total_na = sum(isna(positive_df[column]))
        if total_na > 0:
            median = positive_df[column].median(skipna = True)
            positive_df[column].fillna(median, inplace=True)
        positive_df[column].replace([np.inf],max(positive_df[column].replace(np.inf, np.nan)),inplace=True)

    for column in negative_df.columns[1:]:
        negative_df[column] = pd.to_numeric(negative_df[column],errors='coerce')
        total_na = sum(isna(negative_df[column]))
        if total_na > 0:
            median = negative_df[column].median(skipna = True)
            negative_df[column].fillna(median, inplace=True)
        negative_df[column].replace([np.inf],max(negative_df[column].replace(np.inf, np.nan)),inplace=True)
    
    merged_df = merge_data_frames(negative_df, positive_df)
    normalized_df = normalize(merged_df, cols_to_norm)
    print(normalized_df)
    normalized_positive_df = normalized_df[(normalized_df['class'] == 1)]
    normalized_negative_df = normalized_df[(normalized_df['class'] == 0)]
    return normalized_df, normalized_positive_df, normalized_negative_df

def merge_data_frames(negative_df, positive_df):
    data_frames = [negative_df, positive_df]
    merged_df = pd.concat(data_frames)
    return merged_df

def normalize(df, cols_to_norm):
    result = df.copy()
    for feature_name in df.columns[1:]:
        if feature_name in cols_to_norm:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            if min_value != max_value:
                result[feature_name] = (df[feature_name] - df[feature_name].mean()) / (df[feature_name].max() - df[feature_name].min())
            else:
                print("equal")
    return result

def correlation(data, cols_to_drop):
    corr_matrix = data.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    for col in cols_to_drop:
        if col not in to_drop:
            to_drop.append(col)
    correlated_df = data[to_drop]
    
    # Drop features 
    #get correlations of each features in dataset
    corrmat = correlated_df.corr()
    top_corr_features = corrmat.index
    data.drop(to_drop, axis=1, inplace=True)
    return data, to_drop

# FEATURE SELECTION
def best_feature_selection(data):
    X = data.iloc[:,1:]
    y = data.iloc[:,0:1]
    y=y.astype('int')
    
    # SELECT BEST 
    # Statistical tests can be used to select those features that 
    # have the strongest relationship with the output variable.

    bestfeatures = SelectKBest(score_func=f_classif, k=90)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(90,'Score'))  #print 10 best features

    best_features = []
    for feature in featureScores.nlargest(90,'Score')['Specs']:
        best_features.append(feature)
    return best_features


# FEATURE IMPORTANCE
# Logistic Regression Feature Importance
def logistic_regression_feature_importance(data):
    X = data.iloc[:,1:]
    y = data.iloc[:,0:1]
    y=y.astype('int')
    model = LogisticRegression()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.coef_[0]
    # summarize feature importance
    feature_importance = []
    index_score_dict = {}
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()
    for i,v in enumerate(importance):
        print('Feature: ' + str(data.columns[i+1]) + ', Score: ' + str(v))
        index_score_dict[i] = v
    sorted_index_score_dict = {}
    for value in sorted(index_score_dict.values()):
        key = list(index_score_dict.keys())[list(index_score_dict.values()).index(value)]
        sorted_index_score_dict[key] = index_score_dict[key]
    print("sorted: " + str(sorted_index_score_dict))

def extraTreesClassifier(data):
    X = data.iloc[:,1:]
    y = data.iloc[:,0:1]
    y=y.astype('int')
    from sklearn.ensemble import ExtraTreesClassifier
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()

# MODEL CREATION
def create_model_best_feature_with_positive_negative(train_normalized_positive_df, train_normalized_negative_df, test_normalized_positive_df, 
                                                    test_normalized_negative_df, best_positive_train_data_frame, best_negative_train_data_frame, best_positive_test_data_frame, 
                                                    best_negative_test_data_frame):
    from sklearn.metrics import confusion_matrix
    best_train_negative_df = best_negative_train_data_frame.sample(n = 59000)
    train_merged_df = merge_data_frames(best_train_negative_df, best_positive_train_data_frame)
    X = train_merged_df.iloc[:,0:]
    
    train_y_negative = train_normalized_negative_df.sample(n=59000)
    y_merged = merge_data_frames(train_y_negative, train_normalized_positive_df)
    y = y_merged.iloc[:,0:1]
    y = y.astype('int')

    # Random Forest Classifier
    model = random_forest(X, y)
    test_merged_df = merge_data_frames(best_negative_test_data_frame, best_positive_test_data_frame)
    X_test = test_merged_df.iloc[:,0:]

    test_y_merged = merge_data_frames(test_normalized_negative_df, test_normalized_positive_df)
    y_test = test_y_merged.iloc[:,0:1]
    y_test = y_test.astype('int')

    predicted_classes = model.predict(X_test)
    confusion_matrix = confusion_matrix(y_test, predicted_classes)
    show_confusion_matrix(confusion_matrix)
   
    print(confusion_matrix)
    print_result_individuals(confusion_matrix)
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    print(classification_report(y_test, predicted_classes))
    print(accuracy_score(y_test, predicted_classes))
    print("\n")

def logistic_regression(X, y):
    weights = {0:10.0, 1:590.0}
    model = LogisticRegression(solver='lbfgs', class_weight=weights)
    return model.fit(X, y)
    
def random_forest(X, y):
    weights = {0:10.0, 1:590.0}
    model = RandomForestClassifier(n_estimators= 1330, min_samples_split= 100, min_samples_leaf= 2, max_features= 'auto', 
                                    max_depth= 50, criterion= 'entropy', bootstrap= False, class_weight=weights)
    return model.fit(X, y)

def SVM(X, y):
    from sklearn.svm import SVC
    weights = {0:10.0, 1:590.0}
    model = SVC(class_weight=weights, C = 10.0, kernel= 'poly', gamma = 'scale' )
    return model.fit(X, y)

def ridge_classifier(X, y):
    weights = {0:10.0, 1:590.0}
    model = RidgeClassifier(alpha=5.0, class_weight=weights)
    return model.fit(X, y)

def show_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(confusion_matrix)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='red')
    plt.show()

def print_result_individuals(cm):
    print('True positive = ', cm[0][0])
    print('False positive = ', cm[1][0])
    print('False negative = ', cm[0][1])
    print('True negative = ', cm[1][1])

'''
# PCA
def apply_pca():
    X_train = merged_df.drop(['class'], axis=1)
    y_train = merged_df['class']
    X_scaled = StandardScaler().fit_transform(X_train)

    pca = decomposition.PCA().fit(X_scaled)

    plt.figure(figsize=(10,7))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
    plt.xlabel('Number of components')
    plt.ylabel('Total explained variance')
    plt.xlim(0, 100)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.axvline(8, c='b') # Tune this so that you obtain at least a 95% total variance explained
    plt.axhline(0.95, c='r')
    plt.show();

    pca = decomposition.PCA(n_components=20)
    X_pca = pca.fit_transform(X_scaled)
'''

'''
def select_hyperparameters(X, y):
    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 300, stop = 1500, num = 100)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
    # criterion{“gini”, “entropy”}, default=”gini”
    criterion = ['gini', 'entropy']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'criterion': criterion,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    weights = {0:10.0, 1:500.0}
    rf = RandomForestClassifier(class_weight=weights)
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=8, n_jobs = -1)
    rf_random.fit(X, y)
    best_params = rf_random.best_params_
    print("best results: " + str(rf_random.best_params_))
    
    # best results: {'n_estimators': 1330, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 50, 'criterion': 'entropy', 'bootstrap': False}
'''

# If desired to use SMOTE to data over-sampling
'''
def mock_data(X, y):
    from imblearn.over_sampling import SMOTE
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    print(X)
    print(y)
    return X, y
'''

if __name__ == "__main__":
    data = pd.read_csv("aps_failure_training_set.csv")
    test_data = pd.read_csv("aps_failure_test_set.csv")
    
    data = changed_neg_pos_to_int(data)
    
    train_data_frame, cols_to_norm, cols_to_drop = decide_drop_and_normalized_columns(data)
    train_data_frame, to_drop = correlation(train_data_frame, cols_to_drop)
    train_merged_df, train_normalized_positive_df, train_normalized_negative_df = feature_engineering(train_data_frame, cols_to_norm, cols_to_drop)
    
    test_data = changed_neg_pos_to_int(test_data)
    test_data.drop(to_drop, axis=1, inplace=True)
    test_merged_df, test_normalized_positive_df, test_normalized_negative_df = feature_engineering(test_data, cols_to_norm, cols_to_drop)

    best_featrues_list = best_feature_selection(train_merged_df)

    best_feature_train_positive_df = train_normalized_positive_df[best_featrues_list]
    best_feature_train_negative_df = train_normalized_negative_df[best_featrues_list]
    best_feature_train_df = train_merged_df[best_featrues_list]

    best_feature_test_positive_df = test_normalized_positive_df[best_featrues_list]
    best_feature_test_negative_df = test_normalized_negative_df[best_featrues_list]
    best_feature_test_df = test_merged_df[best_featrues_list]
    
    create_model_best_feature_with_positive_negative(train_normalized_positive_df, train_normalized_negative_df, test_normalized_positive_df, 
                                                    test_normalized_negative_df, best_feature_train_positive_df, best_feature_train_negative_df, 
                                                    best_feature_test_positive_df, best_feature_test_negative_df)
