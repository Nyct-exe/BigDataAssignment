import numpy as np
import pandas as pd
import sklearn.tree as tree
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from error import error_handler
from plot import Plot


class Model(object):
    @staticmethod
    @error_handler
    def balance_dataset(df):
        smote = SMOTE(random_state=1)
        smote_input = df.drop(["readmitted"], axis=1)
        smote_output = df["readmitted"]
        x, y = smote.fit_resample(smote_input, smote_output)
        df = pd.DataFrame(x, columns=list(x.columns))
        df["readmitted"] = y
        return df

    @staticmethod
    @error_handler
    def logistic_regression(df, features):
        f_columns = features
        model = linear_model.LogisticRegression(max_iter=500, n_jobs=-1)
        x = df[f_columns]
        y = df['readmitted']
        # Splits Data Into Training and Testing Sets
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)
        model.fit(train_x, train_y)
        predicted = model.predict(test_x)
        print("********" * 5)
        print("------- Logistic Regression -------")
        print("Accuracy score for training data is: {:4.3f}".format(model.score(train_x, train_y)))
        print("Accuracy score: {:4.3f}".format(metrics.accuracy_score(test_y, predicted)))
        scores = cross_val_score(linear_model.LogisticRegression(), x, y, scoring='accuracy', cv=10)
        print("Cross validation mean scores: {}".format(scores.mean()))
        print("********" * 5)
        Plot.create_confusion_matrix(model, test_y, predicted)

    @staticmethod
    @error_handler
    def decision_tree_classifier(df):
        x = df.drop(["readmitted"], axis=1)
        y = df["readmitted"]
        # Splits Data Into Training and Testing Sets
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20, random_state=0)
        model = DecisionTreeClassifier(max_depth=8, criterion="entropy", min_samples_split=10)
        plt.figure(figsize=(12, 12))
        model.fit(train_x, train_y)
        print(x["discharge_disposition_id"])
        tree.plot_tree(model, max_depth=3, fontsize=10, feature_names=x.columns.tolist(), class_names=["0", "1"])
        plt.show()
        predicted = model.predict(test_x)
        print("********" * 5)
        print("------- Decison Tree Classifier -------")
        print("Accuracy score for training data is: {:4.3f}".format(model.score(train_x, train_y)))
        print("Accuracy score: {:4.3f}".format(metrics.accuracy_score(test_y, predicted)))
        scores = cross_val_score(DecisionTreeClassifier(max_depth=28, criterion="entropy", min_samples_split=10), x, y,
                                 scoring='accuracy', cv=10)
        print("Cross validation mean scores: {}".format(scores.mean()))
        print("********" * 5)
        Plot.create_confusion_matrix(model, test_y, predicted)

    @staticmethod
    @error_handler
    def random_forest(df, features):
        x = df[features]
        y = df['readmitted']
        # Splits Data Into Training and Testing Sets
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)
        model = RandomForestClassifier(n_jobs=-1, n_estimators=500, max_depth=5)
        model.fit(train_x, train_y)
        predicted = model.predict(test_x)
        print("********" * 5)
        print("------- Random Forest -------")
        print("Accuracy score for training data is: {:4.3f}".format(model.score(train_x, train_y)))
        print("Accuracy score for test data: {:4.3f}".format(model.score(test_x, test_y)))
        scores = cross_val_score(RandomForestClassifier(), x, y, scoring='accuracy', cv=10)
        print("Cross validation mean scores: {}".format(scores.mean()))
        print("********" * 5)
        Plot.create_confusion_matrix(model, test_y, predicted)

    @staticmethod
    @error_handler
    def sfs(df):
        # Gets all numerical columns
        f_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Removes readmitted to avoid comparing it with itself
        f_cols.remove('readmitted')
        x = df[f_cols]
        y = df['readmitted']
        estimator = linear_model.LogisticRegression(max_iter=500)
        estimator.fit(x, y)
        coefficients = estimator.coef_
        # sets a selector to find 5 features using SequentialFeatureSelector which goes over the dataset provided
        # and adds or removes fetures based on cross validation
        selector = SequentialFeatureSelector(estimator, n_features_to_select=5)
        selector.fit(x, y)
        selector = selector.get_support()
        df = df.select_dtypes(include=np.number)
        df = df.drop(["readmitted", ], axis=1)
        select_features = df.loc[:, selector].columns.tolist()
        print(select_features)
        return select_features, coefficients
