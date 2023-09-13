import numpy as np

from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from plot import Plot


class Model(object):
    @staticmethod
    def logistic_regression(df, predictors):
        # f_columns = ['encounter_id', 'patient_nbr', 'number_emergency', 'number_inpatient', 'number_diagnoses']
        model = linear_model.LogisticRegression()
        x = df[predictors]
        y = df['readmitted']
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)
        model.fit(train_x, train_y)
        Model.accuracy(model, x, y, test_x, test_y)
        
    @staticmethod
    def random_forest(df, predictors):
        # f_columns = ['encounter_id', 'patient_nbr', 'number_emergency', 'number_inpatient', 'number_diagnoses']
        x = df[predictors]
        y = df['readmitted']
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)
        model = RandomForestClassifier(n_jobs=-1, n_estimators=500, max_depth=5)
        model.fit(train_x,train_y)
        predicted = model.predict(test_x)
        print("********" * 5)
        print("Accuracy score for training data is: {:4.3f}".format(model.score(train_x,train_y)))
        print("Accuracy score for test data: {:4.3f}".format(model.score(test_x, test_y)))
        scores = cross_val_score(RandomForestClassifier(), x, y, scoring='accuracy', cv=10)
        print("Cross validation mean scores: {}".format(scores.mean()))
        print("********" * 5)
        Plot.create_confusion_matrix(model, test_y, predicted)

    @staticmethod
    def accuracy(model, x, y, test_x, test_y):
        predicted = model.predict(test_x)
        print("********" * 5)
        print("Mean hits: {:4.3f}".format(np.mean(predicted == test_y)))
        print("Accuracy score: {:4.3f}".format(metrics.accuracy_score(test_y, predicted)))
        scores = cross_val_score(linear_model.LogisticRegression(), x, y, scoring='accuracy', cv=10)
        print("Cross validation mean scores: {}".format(scores.mean()))
        print("********" * 5)
        Plot.create_confusion_matrix(model, test_y, predicted)

    @staticmethod
    def rfe(df):
        f_cols = df.select_dtypes(include=np.number).columns.tolist()
        f_cols.remove('readmitted')
        x = df[f_cols]
        y = df['readmitted']
        estimator = linear_model.LogisticRegression()
        selector = RFE(estimator, n_features_to_select=5, step=1)
        selector = selector.fit(x, y)
        select_features = np.array(f_cols)[selector.ranking_ == 1].tolist()
        print(select_features)
