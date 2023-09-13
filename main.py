import re

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats

from error import error_handler
from model import Model
from plot import Plot


@error_handler
def read_file(path):
    return pd.read_csv(path)


class DatasetWrapper(object):
    def __init__(self, dataset):
        self._df = dataset

    @property
    @error_handler
    def df(self):
        return self._df

    @df.setter
    @error_handler
    def df(self, dataset):
        self._df = dataset

    @property
    @error_handler
    def shape(self):
        return self.df.shape

    @property
    @error_handler
    def numeric_columns(self):
        return self.df.select_dtypes(include=np.number).columns.tolist()

    @property
    @error_handler
    def numeric_columns_without_ids(self):
        return [column for column in self.numeric_columns if not self.df[column].is_unique]

    @property
    @error_handler
    def categorical_columns(self):
        return list(set(self.df.columns) - set(self.numeric_columns))

    @error_handler
    def drop_columns(self, columns=()):
        self.df = self.df.drop(columns=columns)

    @error_handler
    def replace_value(self, replacement_map, columns=(), forward_fill=False):
        if len(columns):
            for column in columns:
                self.df[column] = self.df[column].replace(replacement_map)
                self.df[column] = self.df[column].ffill() if forward_fill else self.df[column]
        else:
            self.df = self.df.replace(replacement_map)
            self.df = self.df.ffill() if forward_fill else self.df

    @error_handler
    def remove_rows_with_missing_values(self):
        self.df = self.df.dropna()

    @error_handler
    def remove_rows_with_duplicate_values(self, columns=None):
        self.df = self.df.drop_duplicates(subset=columns)

    @error_handler
    def remove_rows_with_outliers(self, threshold):
        self.df = self.df[(np.abs(stats.zscore(self.df.select_dtypes(include=np.number))) < threshold).all(axis=1)]

    @error_handler
    def remove_columns_with_missing_values_percentage(self, threshold):
        self.df = self.df.loc[:, self.df.isnull().sum() < threshold * self.shape[0]]

    @error_handler
    def remove_columns_with_same_values_percentage(self, threshold):
        res = self.df
        for column in self.df.columns:
            if self.df[column].value_counts().iloc[0] / self.shape[0] >= threshold:
                res = res.drop(columns=[column, ])
        self.df = res

    @error_handler
    def transform_range_to_middle_point(self, columns, regex):
        for column in columns:
            self.df[column] = self.df[column].apply(lambda value: int(sum(map(int, re.findall(regex, value))) / 2))

    @error_handler
    def transform_readmission_to_binary(self):
        self.df.readmitted = self.df.readmitted.map({'NO': 0, '<30': 1, '>30': 0})

    @staticmethod
    @error_handler
    def apply_icd_code(item, key, value):
        if type(item) is str:
            return item
        elif key[0] <= item <= key[1] or len(key) == 3 and item == key[2]:
            return value
        else:
            return item

    @error_handler
    def transform_icd_codes_to_categories(self, icd_codes, columns=()):
        for column in columns:
            self.df[column] = pd.to_numeric(self.df[column], errors="coerce").fillna(0).astype(int)
            for key, value in icd_codes.items():
                self.df[column] = self.df[column].apply(self.apply_icd_code, args=(key, value))

    @error_handler
    def transform_numerical_columns_to_normalised_data(self, columns):
        for column in columns:
            self.df[column] = (self.df[column] - self.df[column].min()) / (
                    self.df[column].max() - self.df[column].min())

    @error_handler
    def transform_categorical_columns_to_numeric_data(self):
        for column in self.df.columns:
            if not is_numeric_dtype(self.df[column]):
                dummies = pd.get_dummies(self.df[column], prefix=column)
                self.df = pd.concat([self.df, dummies], axis='columns')
                self.df = self.df.drop([column, ], axis=1)

    @error_handler
    def thirty_days_switch_removal(self, condition):
        if condition:
            data_wrapper.replace_value({">30": np.nan}, columns=["readmitted", ])
            data_wrapper.remove_rows_with_missing_values()
            data_wrapper.transform_readmission_to_binary()
        else:
            self.df.readmitted = self.df.readmitted.map({'NO': 0, '<30': 1, '>30': 1})


irrelevant_columns = ['encounter_id', 'payer_code', 'number_emergency', 'number_outpatient', 'medical_specialty']
diagnosis_columns = ["diag_1", "diag_2", "diag_3"]

unknown_gender_map = {"Unknown/Invalid": np.nan}
dead_patient_map = {11: np.nan, 18: np.nan, 19: np.nan, 20: np.nan, 21: np.nan}
icd_map = {(390, 459, 785): "Circulatory system",
           (250, 250.99): "Diabetes",
           (460, 519, 786): "Respiratory system",
           (520, 579, 787): "Digestive system",
           (800, 999): "Injury/Poisoning",
           (710, 739): "Musculoskeletal system/Connective tissue",
           (580, 629, 788): "Genitourinary system",
           (140, 239): "Neoplasms",
           (0, float("inf")): "Other"}

df = read_file("Data/diabetic_data.csv")
data_wrapper = DatasetWrapper(df)
data_wrapper.drop_columns(irrelevant_columns)

data_wrapper.replace_value({"?": np.nan})
data_wrapper.remove_columns_with_missing_values_percentage(0.55)
data_wrapper.remove_columns_with_same_values_percentage(0.95)

data_wrapper.transform_range_to_middle_point(["age"], "\d+")

data_wrapper.replace_value({np.nan: 0}, columns=diagnosis_columns)
data_wrapper.replace_value({6: np.nan, 5: np.nan, 8: np.nan}, columns=["admission_type_id"], forward_fill=True)
data_wrapper.replace_value(unknown_gender_map, columns=["gender"])
data_wrapper.replace_value(dead_patient_map, columns=["discharge_disposition_id", ])
data_wrapper.remove_rows_with_duplicate_values(columns=["patient_nbr", ])
data_wrapper.drop_columns(["patient_nbr", ])
data_wrapper.remove_rows_with_missing_values()
data_wrapper.remove_rows_with_outliers(3)
# True to remove People who have been readmitted after 30 days
data_wrapper.thirty_days_switch_removal(True)
data_wrapper.transform_icd_codes_to_categories(icd_map, diagnosis_columns)
Plot.create_bar_chart_from_crosstab_of_column(data_wrapper.df, ["age", "gender"])
Plot.create_bar_chart_from_crosstab_of_column(data_wrapper.df, diagnosis_columns, True)
data_wrapper.transform_categorical_columns_to_numeric_data()
data_wrapper.transform_numerical_columns_to_normalised_data(data_wrapper.numeric_columns)
data_wrapper.df = Model.balance_dataset(data_wrapper.df)
data_wrapper.df.to_csv("updated_data.csv")
# Finds the most impactful columns for predicting the accuracy
select_features, coefficients = Model.sfs(data_wrapper.df)
# Disabled due to causing issues with elbows. Use separately from elbow functions Testing
Plot.create_coefficient_chart(data_wrapper.df, coefficients)
# Computes the sum of squared distances from each point to its assigned center
Plot.elbow(data_wrapper.df)
# score computes the ratio of dispersion between and within clusters
Plot.elbow(data_wrapper.df, 'calinski_harabasz')
# Score calculates the mean Silhouette Coefficient of all samples
# This one takes the longest to compute, if is not needed disable.
Plot.elbow(data_wrapper.df, 'silhouette')
# Draws a graph of clusters and a histogram containing frequency of those clusters
Plot.cluster_graphs(data_wrapper.df)
# Model Creation with 3 different models
Model.logistic_regression(data_wrapper.df, select_features)
Model.random_forest(data_wrapper.df, select_features)
Model.decision_tree_classifier(data_wrapper.df)
Plot.show_plots()
