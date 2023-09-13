import pandas as pd
import numpy as np
import re

from pandas.api.types import is_numeric_dtype
from scipy import stats

from model import Model
from plot import Plot


class DatasetWrapper(object):
    def __init__(self, dataset):
        self.df = dataset
        self.icd_codes = {(390, 459, 785): "Circulatory system",
                          (250, 250.99): "Diabetes",
                          (460, 519, 786): "Respiratory system",
                          (520, 579, 787): "Digestive system",
                          (800, 999): "Injury/Poisoning",
                          (710, 739): "Musculoskeletal system/Connective tissue",
                          (580, 629, 788): "Genitourinary system",
                          (140, 239): "Neoplasms",
                          (0, float("inf")): "Other"}

    @property
    def shape(self):
        return self.df.shape

    @property
    def numeric_columns(self):
        return self.df.select_dtypes(include=np.number).columns.tolist()

    @property
    def numeric_columns_without_ids(self):
        return [column for column in self.numeric_columns if not self.df[column].is_unique]

    @property
    def categorical_columns(self):
        return list(set(self.df.columns) - set(self.numeric_columns))

    def replace_value(self, replacement_map, columns=()):
        if len(columns):
            for column in columns:
                self.df[column] = self.df[column].replace(replacement_map)
        else:
            self.df = self.df.replace(replacement_map)

    def remove_rows_with_missing_values(self):
        self.df = self.df.dropna()

    def remove_rows_with_duplicate_values(self, columns=None):
        self.df = self.df.drop_duplicates(subset=columns)

    def remove_rows_with_outliers(self, threshold):
        self.df = self.df[(np.abs(stats.zscore(self.df.select_dtypes(include=np.number))) < threshold).all(axis=1)]

    def remove_columns_with_missing_values_percentage(self, threshold):
        self.df = self.df.loc[:, self.df.isnull().sum() < threshold * self.shape[0]]

    def remove_columns_with_same_values_percentage(self, threshold):
        res = self.df
        for column in self.df.columns:
            if self.df[column].value_counts().iloc[0] / self.shape[0] >= threshold:
                res = res.drop(columns=[column, ])
        self.df = res

    def transform_range_to_middle_point(self, columns, regex):
        for column in columns:
            self.df[column] = self.df[column].apply(lambda value: int(sum(map(int, re.findall(regex, value))) / 2))

    def transform_readmission_to_binary(self):
        self.df.readmitted = self.df.readmitted.map({'NO': 0, '>30': 1, "<30": 1})

    @staticmethod
    def apply_icd_code(item, key, value):
        if type(item) is str:
            return item
        elif key[0] <= item <= key[1] or len(key) == 3 and item == key[2]:
            return value
        else:
            return item

    def transform_icd_codes_to_categories(self, columns):
        for column in columns:
            self.df[column] = pd.to_numeric(self.df[column], errors="coerce").fillna(0).astype(int)
            for key, value in self.icd_codes.items():
                self.df[column] = self.df[column].apply(self.apply_icd_code, args=(key, value))

    def transform_numerical_columns_to_normalised_data(self, columns):
        for column in columns:
            self.df[column] = (self.df[column] - self.df[column].min()) / (self.df[column].max() - self.df[column].min())

    def transform_categorical_columns_to_numeric_data(self):
        for column in self.df.columns:
            if not is_numeric_dtype(self.df[column]):
                unique_keys = self.df[column].unique()
                unique_values = list(range(0, len(unique_keys)))
                unique_map = dict(zip(unique_keys, unique_values))
                self.replace_value(unique_map, columns=(column, ))


df = pd.read_csv("Data/diabetic_data.csv")
data_wrapper = DatasetWrapper(df)
diagnosis_columns = ["diag_1", "diag_2", "diag_3"]

# Shape before data cleaning
print("Shape before data cleaning " + str(data_wrapper.shape))

# Part 1 - Building up a basic predicitive model

# Replace all missing values with numpy.nan
data_wrapper.replace_value({"?": np.nan})
# Dropping all columns with more than 50% missing values
data_wrapper.remove_columns_with_missing_values_percentage(0.5)
# Dropping all columns with over 95% of the same values
data_wrapper.remove_columns_with_same_values_percentage(0.95)
# Transforming the age to be the middle value of its range
data_wrapper.transform_range_to_middle_point(["age"], "\d+")
# Replace missing values in the diag columns with 0
data_wrapper.replace_value({np.nan: 0}, columns=diagnosis_columns)
# Dropping all rows with missing values
data_wrapper.remove_rows_with_missing_values()
# List of all categorical columns
print("Categorical columns: " + str(list(data_wrapper.df.select_dtypes(include=['object']).columns)))
# List of all numerical columns
print("Numerical columns: " + str(list(data_wrapper.df.select_dtypes(include=['int64']).columns)))
# Removing outliers that are not within 3 standard deviations
data_wrapper.remove_rows_with_outliers(3)
# Removing rows that have duplicate patient_nbr values
data_wrapper.remove_rows_with_duplicate_values(columns=["patient_nbr", ])

# Shape after data cleaning
print("Shape after data cleaning " + str(data_wrapper.shape))

# Part 1 - Data exploration

# Transforming the readmission column into binary values
data_wrapper.transform_readmission_to_binary()
# Converting the icd codes using icd_codes.csv, along with grouping based on certain diagnosis
data_wrapper.transform_icd_codes_to_categories(diagnosis_columns)

# Created a new updated csv file for easier data exploration
# data_wrapper.df.to_csv("updated_data.csv")
# Created a bar charts to apprive/disprove the age and gender hypothesis
Plot.create_bar_chart_from_crosstab_of_column(data_wrapper.df, ["age", "gender", "race"])
# Plotted bar charts to highlight if a certain diag has an affect on readmission
Plot.create_bar_chart_from_crosstab_of_column(data_wrapper.df, ["diag_1", "diag_2", "diag_3"], True)

# Conversion of columns to numerial data
data_wrapper.transform_categorical_columns_to_numeric_data()
# Normalising the data before better model building
data_wrapper.transform_numerical_columns_to_normalised_data(data_wrapper.numeric_columns)
# Model.rfe(data_wrapper.df)
# Building the model based on the given columns using logistic regression
Model.logistic_regression(data_wrapper.df, ['num_medications', 'number_outpatient', 'number_emergency', 'time_in_hospital', 'number_inpatient', 'encounter_id', 'age', 'num_lab_procedures', 'number_diagnoses', 'num_procedures'])
# Building the model using random forest
Model.random_forest(data_wrapper.df, ['num_medications', 'number_outpatient', 'number_emergency', 'time_in_hospital', 'number_inpatient', 'encounter_id', 'age', 'num_lab_procedures', 'number_diagnoses', 'num_procedures'])

Plot.show_plots()
