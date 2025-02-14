# Personal Income Classifier

# To work with dataframes
import pandas as pd

# To perform numerical operations
import numpy as np

# To visualize data
import seaborn as sns

# To partition the data
from sklearn.model_selection import train_test_split

# Importing Libraries for Logistic regression
from sklearn.linear_model import LogisticRegression

# Importing Performance matrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

# For connecting with database
import pymysql as m


# Importing data
data_income = pd.read_csv("C:/Users/SAHIL/Data Science Projects/Subsidy Income/Dataset/income1.csv")

# Creating Copy of Original data
data = data_income.copy()

# Know the data
# print(data.info)

# Missing value
# print("Data with null values : \n", data.isnull().sum())

# Summary of numerical variables
summary_num = data.describe()
# print(summary_num)

# Summary of Categorical variables
pd.set_option("display.max_columns", None) # For all columns display in pycharm
summary_cat = data.describe(include="object")
# print(summary_cat)

# Frequency of each category
pd.set_option("display.max_rows", None)
# print(data.columns)
# print(data["age"].value_counts())
# print(data["JobType"].value_counts())
# print(data["EdType"].value_counts())
# print(data["maritalstatus"].value_counts())
# print(data["occupation"].value_counts())
# print(data["relationship"].value_counts())
# print(data["race"].value_counts())
# print(data["gender"].value_counts())
# print(data["capitalgain"].value_counts())
# print(data["capitalloss"].value_counts())
# print(data["hoursperweek"].value_counts())
# print(data["nativecountry"].value_counts())
# print(data["SalStat"].value_counts())

# Checking for unique Classes
# print(np.unique(data["JobType"]))
# print(np.unique(data["occupation"]))

data = pd.read_csv("C:/Users/SAHIL/Data Science Projects/Subsidy Income/Dataset/income1.csv", na_values=" ?")

# data Preprocessing
# print(data.isnull().sum())

# Missing data
missing = data[data.isnull().any(axis=1)]
# print(missing)

# Deleting the missing data
data2 = data.dropna(axis=0)
# print(data2)

