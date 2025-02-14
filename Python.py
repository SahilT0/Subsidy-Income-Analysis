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
print(data.info)

# Missing value
print("Data with null values : \n", data.isnull().sum())



