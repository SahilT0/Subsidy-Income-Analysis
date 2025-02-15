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

# For viewing
import matplotlib.pyplot as plt


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

# Relationship between Independent Variables
correlation = data2.select_dtypes(include="number").corr()
# print(correlation)

# Cross Table & Data visualization
# print(data2.columns)

# Gender proportion table
gender = pd.crosstab(index= data2["gender"],
                     columns="count",
                     normalize=True)
# print(gender)

# Education Type proportion table
EdType = pd.crosstab(index= data2["EdType"],
                     columns="count",
                     normalize=True)
# print(EdType)

# Marital Status proportion table
maritalsta = pd.crosstab(index=data2["maritalstatus"],
                         columns="count",
                         normalize=True)
# print(maritalsta)

# Occupation proportion table
occupation = pd.crosstab(index=data2["occupation"],
                         columns="count",
                         normalize=True)
# print(occupation)

# Relationship proportion table
relationship = pd.crosstab(index=data2["relationship"],
                           columns="count",
                           normalize=True)
# print(relationship)

# Race proportion tables
race = pd.crosstab(index=data2["race"],
                   columns="count",
                   normalize=True)
# print(race)

# Nativecountry Propotion table
nativeco = pd.crosstab(index=data2["nativecountry"],
                       columns="count",
                       normalize=True,
                       )
nativeco = nativeco.sort_values(by="count", ascending=True)
# print(nativeco)

# Gender vs Salary Status
gen_salst = pd.crosstab(index=data2["gender"],
                        columns=data2["SalStat"],
                        margins=True,
                        normalize="index")
# print(gen_salst)

# Jobtype vs Salary status
job_sal = pd.crosstab(index=data2["JobType"],
                        columns=data2["SalStat"],
                        margins=True,
                        normalize="index")
# print(job_sal)

# Education type vs Salary status
ed_sal = pd.crosstab(index=data2["EdType"],
                        columns=data2["SalStat"],
                        margins=True,
                        normalize="index")
# print(ed_sal)

# Marital Status vs Salary Status
mart_sal = pd.crosstab(index=data2["maritalstatus"],
                        columns=data2["SalStat"],
                        margins=True,
                        normalize="index")
# print(mart_sal)

# Occupation vs Salary status
occ_sal = pd.crosstab(index=data2["occupation"],
                        columns=data2["SalStat"],
                        margins=True,
                        normalize="index")
# print(occ_sal)

# Relationship vs Salary status
rel_sal = pd.crosstab(index=data2["relationship"],
                        columns=data2["SalStat"],
                        margins=True,
                        normalize="index")
# print(rel_sal)

# Race vs Salary status
race_sal = pd.crosstab(index=data2["race"],
                        columns=data2["SalStat"],
                        margins=True,
                        normalize="index")
# print(race_sal)

# Native Country vs Salary Status
nat_sal = pd.crosstab(index=data2["nativecountry"],
                        columns=data2["SalStat"],
                        margins=True,
                        normalize="index")
# print(nat_sal)

# Frequency Distribution of Salary Status
# SalStat = sns.countplot(data2["SalStat"])
# plt.show()

# Histogram of Age
# sns.distplot(data2["age"], bins=10, kde=True)
# plt.show()

# Box Plot - Age vs Salary Status
# sns.boxplot(x='SalStat',y="age", data=data2)
# plt.show()

# Bar plot of JobType vs Salstat
# sns.countplot(y="JobType", hue="SalStat", data=data2, order=data2["JobType"].value_counts().index)
# plt.show()

# Bar plot of Education vs Salstat
# sns.countplot(y="EdType", hue="SalStat", data=data2, order=data2["EdType"].value_counts().index)
# plt.show()

# Bar plot Occupation vs Salstat
# sns.countplot(y="occupation", hue="SalStat", data=data2, order=data2["occupation"].value_counts().index)
# plt.show()

# Histogram of Capital gain
# sns.distplot(data2["capitalgain"])
# plt.show()
#
# # Histogram of Capital gain
# sns.distplot(data2["capitalgain"])
# plt.show()

# Box plot of Capital gain and loss for outliers detection
# sns.boxplot(data2["capitalgain"], color="blue")
# plt.show()
#
# sns.boxplot(data2["capitalloss"], color="red")
# plt.show()

# Box plot for hours per week vs salary status
# sns.boxplot(x="SalStat", y="hoursperweek", data=data2)
# plt.show()


# ************** LOGISTIC REGRESSION ***********

# Reindexing the salary status to 0 or 1
# data2["SalStat"] = data2["SalStat"].map({" less than or equal to 50,000":0, " greater than 50,000":1})
# print(data2["SalStat"])

# Converting categorical data into numerical by get dummies function
new_data = pd.get_dummies(data2, drop_first=True)

# Storing the columns names
column_list = list(new_data.columns)
# print(column_list)


# Seprate the input names from data
features = list(set(column_list)-set(["SalStat"]))
# print(features)

# Store the output values in y
y = new_data["SalStat_ less than or equal to 50,000"].values
# print(y)

# Store the value from input features
x = new_data[features].values
# print(x)

# Separating the data into train and test
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)

# Make an instance of logistic regression
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)
# print(logistic.coef_)
# print(logistic.intercept_)

# Prediction from test data
prediction = logistic.predict(test_x)
# print(prediction)

# Confusion Matrix
confusion_mat = confusion_matrix(test_y, prediction)
# print(confusion_mat)

# Calculating the accuracy
accuracy_sco = accuracy_score(test_y, prediction)
# print(accuracy_sco)

# Printing the misclassified values from prediction
# print("Misclassified samples :- ", (test_y != prediction).sum())


