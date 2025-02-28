# Personal Income Classifier

## Project Overview
This project is a **Personal Income Classifier** that predicts whether a person's income is **less than or equal to $50,000** or **greater than $50,000** using **Logistic Regression** and **KNN Classifier**.

## Dataset
- The dataset used for this project is `income1.csv`.
- It contains various attributes such as `age`, `job type`, `education`, `marital status`, `occupation`, `gender`, `capital gain`, `capital loss`, `hours per week`, and `salary status`.

## Features Used
The following features are considered for classification:
- Age
- Job Type
- Education Level
- Marital Status
- Occupation
- Relationship
- Race
- Gender
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

## Installation and Setup
To run this project, follow these steps:

### **1. Install Required Libraries**
```bash
pip install pandas numpy seaborn scikit-learn matplotlib pymysql
```

### **2. Clone the Repository**
```bash
git clone https://github.com/yourusername/Personal-Income-Classifier.git
cd Personal-Income-Classifier
```

### **3. Run the Python Script**
```bash
python income_classifier.py
```

## Data Preprocessing
- The dataset is loaded and cleaned by handling missing values.
- Categorical variables are converted into numerical using one-hot encoding.
- Features are scaled, and data is split into training and testing sets.

## Correlation Between Numerical Variables
![Correlation ](https://github.com/user-attachments/assets/27c5ed37-9e61-4173-a28a-e6a05b1b2270)

```python
correlation = data2.select_dtypes(include="number").corr()
print(correlation)
```

## Model Training
### **1. Logistic Regression**
- Used to classify income levels.
- Accuracy score: **99.8%**.

### **2. K-Nearest Neighbors (KNN)**
- Accuracy score: **93.4%**.
- Optimal K-value found by testing different values.

## Visualizations
The following visualizations help in data exploration:

### **1. Frequency Distribution of Salary Status**
![salary Status](https://github.com/user-attachments/assets/50eb283b-46ec-4333-a1fa-54352ce18aa7)

```python
sns.countplot(data2["SalStat"])
plt.show()
```

### **2. Histogram of Age**
![histogram_of_age](https://github.com/user-attachments/assets/e428d06b-0d0a-4c97-a530-1e52570562ab)

```python
sns.distplot(data2["age"], bins=10, kde=True)
plt.show()
```

### **3. Box Plot of Hours per Week vs Salary Status**
![Hour per week and salstat](https://github.com/user-attachments/assets/225604ef-28ee-48f6-8d63-881866bba1bc)

```python
sns.boxplot(x="SalStat", y="hoursperweek", data=data2)
plt.show()
```

### **4. Histogram of Capital Gain**
![histogram of capital gain](https://github.com/user-attachments/assets/2be7f3b3-63c6-46e1-9197-68bdf3759db3)

```python
sns.distplot(data2["capitalgain"])
plt.show()
```

### **5. Confusion Matrix for Logistic Regression**
![cunfusion Matrix logitic regression](https://github.com/user-attachments/assets/2cf57dd9-3b40-4850-a903-c809f25a29eb)

```python
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(test_y, prediction)
plt.show()
```

### **6. Confusion Matrix for KNN Classifier**
![confusion Matrix knn](https://github.com/user-attachments/assets/f95533a7-634a-43d2-a1c0-a42945e7d2d1)

```python
ConfusionMatrixDisplay.from_predictions(test_y, knn_prediction)
plt.show()
```

## Results and Conclusion
- **Logistic Regression** performed better than KNN with an accuracy of **99.8%**.
- Various visualizations helped in understanding the data distribution.

## Contributing
If you want to contribute, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

