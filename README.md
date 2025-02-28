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
![Correlation Matrix](./images/correlation_matrix.png)
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
sns.heatmap(data2.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
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
![Frequency Diagram](./images/frequency_diagram.png)
```python
sns.countplot(data2["SalStat"])
plt.show()
```

### **2. Histogram of Age**
![Age Histogram](./images/histogram_age.png)
```python
sns.distplot(data2["age"], bins=10, kde=True)
plt.show()
```

### **3. Box Plot of Hours per Week vs Salary Status**
![Box Plot](./images/boxplot_hoursperweek.png)
```python
sns.boxplot(x="SalStat", y="hoursperweek", data=data2)
plt.show()
```

### **4. Histogram of Capital Gain**
![Capital Gain Histogram](./images/histogram_capitalgain.png)
```python
sns.distplot(data2["capitalgain"])
plt.show()
```

### **5. Confusion Matrix for Logistic Regression**
![Confusion Matrix](./images/confusion_matrix_logistic.png)
```python
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(test_y, prediction)
plt.show()
```

### **6. Confusion Matrix for KNN Classifier**
![Confusion Matrix KNN](./images/confusion_matrix_knn.png)
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

