# 🛳️ Titanic Survival Prediction

This project is a machine learning experiment built from my own understanding and effort, aimed at solving the Titanic: Machine Learning from Disaster challenge on [Kaggle](https://www.kaggle.com/competitions/titanic). 

Using Python, scikit-learn, and pandas, I trained a classification model to predict whether a passenger survived based on features like age, fare, sex, and passenger class. The project walks through the core steps of the machine learning pipeline — from data preprocessing and transformation to model training and evaluation.


## 📁 Dataset

The dataset is provided by Kaggle and consists of:

- `train.csv`: Labeled data (features + survival status)
- `test.csv`: Unlabeled test set used for prediction and submission
- Features include:
  - PassengerId, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
  - Target: `Survived` (0 = No, 1 = Yes)

---

## 🔍 Project Workflow

### 1. Data Cleaning
- Removed features with too many missing values (`Cabin`)
- Identified and handled missing values using `IterativeImputer` for numerical features and `SimpleImputer` for categorical features


### 3. Preprocessing Pipelines
- Created separate pipelines for:
  - **Numerical Features**: Imputation → Scaling → PolynomialFeatures
  - **Categorical Features**: Imputation → OneHotEncoding

### 4. Model Training
- Used `LogisticRegression` as the core model wrapped in a pipeline
- Performed hyperparameter tuning with `GridSearchCV` on:
  - Polynomial degree
  - Regularization strength (`C`)

### 5. Evaluation
- Evaluated model using:
  - Cross-validation accuracy
  - Accuracy score on training data
  - Classification report (precision, recall, F1)


Model performance may vary slightly depending on hyperparameter grid.


## 📦 Tools Used

- Python
- Pandas & NumPy
- Scikit-learn
- Visual Studio Code
- Matplotlib / Seaborn (for optional visualizations)

---

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/Maxwell-Selassie/Titanic-ML
cd Titanic-ML
