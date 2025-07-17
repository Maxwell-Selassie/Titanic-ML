import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PolynomialFeatures
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay
# from scipy import skew, kurtosis
# import seaborn as sns
import matplotlib.pyplot as plt
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_ids = test_df['PassengerId']

missing = train_df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_percent = (missing / len(train_df)) * 100
missing_percent = pd.DataFrame({'missing' : missing , 'missing_percent' : missing_percent})

train_df.drop(columns=['Cabin'],inplace=True,errors='ignore')
test_df.drop(columns=['Cabin'], inplace=True, errors='ignore')

train_features = train_df.drop(['Survived'],axis=1)
train_labels = train_df['Survived']
test_features = test_df.copy()

num_features = train_features.select_dtypes(include=[np.number]).columns.tolist()
cat_features = train_features.select_dtypes(include='object').columns.tolist()

num_transformer = Pipeline(steps=[
    ('Imputer',IterativeImputer(random_state=42)),
    ('Scaler',StandardScaler()),
    ('poly',PolynomialFeatures(include_bias=False))
])
cat_transformer = Pipeline(steps=[
    ('Imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])
transfomer = ColumnTransformer(transformers=[
    ('num',num_transformer,num_features),
    ('cat',cat_transformer,cat_features)
])
model = Pipeline(steps=[
    ('transformer',transfomer),
    ('model',LogisticRegression(penalty='l2',solver='liblinear',random_state=42,verbose=1, n_jobs=-1, class_weight='balanced'))
])
grid_params = ({
    'transformer__num__poly__degree' : [1,2],
    'model__C' : [0.03,0.3,3.0,30.0,300.0],
    'model__max_iter' : [10000],
})
grid = GridSearchCV(model,grid_params,cv=5,scoring='accuracy')
grid.fit(train_features,train_labels)

print('Best Polynomial Degree : ',grid.best_params_['transformer__num__poly__degree'])
print('Best regularization strength : ',grid.best_params_['model__C'])
print('Best cross validation MSE : ', grid.best_score_)
test_pred = grid.predict(test_features)

train_pred = grid.predict(train_features)
print('Accuracy Score : ', accuracy_score(train_labels,train_pred))
print('Classification Report : ',classification_report(train_labels,train_pred))


t_submission = pd.DataFrame(
    {
        'PassengerId' : test_ids,
        'Survived' : test_pred.astype(int)
    }
)
t_submission.to_csv('titanic_submission.csv', index=False)
# ConfusionMatrixDisplay.from_predictions(train_labels, train_pred, display_labels=["Not Survived", "Survived"])
# plt.title('Confusion Matrix - Training Data')
# plt.grid(False)
# plt.show()


