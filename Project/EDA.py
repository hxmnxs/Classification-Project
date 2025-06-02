import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import plotly.express as px
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import loguniform
import statsmodels.formula.api as smf
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFECV
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Dense, Dropout
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
pd.set_option('display.max_columns', None)
data_df = pd.read_csv("churn.csv")
data_df.columns = data_df.columns.str.strip()
def dataoveriew(df, message):
    print(f'{message}:')
    print('Number of rows: ', df.shape[0])
    print("Number of features:", df.shape[1])
    print("Data Features:")
    print(df.columns.tolist())
    print("Missing values:", df.isnull().sum().values.sum())
    print("Unique values:")
    print(df.nunique())
dataoveriew(data_df, 'Overview of the dataset')
target_instance = data_df["Churn"].value_counts().reset_index()
target_instance.columns = ['Category', 'Count']
fig = px.pie(
    target_instance,
    values='Count',
    names='Category',
    color='Category',
    color_discrete_sequence=["#FFFF99", "#FFF44F"],
    color_discrete_map={"No": "#E30B5C", "Yes": "#50C878"},
    title='Distribution of Churn'
)
fig.show()
data_df['TotalCharges'] = pd.to_numeric(data_df['TotalCharges'], errors='coerce')
data_df['TotalCharges'] = data_df['TotalCharges'].fillna(data_df['TotalCharges'].median())
data_df['SeniorCitizen'] = data_df['SeniorCitizen'].astype(str).replace({'0': 'No', '1': 'Yes'})
data_df.drop(["customerID"], axis=1, inplace=True)
def binary_map(feature):
    return feature.map({'Yes': 1, 'No': 0})
binary_list = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
data_df[binary_list] = data_df[binary_list].apply(binary_map)
data_df['gender'] = data_df['gender'].map({'Male': 1, 'Female': 0})
data_df['Churn'] = data_df['Churn'].map({'Yes': 1, 'No': 0})
data_df = pd.get_dummies(data_df, drop_first=True)
X = data_df.drop('Churn', axis=1)
y = data_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('rfecv', RFECV(estimator=LogisticRegression(), cv=StratifiedKFold(10, shuffle=True, random_state=RANDOM_STATE), scoring='accuracy')),
    ('model', LogisticRegression(class_weight='balanced'))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Model Evaluation")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
joblib.dump(pipeline, 'churn_pipeline.sav')
X_train_ann = X_train.copy()
X_test_ann = X_test.copy()
scaler_ann = MinMaxScaler()
X_train_ann[cols_to_scale] = scaler_ann.fit_transform(X_train_ann[cols_to_scale])
X_test_ann[cols_to_scale] = scaler_ann.transform(X_test_ann[cols_to_scale])
model_ann = Sequential()
model_ann.add(Dense(64, input_dim=X_train_ann.shape[1], activation='relu'))
model_ann.add(Dropout(0.3))
model_ann.add(Dense(32, activation='relu'))
model_ann.add(Dropout(0.3))
model_ann.add(Dense(1, activation='sigmoid'))
model_ann.summary()
model_ann.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model_ann.build(input_shape=(None, X_train_ann.shape[1]))
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history = model_ann.fit(X_train_ann, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)
loss, accuracy = model_ann.evaluate(X_test_ann, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")
y_pred_ann = (model_ann.predict(X_test_ann) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred_ann))
print(classification_report(y_test, y_pred_ann))
model_ann.save('ann_churn_model.h5')