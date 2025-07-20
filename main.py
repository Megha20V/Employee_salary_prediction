# Importing Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error 
from joblib import dump

df = pd.read_csv('Salary_Data.csv')
df.dropna(inplace=True)
# Reducing Job titles by omitting titles with less than 25 counts
job_title_count = df['Job Title'].value_counts()
job_title_edited = job_title_count[job_title_count<=25]
df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in job_title_edited else x )

# Combining repeating values of education level
df['Education Level'].replace(["Bachelor's Degree","Master's Degree","phD"],["Bachelor's","Master's","PhD"],inplace=True)

# Mapping Education Level column
education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
df['Education Level'] = df['Education Level'].map(education_mapping)

# Label encoding the categorical variable
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Creating dummies for Job titles
dummies = pd.get_dummies(df['Job Title'],drop_first=True)
df = pd.concat([df,dummies],axis=1)

# Drop Job Title column
df.drop('Job Title',inplace=True,axis=1)

# Save job title columns for later use
job_titles = dummies.columns.tolist()

# Separating the dataset into features and target

# Dataset conntaining all features from df
features = df.drop('Salary',axis=1)

# Series containing target variable to be predicted
target = df['Salary']

# Splitting data into 25% training and 75% test sets
x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.25,random_state=42)

# Random Forest model
rfr = RandomForestRegressor(n_estimators=20,random_state=42)
rfr.fit(x_train,y_train)

print(rfr.score(x_test,y_test))

# Save with joblib
dump({
    "model": rfr,
    "job_titles": job_titles,
    "education_mapping": education_mapping,
    "label_encoder": le
}, "salary_model.joblib")

print("Model saved as salary_model.joblib")