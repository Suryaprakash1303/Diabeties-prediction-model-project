import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib

print("Loading data...")
col_names = ['Pregnancy_Count','Glucose_conc','Blood_pressure','Skin_thickness','Insulin','BMI','DPF','Age','Class']
data = pd.read_csv('Diabetes.csv', skiprows=9, names=col_names)

print("Preprocessing data...")
data.iloc[:,1:6] = data.iloc[:,1:6].replace(0, np.NaN)
data.dropna(thresh=2, axis=0, inplace=True)

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
data.iloc[:,1:6] = imputer.fit_transform(data.iloc[:,1:6])

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training model...")
clf = LogisticRegression(C=1000, random_state=5)
clf.fit(X_scaled, y)

print("Saving model and scaler...")
joblib.dump(clf, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')

print("Done! You can now run the app.")
