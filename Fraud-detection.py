import pandas as pd

# load dataset
data = pd.read_csv("creditcard.csv")

# show first 5 rows
print(data.head())

# check fraud vs normal count
print("\nClass Distribution:")
print(data["Class"].value_counts())

# ---------------- SMOTE PART ----------------
from imblearn.over_sampling import SMOTE

# separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# apply SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# check new distribution
print("\nAfter SMOTE:")
print(pd.Series(y_res).value_counts())

# ---------------- MODEL PART ----------------
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05, random_state=42)

model.fit(X_res)

pred = model.predict(X_res)

# convert output
pred = [1 if x == -1 else 0 for x in pred]

print("\nModel Predictions (first 20):")
print(pred[:20])

from sklearn.metrics import classification_report

print("\nModel Evaluation:")
print(classification_report(y_res, pred))