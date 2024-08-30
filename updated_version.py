
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


data_2011 = pd.read_pickle("/Users/iremgonen/PycharmProjects/pythonProject3/2011.pkl")
data_2012 = pd.read_pickle("/Users/iremgonen/PycharmProjects/pythonProject3/2012.pkl")
data_2013 = pd.read_pickle("/Users/iremgonen/PycharmProjects/pythonProject3/2013.pkl")
data_2014 = pd.read_pickle("/Users/iremgonen/PycharmProjects/pythonProject3/2014.pkl")
data_2015 = pd.read_pickle("/Users/iremgonen/PycharmProjects/pythonProject3/2015.pkl")


data = pd.concat([data_2011, data_2012, data_2013, data_2014, data_2015], ignore_index=True)

#Only 1, 3
data = data[data['DIABETE3'].isin([1, 3])]


data = data.dropna()


X = data.drop(columns=['DIABETE3'])
y = data['DIABETE3']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


rf = RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample')

#Hyperparameters
param_dist = {
    'n_estimators': np.arange(100, 1001, 100),
    'max_depth': [None] + list(np.arange(10, 101, 10)),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}


random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,  # Number of different combinations to try
    cv=skf,  # Stratified K-Fold Cross-Validation
    verbose=2,  # Verbosity mode
    random_state=42,  # For reproducibility
    n_jobs=-1  # Use all available cores
)


random_search.fit(X_train, y_train)


print(f"Best Hyperparameters: {random_search.best_params_}")


y_pred = random_search.best_estimator_.predict(X_test)


print("Classification Report:\n", classification_report(y_test, y_pred))

#Precision, Recall, F1 for class 1 using default threshold
precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
recall = recall_score(y_test, y_pred, pos_label=1, average='binary')
f1 = f1_score(y_test, y_pred, pos_label=1, average='binary')

print(f"Precision for class 1: {precision:.4f}")
print(f"Recall for class 1: {recall:.4f}")
print(f"F1-Score for class 1: {f1:.4f}")

#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


y_pred_proba = random_search.best_estimator_.predict_proba(X_test)[:, 1]


min_proba = y_pred_proba.min()
max_proba = y_pred_proba.max()


thresholds = np.arange(min_proba, max_proba, 0.01)


precision_scores = []
recall_scores = []

for t in thresholds:

    y_pred_threshold = np.where(y_pred_proba > t, 1, 0)


    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)


    precision_scores.append((t, precision))
    recall_scores.append((t, recall))

    print(f"Threshold: {t:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}")




