
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score


data_2011 = pd.read_pickle("/Users/iremgonen/PycharmProjects/pythonProject3/2011.pkl")
data_2012 = pd.read_pickle('/Users/iremgonen/PycharmProjects/pythonProject3/2012.pkl')
data_2013 = pd.read_pickle('/Users/iremgonen/PycharmProjects/pythonProject3/2013.pkl')
data_2014 = pd.read_pickle('/Users/iremgonen/PycharmProjects/pythonProject3/2014.pkl')
data_2015 = pd.read_pickle('/Users/iremgonen/PycharmProjects/pythonProject3/2015.pkl')


data = pd.concat([data_2011, data_2012, data_2013, data_2014, data_2015], ignore_index=True)

#Only class 1, 3
data = data[data['DIABETE3'].isin([1, 3])]

#Missing values
data = data.dropna()


X = data.drop(columns=['DIABETE3'])
y = data['DIABETE3']

#Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


rf = RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample')


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
    n_iter=100,
    cv=skf,
    verbose=2,
    random_state=42,
    n_jobs=-1
)


random_search.fit(X, y)


print(f"Best Hyperparameters: {random_search.best_params_}")

#Predictions using the best estimator
y_pred = random_search.best_estimator_.predict(X)


print("Classification Report:\n", classification_report(y, y_pred))

precision = precision_score(y, y_pred, pos_label=1, average='binary')
recall = recall_score(y, y_pred, pos_label=1, average='binary')
f1 = f1_score(y, y_pred, pos_label=1, average='binary')

print(f"Precision for class 1: {precision:.4f}")
print(f"Recall for class 1: {recall:.4f}")
print(f"F1-Score for class 1: {f1:.4f}")

#Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", conf_matrix)
