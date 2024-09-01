import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

#data
url = 'https://raw.githubusercontent.com/Blubberbrumm/Techlabs-ds-2/main/Clean_data.csv'
data = pd.read_csv(url)
df.dropna()
x = df.drop('DIABETE3', axis= 1)
y = df['DIABETE3']
y.value_counts(normalize=True)

#train with randomforestclassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
imputer = SimpleImputer(strategy='mean')
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)
smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train_imputed, y_train)
rf_classifier = RandomForestClassifier(bootstrap=True, class_weight="balanced_subsample", random_state=42)
rf_classifier.fit(x_train_res, y_train_res)


from sklearn.preprocessing import StandardScaler, Binarizer
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
param_grid = {
    'n_estimators': np.arange(100, 1001, 100),
    'max_depth': [None] + list(np.arange(10, 101, 10)),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(class_weight='balanced_subsample')



from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=10,  
    cv=skf,
    verbose=2,
    random_state=42,
    n_jobs=1
)
rf_random


rf_random.fit(x_train, y_train)



#predict_proba to get better precision for DIABETE3= 1
best_rf = rf_random.best_estimator_

y_pred_proba = best_rf.predict_proba(x_test)[:, 1]

binarizer = Binarizer(threshold=0.5)

y_pred_binary = binarizer.fit_transform(y_pred_proba.reshape(-1, 1)).ravel()

y_test_binary = binarizer.fit_transform(y_test.values.reshape(-1, 1)).ravel()

conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)
class_report = classification_report(y_test_binary, y_pred_binary)

print("Best Parameters from RandomizedSearchCV:\n", rf_random.best_params_)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)


#best parameters
print(f"Best Hyperparameters: {rf_random.best_params_}")


min_proba = y_pred_proba.min()
max_proba = y_pred_proba.max()

thresholds = np.arange(min_proba, max_proba, 0.01)

precision_scores = []
recall_scores = []

from sklearn.metrics import  precision_score, recall_score, f1_score

for t in thresholds:

    y_pred_threshold = np.where(y_pred_proba > t, 1, 0)


    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)


    precision_scores.append((t, precision))
    recall_scores.append((t, recall))

    print(f"Threshold: {t:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}")















