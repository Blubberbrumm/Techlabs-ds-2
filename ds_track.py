
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

data_2011 = pd.read_pickle("/Users/iremgonen/PycharmProjects/pythonProject3/2011.pkl")
data_2012 = pd.read_pickle('/Users/iremgonen/PycharmProjects/pythonProject3/2012.pkl')
data_2013 = pd.read_pickle('/Users/iremgonen/PycharmProjects/pythonProject3/2013.pkl')
data_2014 = pd.read_pickle('/Users/iremgonen/PycharmProjects/pythonProject3/2014.pkl')
data_2015 = pd.read_pickle('/Users/iremgonen/PycharmProjects/pythonProject3/2015.pkl')

#combining data
data = pd.concat([data_2011, data_2012, data_2013, data_2014, data_2015], ignore_index=True)

data = pd.DataFrame({
    'Diabetes': data['DIABETE3'],
    'BMI': data['_BMI5'],
    'Education': data['EDUCA'],
    'Is_Smoker': data['_RFSMOK3']
})

#missing values
data = data.dropna()

#feature matrix (X) and target vector (y)
X = data[['BMI', 'Education', 'Is_Smoker']]
y = data['Diabetes']

#training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#initializing random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

#training model
rf.fit(X_train, y_train)

#predictions on the test set
y_pred = rf.predict(X_test)

#evaluation of model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Classification Report:\n", report)

#feature importance analysis
importances = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:\n", importance_df)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance in Random Forest Model')
plt.show()

#boxplots
plt.figure(figsize=(14, 6))

#BMI and Diabetes
plt.subplot(1, 3, 1)
sns.boxplot(x='Diabetes', y='BMI', data=data)
plt.title('BMI and Diabetes')

#Education and Diabetes
plt.subplot(1, 3, 2)
sns.boxplot(x='Diabetes', y='Education', data=data)
plt.title('Education and Diabetes')

#Smoking and Diabetes
plt.subplot(1, 3, 3)
sns.boxplot(x='Diabetes', y='Is_Smoker', data=data)
plt.title('Smoking and Diabetes')

plt.tight_layout()
plt.show()
