import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('telco_dataset.csv')


df['Churn'].value_counts().plot(kind='bar', color=['green','red'])
plt.title('Churn vs No Churn')
plt.savefig('graph1.png')
plt.close()


df.boxplot(column='MonthlyCharges', by='Churn')
plt.title('Monthly Charges vs Churn')
plt.savefig('graph2.png')
plt.close()


df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df_model = df.select_dtypes(include=['number'])
X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Project Complete!")
