import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

df = sns.load_dataset('titanic')

plt.figure()
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap (Before Cleaning)")
plt.savefig("missing_values_heatmap.png")
plt.show()

df = df.drop(['deck', 'embark_town', 'alive', 'who', 'adult_male', 'class'], axis=1)

df = pd.get_dummies(df, drop_first=True)

df = df.fillna(df.median())

X = df.drop('survived', axis=1)
y = df['survived']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

importances = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:\n")
print(feature_importance.head(10))

top10 = feature_importance.head(10)

plt.figure()
sns.barplot(x='Importance', y='Feature', data=top10)

plt.title('Top 10 Feature Importance (Titanic Dataset)')
plt.xlabel('Importance Score')
plt.ylabel('Features')

plt.savefig("top10_features.png")

plt.show()