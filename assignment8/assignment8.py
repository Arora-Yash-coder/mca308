from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd

data = load_iris()
X, y = data.data, data.target

model = DecisionTreeClassifier(criterion='gini')
model.fit(X, y)

plt.figure(figsize=(12,6))
plot_tree(model, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()

sample = [[5.0, 3.4, 1.5, 0.2]]
print("CART Prediction:", data.target_names[model.predict(sample)][0])

df = pd.DataFrame(data.data, columns=data.feature_names)
df["class"] = data.target

def oneR(df):
    best_feature = None
    min_error = float('inf')
    best_rules = {}
    for col in df.columns[:-1]:
        rules = {}
        for val in sorted(df[col].unique()):
            rules[val] = df[df[col] == val]["class"].mode()[0]
        predictions = df[col].map(rules)
        error = sum(predictions != df["class"])
        if error < min_error:
            min_error = error
            best_feature = col
            best_rules = rules
    return best_feature, best_rules, min_error

feature, rules, error = oneR(df)
print("OneR Best Feature:", feature)
print("Rules:", rules)
print("Classification Error:", error)
