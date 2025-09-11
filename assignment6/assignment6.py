import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_std = (X - X.mean()) / X.std()

cov_matrix = np.cov(X_std.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print(eigenvalues)
print(eigenvectors)

idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]
eigenvalues = eigenvalues[idx]

W = eigenvectors[:, :2]

X_pca = X_std.dot(W)

plt.figure(figsize=(8,6))
plt.scatter(X_pca.iloc[:,0], X_pca.iloc[:,1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset')
plt.show()
