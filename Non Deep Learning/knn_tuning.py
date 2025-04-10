import pickle
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

with open("data/models/features.pkl", "rb") as f:
    X, y = pickle.load(f)

print("\n=== DATA SUMMARY ===")
print(f"Samples: {len(X)}, Features: {X.shape[1]}")
print(f"Classes: {np.unique(y)}")

# Define K-Fold Strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=28)

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=28, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA to reduce dimensionality
n_components = 150
pca = PCA(n_components=n_components)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Define paremeter grid

knn_params = {
    'n_neighbors': list(range(1, 15, 2)),  # Test more neighbor values
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Create KNN grid search
knn_grid = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=knn_params,
    cv=cv,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

# Perform grid search for KNN
print("=== KNN GRID SEARCH ===")
start_time = time.time()
knn_grid.fit(X_train, y_train)
end_time = time.time()

# Print results
print(f"\nGrid search completed in {end_time - start_time:.2f} seconds")
print(f"Best parameters: {knn_grid.best_params_}")
print(f"Best cross-validation score: {knn_grid.best_score_:.4f}")

# Evaluate on test set
best_knn = knn_grid.best_estimator_
y_pred = best_knn.predict(X_test)
print("\n=== BEST MODEL TEST RESULTS ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Saving best KNN model
model_path = "data/models/best_knn_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(knn_grid.best_estimator_, f)
    print(f"Best KNN model saved to {model_path}")
