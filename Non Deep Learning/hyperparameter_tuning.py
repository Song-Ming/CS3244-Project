import time
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load in preprocessed data
with open("data/models/features.pkl", "rb") as f:
    X, y = pickle.load(f)

print("\n=== DATA SUMMARY ===")
print(f"Samples: {len(X)}, Features: {X.shape[1]}")
print(f"Classes: {np.unique(y)}")

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=28, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature selection using SelectKBest
num_features = 200
selector = SelectKBest(score_func=f_classif, k=num_features)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=28)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=28) # ensures each fold has the same proportion of classes in the original dataset

# # SVM Hyperparameter Tuning
# # Defining SVM parameter grid
# svm_params = {
#     'C' : [0.1, 1, 10],
#     'gamma' : ['scale', 'auto', 0.1],
#     'kernel': ['rbf'] # did not bother with linear due to weed classification requiring identification of non-linear visual patterns
# }

# # Create SVM grid search
# svm_grid = GridSearchCV(
#     estimator=SVC(class_weight='balanced', random_state=28, max_iter=5000, tol=0.001), # based on original model
#     param_grid=svm_params,
#     cv=cv,
#     n_jobs=-1, # uses all available CPU cores for faster model training,
#     scoring='accuracy',
#     verbose=2
# )

# # Fitting SVM grid search
# print("=== SVM Grid Search ===")
# start_time = time.time()
# svm_grid.fit(X_train, y_train)
# end_time = time.time()

# # Print results
# print(f"SVM Grid Search completed in {end_time - start_time:.2f} seconds.")
# print(f"Best parameters: {svm_grid.best_params_}")
# print(f"Best CV accuracy: {svm_grid.best_score_:.4f}")

# # Evaluate on test set
# y_pred = svm_grid.predict(X_test)
# print("\nOptimized SVM Results:")
# print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print(classification_report(y_test, y_pred))

# Saving est model
# model_path = "data/models/best_svm_model.pkl"
# with open(model_path, "wb") as f:
#     pickle.dump(svm_grid.best_estimator_, f)
#     print(f"\nBest SVM model saved to {model_path}")

# Random Forests Hyperparameter Tuning
# intialise hyperparameters
rf_params = {
    'n_estimators' : [200, 300],
    'max_depth' : [None, 15], 
    'min_samples_split' : [2, 5],
    'min_samples_leaf' : [1, 2],
    'max_features': ['sqrt', 0.8]
}

# Create RF grid search
rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(
        class_weight='balanced_subsample',
        random_state=28,
        n_jobs=-1
    ),
    param_grid=rf_params,
    cv=cv,
    scoring='accuracy',
    verbose=2,
)

# Fitting RF Grid Search
print("=== Random Forest Grid Search ===")
start_time = time.time()
rf_grid.fit(X_train, y_train)
end_time = time.time()

# Print results
print(f"RF Grid Search completed in {end_time - start_time:.2f} seconds.")
print(f"Best parameters: {rf_grid.best_params_}")
print(f"Best CV accuracy: {rf_grid.best_score_:.4f}")

# Evaluating on test set
y_pred = rf_grid.predict(X_test)
print("\nOptimized Random Forest Results:")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Saving best model
model_path = "data/best_rf_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(rf_grid.best_estimator_, f)
    print(f"\nBest RF model saved to {model_path}")
