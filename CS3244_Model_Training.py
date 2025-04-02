#SVM and Random Forest

import pickle
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

with open("data/models/features.pkl", "rb") as f:
    X, y = pickle.load(f)

print("\n=== DATA SUMMARY ===")
print(f"Samples: {len(X)}, Features: {X.shape[1]}")
print(f"Classes: {np.unique(y)}")

#Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=28, stratify=y
)

#Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Feature selection using SelectKBest
num_features = 200
selector = SelectKBest(score_func=f_classif, k=num_features)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

#Handle class imbalance with SMOTE
smote = SMOTE(random_state=28)
X_train, y_train = smote.fit_resample(X_train, y_train)

#Initialise models with class weighting
models = {
    "SVM": SVC(kernel='rbf', C=1.0, max_iter=5000, tol=0.001, random_state=28, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=28)
}

#Train and evaluate models
print("\n=== TRAINING ===")
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print(f"{name} training completed in {end_time - start_time:.2f} seconds.")
    
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))


#KNN


import pickle
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

with open("data/models/features.pkl", "rb") as f:
    X, y = pickle.load(f)

print("\n=== DATA SUMMARY ===")
print(f"Samples: {len(X)}, Features: {X.shape[1]}")
print(f"Classes: {np.unique(y)}")

#Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=28, stratify=y
)

#Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Apply PCA to reduce dimensionality
n_components = 150
pca = PCA(n_components=n_components)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


#Initialise KNN with class weighting for imbalance
models = {"KNN": KNeighborsClassifier(n_neighbors=5, metric="manhattan", weights="distance")}

#Train and evaluate KNN
print("\n=== TRAINING ===")
for name, model in models.items():
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"{name} training completed in {end_time - start_time:.2f} seconds.")

    y_pred = model.predict(X_test)
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))




