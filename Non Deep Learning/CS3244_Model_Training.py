# === ALL IMPORTS ===

import pickle
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# === LOAD DATA ===
with open("data/models/features.pkl", "rb") as f:
    X, y = pickle.load(f)

print("\n=== DATA SUMMARY ===")
print(f"Samples: {len(X)}, Features: {X.shape[1]}")
print(f"Classes: {np.unique(y)}")

# === GLOBAL STORAGE FOR EVALUATION ===
all_preds = []
all_truths = []
f1_scores = {}

# === SVM and RANDOM FOREST ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=28)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
selector = SelectKBest(score_func=f_classif, k=200)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)
smote = SMOTE(random_state=28)
X_train, y_train = smote.fit_resample(X_train, y_train)

models = {
    "SVM": SVC(kernel='rbf', C=1.0, max_iter=5000, tol=0.001, class_weight='balanced', random_state=28),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=28)
}

print("\n=== TRAINING (SVM & RF) ===")
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)
    all_preds.extend(y_pred)
    all_truths.extend(y_test)

    f1 = f1_score(y_test, y_pred, average='micro')
    f1_scores[name] = f1

    print(f"\n{name} Results:")
    print(f"Training Time: {end - start:.2f}s")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score (micro): {f1:.4f}")
    print(classification_report(y_test, y_pred))

# === KNN ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=28)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
pca = PCA(n_components=150)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5, metric="manhattan", weights="distance")

print("\n=== TRAINING (KNN) ===")
start = time.time()
knn_model.fit(X_train, y_train)
end = time.time()

y_pred = knn_model.predict(X_test)
all_preds.extend(y_pred)
all_truths.extend(y_test)

f1 = f1_score(y_test, y_pred, average='micro')
f1_scores["KNN"] = f1

print("\nKNN Results:")
print(f"Training Time: {end - start:.2f}s")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score (micro): {f1:.4f}")
print(classification_report(y_test, y_pred))

# === PRINT OVERALL F1 SCORES ===
print("\n=== SUMMARY F1 SCORES ===")
for model_name, score in f1_scores.items():
    print(f"{model_name}: F1 Score (micro) = {score:.4f}")

# === COMBINED CONFUSION MATRIX ===
print("\n=== Combined Confusion Matrix ===")
cm = confusion_matrix(all_truths, all_preds)
labels = ['Chinee apple', 'Snake weed', 'Lantana', 'Prickly acacia', 'Siam weed',
          'Parthenium', 'Rubber vine', 'Parkinsonia', 'Negative']

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Combined Confusion Matrix (SVM, RF, KNN)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()
