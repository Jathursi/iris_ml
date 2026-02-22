# svm_model.py
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix , roc_auc_score

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("iris.csv")

# Features & target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)  

# ----------------------------
# 2. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# ----------------------------
# 3. Scale Features
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 4. Train SVM
# ----------------------------
svm_model = SVC(probability=True, kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# ----------------------------
# 5. Evaluate
# ----------------------------
y_pred = svm_model.predict(X_test_scaled)
y_score = svm_model.predict_proba(X_test_scaled)

print("SVM Accuracy:", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Cross-validation
cv_scores = cross_val_score(svm_model, scaler.fit_transform(X), y_encoded, cv=5)
print("5-Fold CV Accuracy:", f"{cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ROC-AUC (multi-class)
y_test_bin = label_binarize(y_test, classes=[0,1,2])
roc_auc = {}
for i, class_name in enumerate(le.classes_):
    roc_auc[class_name] = roc_auc_score(y_test_bin[:, i], y_score[:, i])
print("ROC-AUC per class:", roc_auc)