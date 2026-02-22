
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_auc_score

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("iris.csv")

print("First 5 rows:\n", df.head())
print("Shape:", df.shape)
print(df.info())
print("Missing values:\n", df.isnull().sum())
print("Target counts:\n", df["species"].value_counts())

# ----------------------------
# 2. Split Features & Target
# ----------------------------
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)  

# ----------------------------
# 3. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# ----------------------------
# 4. Scale Features
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 5. Initialize Models
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, kernel='rbf', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# ----------------------------
# 6. Train, Evaluate & Cross-Validate
# ----------------------------
for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    
    # Classification Report
    print(f"\n{name} Accuracy: {acc*100:.2f}%")
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"{name} Confusion Matrix:\n", cm)

    # Cross-Validation
    cv_scores = cross_val_score(model, scaler.fit_transform(X), y_encoded, cv=5)
    print(f"{name} 5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    
    # ROC-AUC (multi-class)
    y_test_bin = label_binarize(y_test, classes=[0,1,2])
    y_score = model.predict_proba(X_test_scaled)
    roc_auc = {}
    for i, class_name in enumerate(le.classes_):
        roc_auc[class_name] = roc_auc_score(y_test_bin[:, i], y_score[:, i])
    print(f"{name} ROC-AUC per class:", roc_auc)
