# lr_model.py
# Logistic Regression with Full EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("iris.csv")
print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nInfo:\n", df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget counts:\n", df['species'].value_counts())

# ----------------------------
# 2. Exploratory Data Analysis (EDA)
# ----------------------------

# Statistical Summary
print("\nStatistical Summary:\n", df.describe())

# Distribution of Features
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].hist(bins=10, figsize=(10,6))
plt.suptitle("Feature Distributions")
plt.show()

# Pairplot (Relationship between features and target)
sns.pairplot(df, hue='species')
plt.suptitle("Pairplot of Features colored by Species", y=1.02)
plt.show()

# Correlation Matrix 
numeric_cols = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_cols.corr()
print("\nCorrelation Matrix:\n", corr_matrix)

# Covariance Matrix
cov_matrix = np.cov(numeric_cols.values, rowvar=False)
print("\nCovariance Matrix:\n", cov_matrix)

# Target Distribution
print("Target counts:\n", df["species"].value_counts())

# ----------------------------
# 3. Split Features & Target
# ----------------------------
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 0=setosa,1=versicolor,2=virginica

# ----------------------------
# 4. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# ----------------------------
# 5. Scale Features
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 6. Train Logistic Regression
# ----------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# ----------------------------
# 7. Evaluate Model
# ----------------------------
y_pred = lr_model.predict(X_test_scaled)
y_score = lr_model.predict_proba(X_test_scaled)

# Accuracy
print("\nLogistic Regression Accuracy:", f"{accuracy_score(y_test, y_pred)*100:.2f}%")

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Cross-Validation
cv_scores = cross_val_score(lr_model, scaler.fit_transform(X), y_encoded, cv=5)
print("\n5-Fold CV Accuracy:", f"{cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ROC-AUC (multi-class)
y_test_bin = label_binarize(y_test, classes=[0,1,2])
roc_auc = {}
for i, class_name in enumerate(le.classes_):
    roc_auc[class_name] = roc_auc_score(y_test_bin[:, i], y_score[:, i])
print("\nROC-AUC per class:", roc_auc)