#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# ## DATA INGESTION AND EXPLORATION

# In[5]:


df=pd.read_csv("Telco-Customer-Churn.csv")


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

print("Numerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)


# In[10]:


df[num_cols].describe()


# In[11]:


for col in cat_cols:
    print(f"=== {col} ===")
    print(df[col].value_counts())
    print("\n")


# ## DATA CLEANING AND PREPROCESSING

# In[12]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')

df['TotalCharges'] = df['TotalCharges'].fillna(0)

print("New TotalCharges dtype:", df['TotalCharges'].dtype)


# In[13]:


import pandas as pd

if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

X = df.drop(columns=['Churn']) if 'Churn' in df.columns else df.copy()
y = df['Churn'].map({'Yes': 1, 'No': 0}) if 'Churn' in df.columns and df['Churn'].dtype == 'object' else df['Churn']

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    if col in X.columns and X[col].dtype == 'object':
        X[col] = X[col].map({'Yes': 1, 'No': 0})

if 'gender' in X.columns and X['gender'].dtype == 'object':
    X['gender'] = X['gender'].map({'Female': 1, 'Male': 0})

X = pd.get_dummies(X, drop_first=True, dtype=int)

print("X shape:", X.shape)
print("y shape:", y.shape)
X.head()


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Train-Test Split (67% train, 33% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

# Standard Scaling on continuous numerical features
scaler = StandardScaler()
scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

X_train = X_train.copy()
X_test = X_test.copy()

X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# ### 1. LOGISTIC REGRESSION

# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 8: Instantiate and train model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Step 9: Evaluate model performance
y_pred = lr.predict(X_test)

print("Train Accuracy:", lr.score(X_train, y_train))
print("Test Accuracy:", lr.score(X_test, y_test))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



# In[16]:


# Tuned Logistic Regression model
lr_tuned = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000)
lr_tuned.fit(X_train, y_train)

print("--- Tuned Model Performance ---")
print("Tuned Train Accuracy:", lr_tuned.score(X_train, y_train))
print("Tuned Test Accuracy:", lr_tuned.score(X_test, y_test))
print("\nTuned Classification Report:\n", classification_report(y_test, lr_tuned.predict(X_test)))


# ### 2. K- NEAREST NEIGHBOUR MODEL

# In[17]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Baseline KNN (K=5) ---
knn_baseline = KNeighborsClassifier(n_neighbors=5)
knn_baseline.fit(X_train, y_train)

y_train_pred_knn_base = knn_baseline.predict(X_train)
y_test_pred_knn_base = knn_baseline.predict(X_test)

print("=== BASELINE KNN (K=5) ===")
print("Train Accuracy:", accuracy_score(y_train, y_train_pred_knn_base))
print("Test Accuracy :", accuracy_score(y_test, y_test_pred_knn_base))
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred_knn_base))

print("-" * 50)

# --- Tuned KNN (K=11, weights='distance') ---
knn_tuned = KNeighborsClassifier(n_neighbors=11, weights='distance')
knn_tuned.fit(X_train, y_train)

y_test_pred_knn_tuned = knn_tuned.predict(X_test)

print("=== TUNED KNN (K=11, weights='distance') ===")
print("Tuned Test Accuracy:", accuracy_score(y_test, y_test_pred_knn_tuned))
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred_knn_tuned))


# ### 3. SUPPORT VECTOR CLASSIFIER MODEL

# In[18]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. Baseline SVC (Default: RBF Kernel, C=1.0)
# ==========================================
svc_baseline = SVC(kernel='rbf', C=1.0, random_state=42)
svc_baseline.fit(X_train, y_train)

y_train_pred_svc_base = svc_baseline.predict(X_train)
y_test_pred_svc_base = svc_baseline.predict(X_test)

print("=== BASELINE SVC (kernel='rbf', C=1.0) ===")
print("Baseline Train Accuracy:", accuracy_score(y_train, y_train_pred_svc_base))
print("Baseline Test Accuracy :", accuracy_score(y_test, y_test_pred_svc_base))
print("\nBaseline Test Classification Report:\n", classification_report(y_test, y_test_pred_svc_base))

print("=" * 55)

# ==========================================
# 2. Tuned SVC (Linear Kernel, C=0.1)
# ==========================================
svc_tuned = SVC(kernel='linear', C=0.1, random_state=42)
svc_tuned.fit(X_train, y_train)

y_train_pred_svc_tuned = svc_tuned.predict(X_train)
y_test_pred_svc_tuned = svc_tuned.predict(X_test)

print("=== TUNED SVC (kernel='linear', C=0.1) ===")
print("Tuned Train Accuracy:", accuracy_score(y_train, y_train_pred_svc_tuned))
print("Tuned Test Accuracy :", accuracy_score(y_test, y_test_pred_svc_tuned))
print("\nTuned Test Classification Report:\n", classification_report(y_test, y_test_pred_svc_tuned))


# ### 4. DECISION TREE MODEL

# In[19]:


from sklearn.tree import DecisionTreeClassifier

# --- 1. Baseline Decision Tree ---
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

print("--- Baseline Decision Tree Performance ---")
print("Train Accuracy:", dt.score(X_train, y_train))
print("Test Accuracy:", dt.score(X_test, y_test))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, dt.predict(X_test)))
print("\nClassification Report:\n", classification_report(y_test, dt.predict(X_test)))

# --- 2. Tuned Decision Tree ---
dt_tuned = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt_tuned.fit(X_train, y_train)

print("\n--- Tuned Decision Tree Performance ---")
print("Tuned Train Accuracy:", dt_tuned.score(X_train, y_train))
print("Tuned Test Accuracy:", dt_tuned.score(X_test, y_test))
print("\nTuned Confusion Matrix:\n", confusion_matrix(y_test, dt_tuned.predict(X_test)))
print("\nTuned Classification Report:\n", classification_report(y_test, dt_tuned.predict(X_test)))


# ### 5. RANDOM FOREST MODEL

# In[20]:


from sklearn.ensemble import RandomForestClassifier

# --- 1. Baseline Random Forest ---
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

print("--- Baseline Random Forest Performance ---")
print("Train Accuracy:", rf.score(X_train, y_train))
print("Test Accuracy:", rf.score(X_test, y_test))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf.predict(X_test)))
print("\nClassification Report:\n", classification_report(y_test, rf.predict(X_test)))

# --- 2. Tuned Random Forest ---
rf_tuned = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
rf_tuned.fit(X_train, y_train)

print("\n--- Tuned Random Forest Performance ---")
print("Tuned Train Accuracy:", rf_tuned.score(X_train, y_train))
print("Tuned Test Accuracy:", rf_tuned.score(X_test, y_test))
print("\nTuned Confusion Matrix:\n", confusion_matrix(y_test, rf_tuned.predict(X_test)))
print("\nTuned Classification Report:\n", classification_report(y_test, rf_tuned.predict(X_test)))


# ### 6. NAIVE BAYES MODEL

# In[21]:


from sklearn.naive_bayes import GaussianNB

# --- Naive Bayes Model ---
nb = GaussianNB()
nb.fit(X_train, y_train)

print("--- Naive Bayes Performance ---")
print("Train Accuracy:", nb.score(X_train, y_train))
print("Test Accuracy:", nb.score(X_test, y_test))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, nb.predict(X_test)))
print("\nClassification Report:\n", classification_report(y_test, nb.predict(X_test)))


# In[22]:


import pandas as pd

eval_data = [
    {"Model": "Logistic Regression (Base)", "Test Accuracy": 0.8060, "Class 1 Precision": 0.66, "Class 1 Recall": 0.55, "Class 1 F1-Score": 0.60},
    {"Model": "Logistic Regression (Tuned)", "Test Accuracy": 0.7505, "Class 1 Precision": 0.52, "Class 1 Recall": 0.80, "Class 1 F1-Score": 0.63},
    {"Model": "SVC (Base)", "Test Accuracy": 0.8017, "Class 1 Precision": 0.67, "Class 1 Recall": 0.49, "Class 1 F1-Score": 0.57},
    {"Model": "SVC (Tuned)", "Test Accuracy": 0.7467, "Class 1 Precision": 0.51, "Class 1 Recall": 0.79, "Class 1 F1-Score": 0.62},
    {"Model": "KNN (Base)", "Test Accuracy": 0.7677, "Class 1 Precision": 0.57, "Class 1 Recall": 0.53, "Class 1 F1-Score": 0.55},
    {"Model": "KNN (Tuned)", "Test Accuracy": 0.7712, "Class 1 Precision": 0.57, "Class 1 Recall": 0.53, "Class 1 F1-Score": 0.55},
    {"Model": "Decision Tree (Base)", "Test Accuracy": 0.7359, "Class 1 Precision": 0.50, "Class 1 Recall": 0.51, "Class 1 F1-Score": 0.51},
    {"Model": "Decision Tree (Tuned)", "Test Accuracy": 0.7875, "Class 1 Precision": 0.62, "Class 1 Recall": 0.51, "Class 1 F1-Score": 0.56},
    {"Model": "Random Forest (Base)", "Test Accuracy": 0.7892, "Class 1 Precision": 0.64, "Class 1 Recall": 0.48, "Class 1 F1-Score": 0.55},
    {"Model": "Random Forest (Tuned)", "Test Accuracy": 0.7768, "Class 1 Precision": 0.56, "Class 1 Recall": 0.72, "Class 1 F1-Score": 0.63},
    {"Model": "Naive Bayes", "Test Accuracy": 0.6508, "Class 1 Precision": 0.42, "Class 1 Recall": 0.88, "Class 1 F1-Score": 0.57}
]

comparison_table = pd.DataFrame(eval_data)
comparison_table = comparison_table.sort_values(by="Test Accuracy", ascending=False).reset_index(drop=True)

print("--- Model Performance Comparison Table ---")
display(comparison_table)


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

plot_df = comparison_table.sort_values(by="Test Accuracy", ascending=True)

ax = sns.barplot(
    x="Test Accuracy", 
    y="Model", 
    data=plot_df, 
    palette="Blues_r"
)

for p in ax.patches:
    width = p.get_width()
    ax.text(
        width + 0.003, 
        p.get_y() + p.get_height() / 2., 
        f'{width:.4f}', 
        ha="left", 
        va="center", 
        fontsize=10, 
        color='black', 
        fontweight='semibold'
    )

plt.title("Model Performance Benchmark: Test Accuracy Comparison", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Test Accuracy Score", fontsize=11)
plt.ylabel("Model Configuration", fontsize=11)
plt.xlim(0.60, 0.85) 
plt.tight_layout()
plt.show()


# ## Conclusion & Key Takeaways
# 
# * **Baseline vs. Overfitting:** Unconstrained tree models (Decision Tree & Random Forest) heavily overfit (>99% train accuracy), requiring strict pruning and class balancing to generalize.
# * **Accuracy Champions:** **Baseline Logistic Regression** and **SVC** achieved the highest overall test accuracy (~80.6%), making them ideal for general prediction.
# * **Business Impact (Recall):** For churn prediction, catching actual churners matters more than raw accuracy. **Tuned Logistic Regression** and **SVC** successfully captured ~80% of churners (Class 1 Recall = ~0.80).
# * **Final Verdict:** **Tuned Logistic Regression / Random Forest** offers the best operational balance, allowing telecom companies to flag high-risk customers early and reduce revenue loss.

# In[ ]:




