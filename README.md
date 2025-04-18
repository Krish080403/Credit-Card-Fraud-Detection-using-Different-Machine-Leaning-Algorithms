## Credit Card Fraud Detection Using Machine Learning

This repository provides a comprehensive end-to-end pipeline for credit card fraud detection, leveraging both classical CPU-based and GPU-accelerated machine learning algorithms. It is organized into two Jupyter notebooks:

- **Models Implementation.ipynb**: Implements and benchmarks a suite of nine classification algorithms on the Kaggle credit card dataset, including logistic regression, linear discriminant analysis, K-nearest neighbors (KNN), decision trees (CART), Naive Bayes (NB), support vector machines (SVM), random forests (RF), XGBoost, and LightGBM, with SMOTE oversampling and PCA-driven feature selection.
- **Additional Analysis and Improvements.ipynb**: Explores temporal patterns in fraud (hour-of-day analysis), demonstrates a GPU-accelerated pipeline using RAPIDS (cuDF, cuML) for scaling, PCA, and KNN, and builds a stacked meta-model (KNN + XGBoost + LightGBM) combined via GPU-based logistic regression. This notebook also showcases performance improvements and time-based feature analysis.

---

## Repository Structure
```
├── Models Implementation.ipynb        # Core model training and evaluation (CPU-based)
├── Additional Analysis and Improvements.ipynb  # GPU pipeline, stacking, and temporal analysis
├── creditcard.csv                     # Original Kaggle dataset (download separately)
└── README.md                          # Project overview and instructions
```

---

## Environment Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/Krish080403/Credit-Card-Fraud-Detection-using-Different-Machine-Leaning-Algorithms.git
   cd Credit-Card-Fraud-Detection-using-Different-Machine-Leaning-Algorithms
   ```

2. **Create and activate a Python environment**
   ```bash
   conda create -n fraud-detect python=3.12 -y
   conda activate fraud-detect
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place the Kaggle dataset**
   - Download `creditcard.csv` from [Kaggle's Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) and save it in the project root.

---

## Notebook 1: Models Implementation.ipynb

### Data Preparation
1. Load `creditcard.csv`, drop missing labels, and standardize the `Amount` feature with scikit-learn’s `StandardScaler`.
2. Assemble features: PCA components `V1–V28` + `Amount_scaled`.
3. Split data into 70% train / 30% test (stratified by fraud label).
4. Apply SMOTE on the **training set only** to address the 0.17% class imbalance.
5. Optionally, perform feature pruning by removing highly correlated components (|corr| > 0.99).

### Model Suite and Benchmarking
- **Linear Models**: Logistic Regression (LR), Linear Discriminant Analysis (LDA)
- **Instance-Based**: K-Nearest Neighbors (k = 9)
- **Tree-Based**: CART, Random Forest (RF), XGBoost, LightGBM
- **Others**: Naive Bayes (NB), Support Vector Machine (SVM)

Each model is trained on the SMOTE-balanced training set, then evaluated on the untouched test set across six metrics: ROC AUC, Precision–Recall AUC (PRAUC), F1 score, recall, precision, accuracy, and the Kolmogorov–Smirnov (KS) statistic.

#### Key Results (Test Set)
| Model            | AUC   | PRAUC | F1     | Recall | Precision | Accuracy | KS     |
|------------------|-------|-------|--------|--------|-----------|----------|--------|
| Logistic Reg.    | 0.9517| 0.4901| 0.0937 | 0.9308 | 0.0493    | 0.9726   | 0.9090 |
| LDA              | 0.9153| 0.4616| 0.1408 | 0.8462 | 0.0768    | 0.9843   | 0.9090 |
| KNN              | 0.9337| 0.6424| 0.5622 | 0.8692 | 0.4154    | 0.9979   | 0.9090 |
| CART             | 0.8949| 0.5619| 0.4671 | 0.7923 | 0.3312    | 0.9972   | 0.9090 |
| Naive Bayes      | 0.9261| 0.4643| 0.0971 | 0.8769 | 0.0514    | 0.9752   | 0.9090 |
| SVM              | 0.9293| 0.4724| 0.1258 | 0.8769 | 0.0678    | 0.9815   | 0.9090 |
| Random Forest    | 0.9191| 0.8354| 0.8352 | 0.8385 | 0.8321    | 0.9995   | 0.9090 |
| XGBoost          | 0.9412| 0.6285| 0.5239 | 0.8846 | 0.3722    | 0.9976   | 0.9090 |
| LightGBM         | 0.9339| 0.6721| 0.6141 | 0.8692 | 0.4748    | 0.9983   | 0.9090 |

> **Best single model**: Random Forest (RF) demonstrated leading KS (0.9090) and balanced precision/recall (F1 = 0.8352). However, its recall and PRAUC suggest room for improvement in rare-event detection.

---

## Notebook 2: Additional Analysis and Improvements.ipynb

### Temporal Analysis (Time-of-Day)
- **Feature engineering**: Compute `Hour = (Time // 3600) % 24`, then cyclical encode via `sin`/`cos` transforms.
- **Exploratory plot**: Bar chart of fraud rate by hour of day, revealing peak fraud incidence at the hour where `peak_rate` occurs.

### GPU-Accelerated KNN via RAPIDS
1. **Data transfer**: Convert SMOTE-balanced train and original test sets from pandas to cuDF.
2. **GPU standardization**: `cuml.preprocessing.StandardScaler` on GPU.
3. **GPU PCA**: `cuml.decomposition.PCA(n_components=20)` to reduce to 20 dimensions.
4. **GPU KNN**: `cuml.neighbors.KNeighborsClassifier(n_neighbors=7)` for local anomaly detection.
5. **Metrics (back on CPU)**: ROC AUC = 0.9074, PRAUC = 0.4223, F1 = 0.3833, recall = 0.7770, precision = 0.2544, accuracy = 0.9957, KS = 0.8106.

### Stacked Meta‑Model
- **Base learners**: GPU KNN + XGBoost (GPU `gpu_hist`) + LightGBM (CPU).
- **Meta‑features**: Out‑of‑fold probability predictions for each test point.
- **Meta‑learner**: GPU-based `cuml.linear_model.LogisticRegression`.

#### Stacked Model Performance (Test Set)
| Metric    | Value  |
|-----------|--------|
| AUC       | 0.9670 |
| PRAUC     | 0.8260 |
| F1 Score  | 0.8042 |
| Recall    | 0.7770 |
| Precision | 0.8333 |
| Accuracy  | 0.9993 |
| KS Stat   | 0.8434 |

> **Improvement over RF baseline**: +4.79% AUC, –6.36% PRAUC gap to perfect, +0.20 absolute F1; stronger separation (KS) and higher precision on rare fraud events.

---

## How to Reproduce
1. **Open each notebook** in JupyterLab or Google Colab.
2. **Ensure `creditcard.csv`** is placed at the project root.
3. **Run all cells** sequentially; the notebooks are fully self-contained.

---



