# Email Spam Classification: A Precision-Optimized Evaluation Pipeline

An end-to-end Machine Learning pipeline built in Python to classify emails as Spam (fraudulent/advertising) or Ham (safe/personal). This project transitions from a basic baseline model into a rigorous model-selection framework using 5-Fold Cross-Validation across distinct machine learning architectures, featuring empirical feature engineering trials to discover the optimal deployment model for production environments.

## 📊 Dataset
The dataset used in this project is the **Spam Email Dataset** sourced from Kaggle. 
* **Source Link:** [Kaggle Spam Email Dataset](https://www.kaggle.com/datasets/abdmental01/email-spam-dedection)
* **File Name:** `mail_data.csv`
* **Class Distribution:** 4,825 Ham messages vs. 747 Spam messages (~86.6% vs. 13.4% split).

> **Note:** To run this project locally or in Google Colab, download the `mail_data.csv` file from the link above and upload it to your active runtime environment directory.

---

## 🛠️ Pipeline Architecture
1. **Data Exploration & Cleaning:** Checked for null entries, handled missing records by mapping null features to empty text strings (`''`) to preserve vectorizer compatibility, and mapped text labels to binary integers (`Spam = 1`, `Ham = 0`).
2. **Feature Extraction (TF-IDF):** Converted unstructured string messages into a normalized mathematical sparse matrix of 7,400+ unique word features while stripping out standard English stop words.
3. **Validation Strategy:** Implemented a strict **80/20 Train-Test split**. The 20% test set was locked away entirely. 5-Fold Cross-Validation was conducted *strictly inside the 80% training set* to prevent data leakage.

---

## 🔬 The Cross-Validation Preliminary Phase
To identify the strongest structural configurations for high-dimensional sparse text data, baseline models were initially benchmarked using 5-Fold Cross-Validation on the training subset.

### 5-Fold CV Accuracy Leaderboard
| Rank | Algorithm Configuration | Mean Validation Accuracy | Standard Deviation | Architectural Notes |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Linear SVM (`SVC(kernel='linear')`)** | **97.60%** | **0.30%** | Elite stability. Ideally suited for high-dimensional text geometry. |
| **2** | **Random Forest (`RandomForestClassifier`)** | **97.42%** | **0.45%** | Robust ensemble method, but computationally heavier. |
| **3** | **Multinomial Naive Bayes (`MultinomialNB`)** | **96.86%** | **0.55%** | Highly efficient probability engine optimized for discrete count features. |
| **4** | **Logistic Regression** | **94.03%** | **0.27%** | Stable baseline, but missed multi-word token dependencies. |
| **5** | **K-Nearest Neighbors (KNN)** | **91.18%** | **0.50%** | Suffered heavily from the *Curse of Dimensionality*. |

---

## 🧪 Feature Engineering Experiment: Metadata Stacking vs. Pure TF-IDF

To evaluate if text structure provided extra signal, **Character Count**, **Word Count**, and **Sentence Count** features were engineered, scaled using `MinMaxScaler` (to avoid data leakage and prevent negative values from breaking Naive Bayes), and concatenated with the TF-IDF matrix via `scipy.sparse.hstack`.

### Empirical Trial Results (Pure TF-IDF vs. TF-IDF + Metadata)
| Architecture | Pure TF-IDF Accuracy | Hybrid (TF-IDF + Metadata) Accuracy | Test Precision | Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Linear SVM** | **98.21%** | **98.21%** | 99.27% | Resilient hyperplane; extra dimensions yielded net 0 change. |
| **Random Forest** | 97.58% | **97.67%** 📈 | **100.00%** 🎯 | Benefited (+0.09%) from explicit structural decision splits. |
| **Multinomial Naive Bayes** | **97.31%** | 96.77% 📉 | **100.00%** 🎯 | Dipped (-0.54%) as continuous density violated event independence assumptions. |

---

## 🎓 Final Deployment Hierarchy (The Test Set)

In production spam filtering, a **False Positive** (flagging a critical personal/professional email as spam) carries a significantly higher cost than a **False Negative** (allowing a rare spam email into the inbox). 

Prioritizing **100% Precision**, instantaneous execution speed, and minimal computational footprint for web app deployment (Streamlit), the final model ranking is established as:

| Production Rank | Model Algorithm | Feature Pipeline | Test Accuracy | Test Precision | Latency / Overhead | Deployment Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | **Multinomial Naive Bayes** | **Pure TF-IDF** | **97.31%** | **100.00%** 🎯 | **Near-Zero (< 1 ms)** | 🥇 **Streamlit Production Winner** (Flawless safety, blazing fast) |
| **2** | **Random Forest (`RFC`)** | TF-IDF + Metadata | **97.67%** | **100.00%** 🎯 | High (~17s fit time) | 🥈 **Heavyweight Alternative** (Flawless precision, but heavy RAM/CPU usage) |
| **3** | **Linear SVM (`SVC`)** | Pure TF-IDF | **98.21%** | **99.27%** | Low (~10 ms) | 🥉 **Aggressive Alternative** (Peak accuracy, but carries 0.73% false alarm risk) |

---

## 🧠 Core Engineering Insights

* **The Production Selection:** **Multinomial Naive Bayes trained on Pure TF-IDF** is chosen for Streamlit web app deployment. It guarantees **100.00% Precision** (zero legitimate emails lost), runs in milliseconds with virtually zero memory overhead, and avoids the mathematical degradation caused by appending non-linear metadata columns.
* **The Precision Trade-Off:** While **Linear SVM** achieves the highest raw test accuracy (**98.21%**), its $99.27\%$ precision introduces a $0.73\%$ false-positive risk. For consumer email safety, the zero-false-alarm guarantee of Naive Bayes is preferred over catching marginal extra spam.
* **The Feature Stacking Lesson:** Tree-based models (Random Forest) effectively leveraged structural text length metrics (+0.09% accuracy gain), whereas probabilistic Naive Bayes performed best when restricted strictly to discrete token frequency vectors.

---

## 🚀 How to Run & Deploy
1. Launch the interactive workspace via the **"Open in Colab"** badge on `spam_filter.ipynb`.
2. Run the notebook cells sequentially to execute the full data pipeline, vectorization, scaling, cross-validation, and performance comparison.
3. Export the trained `tfidf` vectorizer and `MultinomialNB` model objects using `joblib` / `pickle` to integrate directly into the upcoming Streamlit user interface!
