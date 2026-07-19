# Email Spam Classification: A Precision-Optimized Evaluation Pipeline

An end-to-end Machine Learning pipeline built in Python to classify emails as Spam (fraudulent/advertising) or Ham (safe/personal). This project transitions from a basic baseline model into a rigorous model-selection framework using 5-Fold Cross-Validation across six distinct machine learning architectures, ultimately optimizing for high-dimensional text classification under strict operational constraints.

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
To identify the strongest structural configurations for high-dimensional sparse data, models were initially benchmarked using 5-Fold Cross-Validation on the training subset.

### 5-Fold CV Accuracy Leaderboard
| Rank | Algorithm Configuration | Mean Validation Accuracy | Standard Deviation | Architectural Notes |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Linear SVM (`SVC(kernel='linear')`)** | **97.60%** | **0.30%** | Elite stability. Ideally suited for high-dimensional text geometry. |
| **2** | **Random Forest (`RandomForestClassifier`)** | **97.42%** | **0.45%** | Robust ensemble method, but computationally heavier. |
| **3** | **Multinomial Naive Bayes (`MultinomialNB`)** | **96.86%** | **0.55%** | Highly efficient probability engine optimized for discrete counts. |
| **4** | **Logistic Regression** | **94.03%** | **0.27%** | Stable baseline, but missed multi-word token dependencies. |
| **5** | **K-Nearest Neighbors (KNN)** | **91.18%** | **0.50%** | Suffered heavily from the *Curse of Dimensionality*. |

---

## 🎓 Final Evaluation & Deployment Ranking (The Test Set)

Given the dataset's class imbalance, evaluating strictly on accuracy introduces structural risks. In a production spam filter, a **False Positive** (flagging a critical personal or professional email as spam) introduces a significantly higher operational cost than a **False Negative** (allowing a stray spam email into the inbox). 

Therefore, the evaluation function was expanded to track **Precision**, and the top three architectures were ranked based on a **conservative production strategy** (prioritizing flawless precision, followed by secondary test accuracy):

| Production Rank | Model Algorithm | Training Accuracy | Training Precision | Test Accuracy | Test Precision | Deployment Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Multinomial Naive Bayes** | 98.07% | 100.00% | 97.31% | **100.00%** 🎯 | 🥇 **Production Selection** (Flawless safety, highly lightweight) |
| **2** | **Random Forest (`RFC`)** | 100.00% | 100.00% | 97.58% | **100.00%** 🎯 | 🥈 **Defensive Alternative** (Flawless precision, but overfitted training variance) |
| **3** | **Linear SVM (`SVC`)** | 99.53% | 99.31% | 98.21% | **99.27%** | 🥉 **Aggressive Alternative** (Highest overall accuracy, but carries 0.73% false alarm risk) |

---

## 🧠 Core Engineering Insights

* **The Core Strategy Shift:** While **Linear SVM** captures the highest overall raw test accuracy (**98.21%**), it introduces a non-zero risk of false positives ($0.73\%$ of its spam classifications are legitimate ham). Under a conservative framework, **Multinomial Naive Bayes** is selected as the optimal deployment choice. It maintains an elite **97.31% accuracy** while guaranteeing a **100.00% precision rate**—ensuring zero legitimate user emails are lost.
* **The Variance Gap:** **Random Forest** achieved a perfect 100.00% accuracy and precision on the training data due to unconstrained tree depths (`max_depth=None`). While its test precision remained flawless, the performance gap between training and testing indicates minor structural overfitting compared to the more stable probabilistic behavior of Naive Bayes.
* **The Baseline Benchmark:** Given the dataset's 86.6% Ham baseline, a zero-intelligence model guessing "Ham" for everything yields a baseline accuracy of 86.59%. All three final models confidently clear this threshold by over 10%, confirming high predictive validity.

---

## 🚀 How to Run
Click the **"Open in Colab"** badge at the top of the `spam_filter.ipynb` notebook to launch the interactive workspace. Run the cells sequentially to visualize the feature spaces, execute the evaluation pipeline, or test your own custom string inputs against the production-ready models.
