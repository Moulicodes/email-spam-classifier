# Email Spam Classification: An Algorithmic Tournament

An end-to-end Machine Learning pipeline built in Python to classify emails as Spam (fraudulent/advertising) or Ham (safe/personal). This project transitions from a basic baseline model into a rigorous model-selection tournament using 5-Fold Cross-Validation across six distinct machine learning architectures to discover the absolute optimal classifier for high-dimensional text data.

## 📊 Dataset
The dataset used in this project is the **Spam Email Dataset** sourced from Kaggle. 
* **Source Link:** [Kaggle Spam Email Dataset](https://www.kaggle.com/datasets/abdmental01/email-spam-dedection)
* **File Name:** `mail_data.csv`

> **Note:** To run this project locally or in Google Colab, download the `mail_data.csv` file from the link above and upload it to your active runtime environment directory.

---

## 🛠️ Pipeline Architecture
1. **Data Exploration & Cleaning:** Checked for null entries, analyzed class distributions, and mapped text labels to binary integers (`Spam = 1`, `Ham = 0`).
2. **Feature Extraction (TF-IDF):** Converted unstructured string messages into a normalized mathematical sparse matrix of 7,400+ unique word features while stripping out standard English stop words.
3. **Validation Strategy:** Implemented a strict **80/20 Train-Test split**. The 20% test set was locked away entirely. 5-Fold Cross-Validation was conducted *strictly inside the 80% training set* to prevent data leakage.

---

## 🏆 The Cross-Validation Tournament
To find the absolute best "brain" for the dataset, an automated validation function was written to test multiple families of machine learning algorithms on an even playing field. 

### Phase 1: 5-Fold CV Leaderboard
| Rank | Algorithm Configuration | Mean Validation Accuracy | Standard Deviation | Architectural Notes |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Linear SVM (`SVC(kernel='linear')`)** | **97.60%** | **0.30%** | **Tournament Winner.** Perfectly suited for high-dimensional space. |
| **2** | **Random Forest (`RandomForestClassifier`)** | **97.42%** | **0.45%** | Robust ensemble method, but computationally heavier. |
| **3** | **Multinomial Naive Bayes (`MultinomialNB`)** | **96.86%** | **0.55%** | Highly efficient probability engine optimized for text. |
| **4** | **Logistic Regression** | **94.03%** | **0.27%** | Incredibly stable baseline, but missed complex text patterns. |
| **5** | **K-Nearest Neighbors (KNN)** | **91.18%** | **0.50%** | Struggled severely due to the *Curse of Dimensionality*. |

---

## 🎓 Final Evaluation (The Hidden Test Set)
The top three best-performing models from the cross-validation phase were advanced to the final exam, where they were trained on the *complete* training dataset and evaluated exactly once on the untouched **Test Set**:

| Model Algorithm | Training Accuracy | Final Test Accuracy | Status |
| :--- | :---: | :---: | :---: |
| **Linear SVM (`SVC`)** | **99.53%** | **98.21%** | 🥇 *Production Champion* |
| **Random Forest (`RFC`)** | 100.00% | 97.58% | *Slightly Overfitted* |
| **Multinomial Naive Bayes** | 98.07% | 97.31% | *Most Lightweight* |

### 🧠 Core Engineering Insights
* **The Victory of Geometry:** **Linear SVM** emerged as the definitive production champion. Because a vocabulary matrix creates thousands of dimensions, the data points naturally spread out, making them linearly separable. Dropping a flat linear boundary sheet (hyperplane) between the classes generalized beautifully to unseen data.
* **The Overfitting Trap:** **Random Forest** achieved a flawless **100.00%** score on the training data because its tree depths were unconstrained, causing it to memorize hyper-specific training quirks. Consequently, it dropped slightly on the real test exam compared to the smoother linear boundary of the SVM.
* **The Consistency Indicator:** Every top model maintained a cross-validation Standard Deviation of **$< 0.60\%$**, proving that the underlying data preprocessing pipeline is highly stable, uniform, and reliable for future streaming data.

---

## 🚀 How to Run
Click the **"Open in Colab"** badge at the top of the `spam_filter.ipynb` notebook to launch the interactive workspace. Run the cells sequentially to visualize the feature spaces, execute the tournament function, or test your own custom email strings against the trained production champion!
