# 🎓 Student Mental Health & Dropout Risk: From EDA to Predictive Modeling

This repository presents an end-to-end data science pipeline—ranging from high-performance **Exploratory Data Analysis (EDA)** to **Predictive Regression Modeling**—on a massive dataset of 1,000,000 students. The project investigates the complex psychological and academic triggers behind student burnout and dropout risks.

---

## 📂 Project Structure

*   **`student-mental-health-eda.ipynb`**: Focuses on memory-efficient data loading, statistical distributions, and multivariate analysis.
*   **`student-mental-health-model.py`**: Focuses on feature selection, pipeline construction, and the implementation of regression algorithms.

---

## 🛠️ High-Performance Engineering

Handling 1,000,000 rows requires more than standard procedures. This project highlights:

### 1. Memory Optimization (Downcasting)
To ensure the project runs smoothly on standard environments (Kaggle/Colab), we implemented a custom optimization function:
*   **Numerical Downcasting:** Converted `float64` to `float32` and `int64` to `int8/int16`.
*   **Categorical Conversion:** Transformed repetitive object types into `category` dtypes.
*   **Efficiency:** Reduced RAM consumption by **~65%**, enabling faster computations without losing precision.

### 2. Intelligent Sampling
For visualization and iterative model testing:
*   **Stratified Sampling:** We utilized stratified techniques to extract a representative 50,000-row subset, ensuring that the distribution of `Dropout Risk` remains identical to the full 1M-row population.

---

## 🔍 Key EDA Findings

### 🧬 The "Burnout" Catalyst
Our analysis identifies that dropout risk is not just about grades:
*   **Strongest Correlations:** `Burnout Score` (+0.69) and `Depression Score` (+0.65) are the primary drivers of attrition.
*   **Multicollinearity:** Detected severe overlap between `Stress Level` and `Mental Health Index` ($r = -0.95$), leading to strategic feature dropping to prevent model overfitting.
*   **Noise Identification:** Demographic factors like `Age` and `Academic Year` showed near-zero correlation with risk levels.

---

## 🤖 Regression Modeling & Hyperparameter Tuning

The regression phase transforms EDA insights into a predictive framework using advanced Gradient Boosting

### **Preprocessing & Feature Engineering**
*   **Categorical Encoding:** Implementation of One-Hot Encoding for low-cardinality features.
*   **Scaling:** Robust Scaling to handle potential outliers in psychological scores.
*   **Feature Selection:** Dropping redundant variables identified during the EDA (e.g., Internet Usage, Age).

### **Model Performance & Results**
We utilized a tuned **LGBMRegressor** (LightGBM) within a structured pipeline. The results demonstrate a highly robust model that generalizes well to unseen data:

*   **Best Parameters:** 
    *   `learning_rate`: 0.03, `n_estimators`: 1000, `max_depth`: 6
    *   `reg_lambda`: 1.0, `reg_alpha`: 1.0, `num_leaves`: 31
*   **Best CV $R^2$ Score:** 0.5859
*   **Total Processing Time:** 8 min 3.66 sec
*   **Train $R^2$ Score:** 0.5943
*   **Test $R^2$ Score:** 0.5866
*   **MAE (Mean Absolute Error):** 0.6666

---

### **💡 Why is an $R^2$ of ~59% "Good" in this Context?**
In psychological and behavioral datasets, achieving a 99% accuracy is often a **red flag** rather than a success. Here is why our result is scientifically sound:

1.  **Human Subject Variability:** Unlike physics or chemistry, human behavior (burnout and dropout risk) is influenced by thousands of unmeasured external factors. A 59% variance explanation is highly significant in social sciences.
2.  **Prevention of Overfitting:** The minimal gap between Train (0.594) and Test (0.586) scores proves that the model has **not memorized noise**. It has learned generalizable patterns.
3.  **The "Complexity" Reality:** If a model reached 99%, it would likely indicate **Data Leakage** (using features that are effectively the target itself) or an unrealistic lack of "stochastic noise" in the 1M records.
4.  **Operational Value:** Even with a 0.66 MAE, the model can accurately categorize students into "Low", "Medium", and "High" risk tiers, providing actionable insights for institutional intervention.
---

## 💻 Tech Stack & Tools

*   **Analysis:** Pandas, NumPy, Scipy
*   **Visualization:** Plotly (Interactive), Seaborn, Matplotlib
*   **Modeling:** Scikit-Learn(RandomizedSearchCV,ColumnTransformer, TransformedTargetRegressor and Pipeline) and lightgmb
*   **Deployment-Ready:** Modular code blocks and environment-aware path management.

---

## 📊 Dataset Reference
**Data Source:** [Student Mental Health & Burnout Dataset (Kaggle)](https://www.kaggle.com/datasets/ayeshasiddiqa123/student-health)

---

## 🚀 How to Use
1.  **Clone the Repo:** `git clone https://github.com/BetulKarakaya/Student-Mental-Health-Regression.git`
2.  **Install Dependencies:** `pip install -r requirements.txt`
3.  **Run EDA:** Start with `student-mental-health-eda.ipynb` to understand the data landscape.
4.  **Run Model:** Execute `student-mental-health-model.py` for risk prediction.

