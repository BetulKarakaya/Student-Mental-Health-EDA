# 🎓 Student Mental Health & Dropout Risk Analysis

This project provides a comprehensive **Exploratory Data Analysis (EDA)** on a large-scale dataset (1 million records) investigating the factors that contribute to student mental health issues and dropout risks. Using advanced visualization and statistical methods, the analysis uncovers the critical intersection between psychological burnout and academic persistence.

## 📊 Dataset Overview
The dataset used in this analysis explores various dimensions of student life, including academic performance, psychological metrics (stress, burnout, depression), and socio-economic factors.

**Data Source:** [Student Mental Health & Burnout Dataset (Kaggle)](https://www.kaggle.com/datasets/ayeshasiddiqa123/student-health)

---

## 🚀 Key Features of the Analysis
- **Robust Data Loading:** Integrated `try-except` blocks with environment-aware path management for fail-safe data ingestion.
- **Advanced Visualization:** High-quality interactive plots using `Plotly` and detailed statistical distributions with `Seaborn`.
- **Feature Engineering & Insights:** Detection of multicollinearity and identification of "noise" vs. "signal" variables.
- **Professional Coding Standards:** Senior-level modular code, comprehensive missing value analysis, and deep-dive correlation matrices.

---

## 🔍 Key Findings & Insights

### 1. The Real Drivers of Attrition
Contrary to traditional beliefs, academic performance is not the primary predictor of dropout risk. Our analysis shows that:
* **Mental Health is Decisive:** `Burnout Score` (+0.69) and `Depression Score` (+0.65) show the strongest positive correlation with dropout risk.
* **The Economic Factor:** `Financial Stress` (+0.58) acts as a significant catalyst, often exacerbating psychological distress.
* **Protective Factors:** A high `Mental Health Index` and strong `Social Support` are the most effective buffers against attrition.

### 2. Redundant Features (Noise)
The data reveals that `age`, `academic_year`, and `screen_time` have near-zero correlation with the target variable, suggesting that student risk profiles are independent of these demographics.

### 3. Feature Redundancy
Significant multicollinearity was detected between:
* `Stress Level` & `Mental Health Index` (-0.95)
* `Burnout Score` & `Depression Score` (+0.82)
* `Screen Time` & `Internet Usage` (+0.89)

---

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Plotly, Seaborn, Matplotlib, Scipy
- **Environment:** Jupyter Notebook / Kaggle / Google Colab

---

## 📂 Project Structure
```text
├── studient-mental-health-eda.ipynb  # Main analysis notebook
├── README.md                          # Project documentation
└── data/                              # Data directory (external link provided)
