# UNP_ML_REGRESSION_PROJECT_APPLIANCES_ENERGY_PREDICTION_USING_PYTHON
This project was completed as part of the United Network of Professionals (UNP) Machine Learning Academic Internship Program.

# ⚡ Appliances Energy Prediction: Regression with Machine Learning and ANN

An end-to-end machine learning project that predicts the **electricity consumption of household appliances (in Wh)** from indoor temperature and humidity sensors, outdoor weather conditions and time-based features. It covers data checks, feature engineering, EDA, multicollinearity removal with VIF, a comparison of six regression models across four train/test splits, a Random Forest hyperparameter check, 5-fold cross-validation and feature importance.

---

## 📌 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Key EDA Insights](#-key-eda-insights)
- [Models and Results](#-models-and-results)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Repository Structure](#-repository-structure)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Problem Statement
Appliances account for a large share of household electricity use. Understanding which environmental and time-related factors drive that use helps with energy planning and smart-home optimisation. The goal is to build a regression model that predicts appliance energy consumption and to identify the factors that influence it most.

- **Task:** Regression
- **Target:** `Appliances` (energy use in Wh)

## 📂 Dataset
- **Source:** Appliances Energy Prediction dataset (UCI Machine Learning Repository), 10-minute readings from a low-energy house
- **Size:** 19,735 rows × 29 columns
- **Period:** 11 Jan 2016 to 27 May 2016
- **Data quality:** no missing values and no duplicate rows

| Group | Features |
|-------|----------|
| Indoor temperature | `T1` to `T9` (°C, different rooms) |
| Indoor humidity | `RH_1` to `RH_9` (%) |
| Outdoor weather | `T_out`, `RH_out`, `Press_mm_hg`, `Windspeed`, `Visibility`, `Tdewpoint` |
| Other | `lights` (lighting energy, Wh), `rv1`, `rv2` (random variables), `date` |

## 🔄 Project Workflow

### 1. Data Preparation
- Converted `date` to datetime and checked nulls, duplicates and data types
- **Feature engineering** from the timestamp: `hour`, `day_of_week`, `month`, `is_weekend`
- Flagged outliers in the target with the IQR rule: limits −25 to 175 Wh, **2,138 outliers** (kept in the data)

### 2. Exploratory Data Analysis
Questions explored with plots:
1. Does usage vary by hour of day?
2. Does it differ between weekdays and weekends?
3. Does it vary by month?
4. How do indoor humidity and indoor temperature ranges relate to usage?
5. Does lighting use (an occupancy proxy) relate to usage?

Plus histograms of all variables and a correlation heatmap.

### 3. Multicollinearity Treatment (VIF)
- Dropped `rv2` (an exact duplicate of `rv1`) and `Tdewpoint` (derived from `T_out` and `RH_out`)
- Iteratively removed features with VIF above 10: `T_out`, `T9`, `T2`, `RH_4`, `T7`, `T1`, `RH_6`
- Final feature set of **22 features**: `lights`, `RH_1`, `RH_2`, `RH_3`, `RH_5`, `RH_7`, `RH_8`, `RH_9`, `T3`, `T4`, `T5`, `T6`, `T8`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `rv1`, `hour`, `day_of_week`, `month`, `is_weekend`

### 4. Modelling
- Six models compared on four splits (60/40, 70/30, 75/25, 80/20) with `StandardScaler` fitted on the training data only
- Hyperparameter checks for Random Forest and KNN
- 5-fold cross-validation (scaler and model in a single `Pipeline`) to confirm the best model

## 🔍 Key EDA Insights
- **Time of day matters most:** `hour` has the highest correlation with appliance use (0.22).
- **Lights:** `lights` and the lights-on flag correlate positively with usage (about 0.20).
- **Outdoor humidity:** `RH_out` has the strongest negative correlation (−0.15). Several indoor humidity sensors (`RH_8`, `RH_6`) are also negative.
- **Weak linear signals:** most variables have low linear correlation with the target (|r| < 0.15), which hints that non-linear models will do better.
- **Skewed target:** usage is right-skewed (mean about 98 Wh, median 60 Wh, max 1,080 Wh), with many high-consumption spikes.
- **Redundant sensors:** many temperature and humidity features are strongly collinear, which is why VIF filtering was needed.

## 🤖 Models and Results

Metrics at the **80/20 split**:

| Model | R² | MAE | RMSE | MAPE |
|-------|----|-----|------|------|
| **Random Forest** | **0.5445** | **32.46** | **67.51** | 32.94% |
| KNN (k=5) | 0.3757 | 38.14 | 79.04 | 37.78% |
| Gradient Boosting | 0.3071 | 45.35 | 83.27 | 51.68% |
| ANN (best: 32-16, Adam, batch 32) | 0.2792 | 46.68 | 84.93 | 52.22% |
| Decision Tree | 0.1997 | 37.13 | 89.49 | 33.36% |
| Linear Regression | 0.1519 | 53.37 | 92.12 | 63.67% |
| SVR (RBF) | 0.0436 | 41.65 | 97.83 | 31.79% |

**Random Forest across splits (R²):** 60/40: 0.479 · 70/30: 0.520 · 75/25: 0.517 · 80/20: 0.545

**Hyperparameter checks**
- Random Forest `n_estimators` 50 / 100 / 200 gave R² of 0.538 / 0.545 / 0.546, so more trees add very little.
- KNN `k` = 3 / 5 / 7 / 10 gave R² of 0.337 / 0.347 / 0.353 / 0.331.

**5-fold cross-validation (Random Forest):** mean R² = **0.534**, std = **0.009**, so the result is stable and not an artefact of one lucky split.

**ANN grid search:** 12 configurations (hidden units `(8,4)`, `(16,8)`, `(32,16)` × Adam/SGD × batch 32/64, with early stopping on validation loss). Adam performed far better than SGD, whose runs stopped early and scored an R² of about 0. The R² of Adam runs ranged from 0.196 to 0.279, and the largest network (32-16) did best.

### 📝 Takeaways
- **Random Forest is the best model** (R² ≈ 0.54, MAE ≈ 32 Wh), well ahead of linear, boosting and neural-network models. This suggests the relationships are strongly non-linear.
- Linear Regression explains only about 15% of the variance, which confirms that a simple linear relationship is not enough.
- Appliance use is inherently noisy and spiky, so a moderate R² is expected from sensor data alone.
- The SVR has a low MAPE but near-zero R². It predicts values close to the typical low levels and misses the spikes.

## 🛠 Tech Stack
**Language:** Python
**Data:** Pandas, NumPy
**Visualization:** Matplotlib, Seaborn
**Statistics:** Statsmodels (VIF)
**ML:** Scikit-learn (Linear Regression, KNN, Decision Tree, Random Forest, SVR, Gradient Boosting)
**Deep Learning:** TensorFlow / Keras
**Environment:** Jupyter Notebook / Google Colab

## ▶️ How to Run
1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Install dependencies
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow
   ```
3. Place the dataset (`dataset1.csv`) in the project folder. The notebook uses Google Colab's file upload (`files.upload()`), so when running locally replace that cell with `pd.read_csv("dataset1.csv")`.
4. Open and run `Another_copy_of_appl.ipynb` from top to bottom.

## 📁 Repository Structure
```
├── Another_copy_of_appl.ipynb   # Full analysis and modelling
├── dataset1.csv                 # Dataset (add locally)
└── README.md
```

## 🚀 Future Improvements
- Tune Random Forest and boosting models with GridSearchCV / Optuna
- Try XGBoost or LightGBM
- Use time-aware splitting (train on earlier dates, test on later ones), since the data is a time series
- Add lag features (previous readings) and rolling averages
- Transform the skewed target (for example log) or handle the spikes explicitly
- Use SHAP for deeper interpretability

Code Link:https://drive.google.com/file/d/1OmaDNTRiRi4BPWHfUhV7ntGMqmHCTL_z/view?usp=sharing
PPT Link:https://docs.google.com/presentation/d/1hxTiDWFg048zns0g1PyJxZN39fH3a-IG/edit?usp=sharing&ouid=115260607757946087302&rtpof=true&sd=true
