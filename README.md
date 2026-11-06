# Titanic: Machine Learning from Disaster

**this was the starting of the end**

My first end-to-end run at a Kaggle competition: predict which passengers survived the Titanic.
It is old, it is messy, and I'm keeping it as it was.

## Data

The Kaggle data is **not included** (the competition rules don't allow redistributing it).
Download `train.csv` and `test.csv` from the competition page and put them next to the notebooks:
<https://www.kaggle.com/competitions/titanic/data>

## What's here

| File | What it is |
|---|---|
| `operation_v1.ipynb` | First look at the data: stats, missing values, distributions, outliers |
| `EDA.ipynb` | The fuller 9-phase EDA (missing values, class imbalance, univariate, bivariate, correlation, multivariate, outliers, feature engineering). Plots are saved in `tmp/` |
| `modeling_v1.ipynb` | Feature engineering, logistic regression baseline, tuned XGBoost, then a v2 with a Random Forest + voting ensemble. Writes the Kaggle submission files |
| `intution.txt`, `intution_v2.txt` | My running notes while working through it |
| `submission_*.csv` | The files I uploaded to Kaggle |

## Approach

- **Missing values:** age is imputed with the median, embarked with the most frequent value, and cabin is not imputed.
- **Features built:** `Title` (from the name, rare titles grouped), `FamilySize`, `IsAlone`, `HasCabin`, and `Fare_log` (log of fare, because fare is skewed).
- **Dropped:** `PassengerId`, `Name`, `Ticket`, `Cabin`, `Fare`, `SibSp`, `Parch`.
- **Pipeline:** scikit-learn `ColumnTransformer`. Numeric columns use median impute + standard scaling; categorical columns use most-frequent impute + one-hot encoding.
- **Imbalance:** about 62/38 died/survived, so I judged on ROC-AUC and F1 instead of accuracy.
- **Models:** logistic regression, XGBoost and Random Forest (both tuned with `RandomizedSearchCV`, 50 iterations x 5 folds), and a soft-voting ensemble of the three.
- **Split:** stratified 80/20 on `train.csv` (712 train / 179 held out).

## Results

On my own 179-passenger hold-out:

| Model | CV AUC (train) | Test AUC | Test accuracy |
|---|---|---|---|
| Logistic regression (default) | 0.875 | 0.865 | 0.83 |
| XGBoost (tuned) | 0.897 | 0.850 | 0.80 |
| Random Forest (tuned) | 0.887 | 0.849 | 0.80 |
| Voting ensemble (XGB + RF + LR) | n/a | 0.856 | 0.83 |

**Kaggle score:** the first XGBoost submission scored **0.76** (rank 10559). I did not record a score for
the voting-ensemble submissions.

Takeaways:
- The plain logistic regression beat every tuned model on AUC. With only 179 test rows these gaps are small enough to be noise, but it was a good lesson that tuning isn't automatically better.
- The hold-out accuracy (0.80-0.83) is much higher than the Kaggle score (0.76).

## Notes

- The saved notebook outputs come from a clean top-to-bottom re-run. The `submission_*.csv` files are the ones I
  actually uploaded to Kaggle, from my original run, so re-running gives slightly different submissions
  (the Random Forest search picks different settings, and the XGBoost threshold moves from 0.39 to 0.40).
- The EDA is not all my own work. The modelling is.

## Running it

```bash
pip install pandas numpy scipy scikit-learn xgboost matplotlib seaborn missingno joblib python-dotenv jupyterlab
```

1. Download `train.csv` and `test.csv` (see Data above).
2. Optional: `cp .env.example .env` and set `DATA_DIR`, `PLOTS_DIR` and `MODELS_DIR` for your machine.
   `.env` is gitignored. If you skip it, the notebooks look for `train.csv` / `test.csv` in the folder you start
   Jupyter from and write plots to `tmp/` and models to `tmp_models/`.
3. Start Jupyter from this folder and run `operation_v1` -> `EDA` -> `modeling_v1`.
