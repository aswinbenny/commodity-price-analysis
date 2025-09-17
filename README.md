# Commodity Price Analysis — Ernakulam, Kerala

Analyzing seasonal price dynamics of agricultural commodities using statistical and machine learning models.  
This project explores daily market price data (sourced from [data.gov.in](https://data.gov.in) and refined into a [clean Kaggle dataset](https://www.kaggle.com/datasets/aswinbenny213/daily-market-prices-of-kerala-commodities)) for commodities traded in Ernakulam, Kerala.  
The repository includes reproducible notebooks, utility functions, and modeling workflows (SARIMAX, Prophet, LightGBM) to understand seasonal trends and build predictive baselines.

## Overview

This project investigates the **seasonal and temporal dynamics of commodity prices** in Kerala, with a focus on Ernakulam district.  
The analysis covers how **season, market, year, commodity, and variety** affect prices, and applies statistical and machine learning models for forecasting.

The repository contains **reproducible notebooks** and utility modules, organized to progressively take a user from exploratory analysis to predictive modeling:

## Visualization

The **Kerala Commodity Market Explorer** dashboard (built in Tableau) provides an interactive view of the analysis.  
It includes:  

- Key price drivers (effect size of market, season, commodity, year, variety)  
- Market-level price map for Ernakulam  
- Time series of commodity price trends  
- Seasonal price distribution (Post Monsoon, SW Monsoon, Summer, Winter)  
- Summary metrics and interactive filters (Market, Product Type, Year)

[Explore the Tableau Dashboard](https://public.tableau.com/views/Keralaagriculturalcommoditymarketanalysis/KeralaCommodityMarketExplorer?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## Notebooks

1. **EDA (`eda.ipynb`)**  
   - Exploratory data analysis, boxplots, heatmaps.  
   - ANOVA, Foneway tests, and multiple testing to understand how season, market, year, and commodity variety affect prices.

2. **Statistical Analysis (`statistical_analysis.ipynb`)**  
   - Computes effect sizes (Eta-squared, Omega-squared) to quantify impact of factors.  
   - Compares statistical measures across markets and commodities.

3. **Time Series Modeling (`time_series_modeling.ipynb`)**  
   - Focuses on temporal dynamics and preparing representative, high-volume exemplars from grouped features.  
   - Builds forecasting pipelines.

4. **SARIMAX Modeling (`sarimax_onion_perumbavoor.ipynb`)**  
   - Forecasting Onion (FAQ variety) prices in Perumbavoor market.  
   - Steps: data prep → train/val/test split → stationarity check → SARIMAX fitting → forecast evaluation.

5. **Prophet Modeling (`prophet_modeling.ipynb`)**  
   - Time series forecasting using Facebook Prophet.  
   - Handles seasonality, trends, and outliers for commodity prices.  
   - Steps: data prep → filtering/formatting → train/test split → model fit → forecast diagnostics.

6. **LightGBM Hyperparameter Tuning (`lightgbm_optuna.ipynb`)**  
   - Performs Optuna-based hyperparameter optimization for predicting `log_Modal_Price_filled`.  
   - Focuses on seasonal, historical, and market features.

7. **LightGBM Final Model (`lightgbm_final.ipynb`)**  
   - Trains the final LightGBM model using tuned hyperparameters.  
   - Steps: dataset preparation → time-based split → feature/target setup → model training → validation RMSE computation.

## Utilities (`utils/`)

The `utils/` directory contains reusable Python modules that support data preprocessing, analysis, and modeling workflows. Key modules include:

1. **`data_splitter.py`**  
   - Class `DataSplitter`: splits datasets into train, validation, and test sets.  
   - Filters out product-market groups with insufficient data and optionally drops helper columns.

2. **`effect_sizes.py`**  
   - Functions to compute and compare **effect sizes** (eta-squared, omega-squared) across product-market or commodity groups:  
     - `compute_effect_sizes_by_group()`  
     - `compute_effect_sizes_by_commodity()`  
     - `compare_eta_omega()`

3. **`exemplar_analysis.py`**  
   - Functions for selecting **representative and high-volume exemplars** of feature groups.  
   - `select_exemplars()`, `time_series_extractor()`, `plot_time_series()` — supports time series extraction and plotting.

4. **`load_data.py`**  
   - `load_data(file_path)`: loads and preprocesses raw commodity price CSVs for EDA and modeling.

5. **`model_filter.py`**  
   - `filter_for_prophet()`: filters train, validation, and test sets to ensure sufficient history for Prophet modeling.

6. **`save_market_location_csv.py`**  
   - Generates a CSV mapping markets to geographic coordinates.

7. **`wrangle_model_data.py`**  
   - `wrangle_ml(df)`: prepares data for machine learning (LightGBM) including feature creation, date parsing, missing value handling, lag/rolling features, and log transformation.  
   - `assign_season(date)`: maps a date to a season.

8. **`wrangle.py`**  
   - `wrangle(df)`: initial data wrangling for EDA and modeling, creates composite product features, filters low-volume pairs, adds seasonal and VFPCK flags, and log-transforms prices.  
   - `assign_season(date)`: season mapping utility.

## Data Filtering & Preprocessing Decisions

1. **Minimum Record Cutoff (≥50 arrival records per Product_Type × Market combination)**  
   - Before filtering: 583 unique combinations  
   - After filtering: 402 combinations  
   - Dropped: 181 combinations (31%)  
   - Rows remaining: 166,028  
   **Reason:** Combinations with fewer than 50 records are too sparse to support stable correlation or variance analysis.

2. **Exclusion of Rows with No Significant Features**  
   - Identified by: `important_features == "None"`  
   - Count: 110 rows removed out of 166,028 (0.066%)  
   - Rows remaining after both filters: 165,918  
   **Reason:** These rows had negligible explanatory power (all eta² and omega² near 0) and were excluded to focus on meaningful feature effects.

## Modeling Overview & Key Findings

- A **holdout test set** was kept untouched during training and hyperparameter tuning, ensuring unbiased evaluation for future models.  

- **Prophet Models**  
  - Trained individually for each Product_Type × Market pair  
  - Mean RMSE on validation: 0.235  
  - Effective for highly seasonal pairs, but requires per-pair training.

- **Global LightGBM Model**  
  - Single model trained across all products and markets  
  - Validation RMSE: 0.110  
  - Outperformed Prophet overall, capturing complex interactions and providing a single, generalizable model.

## Data & Environment Setup

### Dataset

- The notebooks use a **refined, cleaned dataset** uploaded to Kaggle:  
  [Daily Market Prices of Kerala Commodities](https://www.kaggle.com/datasets/aswinbenny213/daily-market-prices-of-kerala-commodities)  
- This dataset is already wrangled and ready for EDA, statistical analysis, and modeling.  
- **Placement:** Download and place the CSV in the `data/` folder
- The notebooks expect the cleaned CSV; no raw data preprocessing is needed.  
- The dataset can also be passed through `wrangle_ml()` for ML modeling features if running LightGBM.

### Python Environment

This project was developed inside a **Python virtual environment**. To replicate the environment:

```bash
# create virtualenv
python3 -m venv .venv

# activate virtualenv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (PowerShell/CMD)

# upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
 ```

 ## Future Work

- Explore additional modeling approaches such as **mixed-effects models** or deep learning methods to better capture hierarchical and temporal patterns.  
- Extend forecasting to other commodities and markets across Kerala.  
- Use insights to support **farmers, traders, and government agencies** in price prediction, planning, and decision-making.  
- Incorporate external features (weather, festivals, demand data) for richer predictive modeling.

## License

This project is released under the **MIT License** 

## Author

Developed by **Aswin Benny**  
