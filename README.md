Kerala Commodity Price Analysis

This project is under development. We are exploring seasonal trends in commodity prices across Ernakulam district, Kerala, with a focus on the role of VFPCK markets.

More details, visuals, and insights will be added as the analysis progresses.

Data Filtering Decisions and Impact
	1.	Minimum Record Cutoff (≥50 arrival records)
	•	Before filtering: 583 unique Product_Type × Market combinations
	•	After filtering: 402 combinations
	•	Dropped: 181 combinations (31%)
	•	Rows remaining: 166,028
Reason: Combinations with fewer than 50 records are too sparse to support stable correlation or variance analysis.
	2.	Exclusion of Rows with No Significant Features
	•	Identified by: important_features == "None"
	•	Count: 110 rows removed out of 166,028 (0.066%)
	•	Rows remaining after both filters: 165,918
	•	Interpretation: These rows had negligible explanatory power (all eta² and omega² near 0).
	•	Decision: Excluded to focus on meaningful feature effects.


We kept a holdout test set untouched during training and hyperparameter tuning. This ensures an unbiased evaluation for future models. New models can be trained and compared on the validation set, and final performance can be assessed on the reserved test set if needed.

We evaluated two modeling approaches for predicting log modal prices: Prophet models trained individually for each (Product_Type, Market) pair, and a single LightGBM model trained across all products and markets. For Prophet, RMSE was computed per pair and summarized using the mean, which was 0.235. LightGBM’s overall RMSE on the same validation set was 0.110. While Prophet is effective for highly seasonal pairs due to its specialized per-pair modeling, the LightGBM model outperformed Prophet overall, providing a single model that generalizes across all products and markets and captures complex interactions. This indicates that, for this dataset, a global LightGBM model offers better predictive performance than individually trained Prophet models.