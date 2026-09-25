# Energy Consumption Project � Decision Log

This file records important analytical and implementation decisions that may affect interpretation of the final results.

| ID | Date | Decision | Reason | Impact |
|----|------|----------|--------|--------|
| D001 | | Separate exploratory notebooks from reproducible source code | Prevent critical preprocessing or modeling steps from depending on manually executed notebook cells | Improves reproducibility |
| D002 | | Preserve original source data and write transformed datasets to data/processed | Prevent accidental modification of source datasets | Allows the analysis to be reproduced from original data |

### Observed Final Test Results

The locked Extra Trees model was refit using the combined training and validation observations and then evaluated once on the previously untouched chronological test partition.

The final test partition contained 468 observations.

Final test performance was:

| Model | MAE | CVRMSE (%) | MAPE (%) | MAPE n |
|---|---:|---:|---:|---:|
| Extra Trees Final | 182.259 | 52.608 | 29.991 | 468 |

All 468 test observations were included in the MAPE calculation.

The final test errors were slightly lower than the tuned validation errors. Validation MAE decreased from 187.837 to 182.259, CVRMSE decreased from 55.822% to 52.608%, and MAPE decreased from 31.826% to 29.991%.

This result was obtained after model and hyperparameter selection had already been completed. No additional tuning or model-selection decisions were made using the test results.

Across the test period, mean actual consumption was approximately 581.45, while mean predicted consumption was approximately 623.10. The mean residual, calculated as actual minus predicted, was approximately -41.65, indicating a modest overall tendency toward overprediction.

The median absolute prediction error was approximately 97.74, although some substantially larger errors remained.

| D017 | 2026-09-24 | Treat the Extra Trees test results as the final UCI supervised-forecast evaluation and make no additional model-selection changes based on the test set | The model architecture and hyperparameters were locked before the final test partition was evaluated | Preserves the test partition as an independent estimate of final prototype forecasting performance |

| D018 | 2026-09-24 | Standardize all K-Means clustering variables before fitting | K-Means depends on Euclidean distance, so variables measured on larger numerical scales could otherwise dominate the cluster assignments | Clustering is performed on standardized features while cluster profiles are reported in the original measurement scales |

| D019 | 2026-09-24 | Evaluate K-Means solutions from k = 2 through k = 8 and select the solution with the highest silhouette score | The project did not specify a fixed number of consumption-pattern clusters, so an explicit data-driven cluster-selection criterion was required | The final UCI cluster count is selected reproducibly rather than chosen arbitrarily |

| D020 | 2026-09-24 | Use a focused set of energy, outdoor-weather, and temporal variables for UCI K-Means clustering rather than all forecasting predictors | Lagged and rolling variables were engineered for supervised forecasting, while numerous correlated indoor sensor variables could dominate distance calculations and reduce cluster interpretability | Clusters are intended to represent interpretable recurring hourly energy and environmental operating patterns |

| D018 | 2026-09-24 | Standardize all K-Means clustering variables before fitting | K-Means depends on Euclidean distance, so variables measured on larger numerical scales could otherwise dominate the cluster assignments | Clustering is performed on standardized features while cluster profiles are reported in the original measurement scales |

| D019 | 2026-09-24 | Evaluate K-Means solutions from k = 2 through k = 8 and select the solution with the highest silhouette score | The project did not specify a fixed number of consumption-pattern clusters, so an explicit data-driven cluster-selection criterion was required | The final UCI cluster count is selected reproducibly rather than chosen arbitrarily |

| D020 | 2026-09-24 | Use a focused set of energy, outdoor-weather, and temporal variables for UCI K-Means clustering rather than all forecasting predictors | Lagged and rolling variables were engineered for supervised forecasting, while numerous correlated indoor sensor variables could dominate distance calculations and reduce cluster interpretability | Clusters are intended to represent interpretable recurring hourly energy and environmental operating patterns |