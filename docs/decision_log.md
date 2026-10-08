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

| D021 | 2026-09-24 | Retain the three-cluster K-Means solution for the UCI prototype | k = 3 produced the highest silhouette score among the evaluated solutions from k = 2 through k = 8 | The three clusters are interpreted as broad recurring consumption/environmental patterns rather than sharply separated categories |

| D022 | 2026-09-29 | Filter the primary NREL ResStock modeling population to California using `in.state = CA` while retaining `bldg_id` as the link to load-profile data | The project is narrowing its primary modeling scope to California, while NREL metadata provides explicit state and building identifiers | Subsequent ResStock load-profile acquisition and forecasting will focus on the California residential subset |

| D023 | 2026-10-07 | Use a pooled building-hour modeling design for the primary NREL ResStock analysis | Pooling preserves building-level variation while allowing a single model to learn across many California residential buildings and supports evaluation on unseen buildings | Each modeling row will represent one building at one timestamp, with time-series features combined with selected building metadata |

| D024 | 2026-10-07 | Use the ResStock baseline scenario (`upgrade=0`) for the primary forecasting analysis | Mixing baseline and retrofit scenarios could introduce intervention-driven differences unrelated to the forecasting task | NREL profile acquisition will use `timeseries_individual_buildings/by_state/upgrade=0/state=CA/` |

| D025 | 2026-10-07 | Evaluate both temporal generalization and building generalization in the pooled NREL workflow | A pooled model should demonstrate both the ability to forecast future observations and the ability to generalize to residential buildings not seen during training | Development buildings will be evaluated using time-respecting splits, while a separate group of building IDs will remain completely held out |

| D026 | 2026-10-07 | Use a reproducible 25-building California ResStock pilot before scaling the NREL modeling dataset | Individual profile files vary in size and the time-series schema, temporal resolution, storage requirements, and preprocessing requirements should be validated before downloading a much larger population | The pilot uses random_state=42 and is for pipeline validation only; the final modeling population will be selected after pilot inspection |

| D027 | 2026-10-07 | Validate NREL acquisition with a reproducible 25-building California pilot before scaling | This confirms that metadata building IDs map correctly to individual profile files and allows storage/schema requirements to be assessed before downloading a much larger sample | All 25 pilot profiles downloaded successfully; the next step is profile schema inspection |

| D028 | 2026-10-07 | Use 250 California ResStock buildings for the primary NREL modeling population, with 200 development buildings and 50 completely held-out buildings | The 25-building pilot demonstrated substantial building-level heterogeneity, while scaling estimates indicate that 250 buildings will provide approximately 2.19 million hourly observations without making ensemble-model tuning unnecessarily expensive | The final population will be selected reproducibly using building-characteristic stratification and random_state=42 |

| ID | Decision | Rationale |
|---|---|---|
| D028 | Use 250 California ResStock buildings for the primary NREL modeling sample | Provides approximately 2.19 million hourly observations while keeping ensemble-model training and tuning computationally manageable |
| D029 | Split the building sample into 200 development buildings and 50 completely held-out buildings | Supports both temporal forecasting evaluation and building-generalization evaluation |
| D030 | Stratify the final building sample by building type and CEC climate zone | This produced close agreement with the full California population while avoiding overly sparse multi-variable strata |
| D031 | Treat floor-area group and vintage as representativeness diagnostics rather than hard stratifiers | Including them in the hard strata created too many sparse combinations for a 250-building sample |
| D032 | Exclude annual-energy outputs and QOI fields from the static predictor set | These variables are derived from the same simulated outcomes being forecast and could introduce target leakage |
| D033 | Use county-level weather files matched through in.county | The ResStock weather archive uses county-style GISJOIN identifiers that correspond to the building metadata geography |
| D034 | Use county-level NREL AMY2018 weather matched through `in.county` | The ResStock weather archive stores hourly weather CSVs using county GISJOIN identifiers that match the `in.county` field already present in the modeling metadata | This provides geographically consistent dynamic weather features without requiring external weather sources |
| D035 | Exclude wind direction from the NREL forecasting predictor set | Quality control found two out-of-range wind-direction values of 801 degrees among 359,160 county-hour weather observations. Because the source meaning of 801 could not be established and wind direction is not essential to the forecasting design, the raw field will be retained for provenance but excluded from model predictors rather than imputed arbitrarily. |
| D036 | Split NREL data by unique target timestamps rather than raw row order | All buildings share the same hourly timeline, so row-wise splitting could place the same forecast hour in multiple partitions and create temporal leakage |
| D037 | Reserve all 50 held-out buildings for evaluation only | This allows separate measurement of temporal generalization on familiar buildings and cross-building generalization on unseen buildings |
| D038 | Evaluate held-out buildings over the same final test window as the temporal test set | This makes familiar-vs-unseen building performance directly comparable over the same weather and seasonal period |
| D039 | Use Seasonal Naive as the primary NREL forecasting benchmark | It achieved the lowest validation MAE, RMSE, CVRMSE, and MAPE among the tested non-learned baselines |
| D040 | Use Histogram Gradient Boosting instead of classic GradientBoostingRegressor for the NREL phase | The NREL training set contains more than 1.2 million observations, making histogram-based boosting substantially more computationally practical |
| D041 | Advance Random Forest, Extra Trees, and Histogram Gradient Boosting to tuning | All three nonlinear models outperformed the Seasonal Naive benchmark on validation MAE, while Extra Trees and Random Forest were nearly tied for best performance and Histogram Gradient Boosting offered much faster training |
| D042 | Use three expanding timestamp-aware CV folds for NREL tree tuning | The pooled NREL dataset is large and all buildings share timestamps, so folds were defined over unique target timestamps to preserve chronology while keeping tuning computationally tractable |
| D043 | Select tuned Extra Trees as the primary NREL supervised model | Extra Trees achieved the lowest temporal-CV MAE and the lowest untouched validation MAE among the tuned nonlinear models |
| D044 | Allow held-out buildings to use their own observed historical load for lag and rolling predictors | The evaluation tests cross-building model-parameter generalization for one-hour-ahead forecasting, not zero-history cold-start forecasting; held-out building IDs and outcomes are never used for model fitting |
| D045 | Lock the tuned Extra Trees configuration after validation and make no further test-driven adjustments | The temporal and held-out test sets are reserved for final unbiased evaluation and must not be used for additional tuning |
| D046 | Retain k=4 for NREL daily load-shape clustering | The four-cluster solution achieved the highest silhouette score (0.291783) among k=2 through k=8 while avoiding the increasingly small clusters produced at larger k values |