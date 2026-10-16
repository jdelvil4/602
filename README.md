# Optimizing Energy Usage with Machine Learning

The project involves the use of machine learning techniques in order to predict building electricity consumption one hour in advance, based on both simulated and real energy data.

The study uses data from both homes and offices, including information on weather conditions, time-related features, previous electricity consumption, unsupervised clustering, and details about the California power grid.

A main aim is not just to check how accurate the predictions are, but to see how well the models can apply to different situations:

1. Is it possible for a model to predict the electricity usage of buildings which were included in the training set?

2. Would this approach be applicable to buildings that have never been seen before?

3. Can a model which has been trained on data relating to offices actually handle real electricity data from offices?

The project involves the use of Python and Scikit-Learn, with a time-based splitting method, and cross-validation that is time-aware and which tests on buildings not seen during training.
The project involves the use of Python and Scikit-Learn, with a time-based splitting method, cross-validation that is time-aware and which tests on buildings not seen during training, and full documentation of the modeling decisions.

---

## Key Results

The findings indicate that tree-based models generally assist with predictions made one hour in advance; the degree of their performance is very much dependent on the kind of generalization under consideration.

| Dataset | Final Model | Evaluation | Model MAE | Seasonal Naive MAE | MAE Improvement |

|---|---|---|---:|---:|---:|

| NREL ResStock | Extra Trees | test | 0.2261, 0.2952 | 23.4% |

| NREL ResStock | Extra Trees | Held-out buildings | 0.2455 0.2844 | 13.7% |

| NREL ComStock | Random Forest | Temporal test | 3.0919 8.6002 | 64.0% |

| NREL ComStock | Random Forest | Held-out buildings | 15.7418 | 13.5190 | -16.4% |

| BDG2 Random Forest | Temporal test | 7.7670 | 11.8266 | 34.3% |

| BDG2 Random Forest | Number of buildings in the held-out set | 8.6491 | 9.4809 | 8.8% |

The model that was best for ComStock performed well when making predictions about buildings it had previously seen, but its MAE was higher than that of the Seasonal Naive model when it came to completely new commercial buildings.

Even though the improvement was smaller than in the case of the time-based test, the actual data model from BDG2 still functioned for buildings.

### Simulated-to-Real Transfer

The experiment which caused the model to go from data to real data yielded the most clear results.

| Evaluation | Model | MAE | CVRMSE |

|---|---|---:|---:|

BDG2 Temporal Test | Transfer of ComStock to BDG2 RF | 16.2295 | 30.685%

BDG2 Temporal Test | BDG2 Internal RF | 7.7670 | 15.242%

| BDG2 Temporal Test | Naive | 11.8266 | 26.292% |

| Held-Out Buildings BDG2 | Percentage of ComStock-to-BDG2 Transfer RF | 12.8735 | 25.024% |

| BDG2 Held-Out Buildings | BDG2 Internal RF | 8.6491 | 16.947% |

| Held-Out Buildings BDG2 | Seasonal Naive | 9.4809 | 19.307% |

The model which had been trained on simulated data performed worse than both the model trained on real data and the Seasonal Naive model when it came to overall error.

This shows that there is a difference between simulated and real data; simulated data can be of use when constructing models and for testing, but high performance in simulations does not mean that the same results will occur in real life without further work or retraining.

---

## Datasets

### NREL ResStock

The electricity usage for homes has been simulated by NREL ResStock.

For this project:

- area covered: California

- number of homes identified: 57,444

- sample size used: 250 homes

- homes used for building the model: 200

- homes completely left out: 50

- data resolution: 15 minutes

- model resolution:

- total hourly observations before filtering: 2,190,000

ResStock is the primary dataset used for predicting home energy consumption.

### NREL ComStock

NREL ComStock has carried out simulations of electricity use in offices.

For this project:

- area covered: California

- number of offices identified: 40,048

- sample size used: 250 offices

- offices used for building the model: 200

- offices completely left out: 50

- data resolution: 15 minutes

- model resolution:

- total hourly observations before filtering: 2,190,000

- 14 different types of commercial buildings

ComStock is the primary simulated dataset used for predicting office energy consumption and serves as the starting point for the simulated-to-real data experiment.

### Building Data Genome Project 2

BDG2 includes the actual hourly electricity meter readings for the buildings.

The analysis chooses 2017 since it provided greater coverage than 2016.

For this project:

- number of buildings with electricity data: 1,578

- buildings with least 95% data in 2017: 1,293

- buildings used after filtering: 1,279

- sample size used: 250 buildings

- buildings used for building the model: 200

- buildings completely left out: 50

- average data coverage is about 99.26%

The missing data was left unsupplied and only those features with all the required data were included.

BDG2 is the dataset used in the real world for predicting office electricity consumption.

### UCI Appliances Energy Prediction

The dataset from UCI Appliances includes 10-minute data from a home.

It was used to develop and check:

- hourly data

- forecasting features

- time-based splitting

- basic models

- tree-based models

- time-aware tuning

- model checks

- K-Means clustering

The last model to be tested was Extra Trees, and its MAE was 182.259.

UCI is to be used for testing methods and not for drawing conclusions regarding California homes.

### EIA-930 California

The portion of the U.S. EIA-930 Hourly Electric Grid Monitor covering California gives information on the grid.

The data include:

- 97,992 observations

- 97,968 hours with data

- data from July 2015 through September 2026

- electricity demand

- forecasted demand

- total generation

- energy exchanges

- generation by energy type

The maximum demand recorded in California was 63,622 MWh at 6:00 PM on 6 September 2022.

The EIA-930 is not used for understanding the grid in the form of a dataset for building-level models.

---

## Forecasting Task

The main task is:

Estimate the amount of electricity that a building will use one hour from now.

The data with a historical character was created in such a way that the model wouldn't have access to it.

Important forecasting features include:

- electricity use from one hour before

- electricity use from 24 hours before

- use from the last 24 hours

- average use from the last 168 hours

- time of day patterns

- day of week patterns

- month patterns

- weekend indicator

- weather data

- heating degree measure

- cooling degree measure

Where appropriate, the NREL models used details that were specific to the building.

---

## Evaluation Plan

For the experiments, random splitting was not used.

The development data was split based on time using forecast timestamps:

- 70%: training

- next 15%: validation

- last 15%: time-based test

Another test used buildings that weren't part of the model development:

- 200 buildings for training

- 50 buildings completely left out

It allows two different questions to be answered.

### Time-based Generalization

Can the model predict the uses for buildings that it has previously seen?

### Building-based Generalization

Can the model be applied to buildings which were never part of the training process?

The buildings that are excluded can make use of their previous data as features; this is a way of testing the model's performance with regard to buildings rather than assessing it based on no data at all.

---

## Models Used

The models tested included:

- Seasonal Naive

- Linear Regression

- Lasso Regression

- Random Forest Regressor

- Extra Trees Regressor

- Gradient Boosting or Histogram Gradient Boosting where possible

The settings were determined using time-aware cross-validation.

In the case of datasets containing a large number of buildings, an compacting method based on a sliding window was employed in order to preserve the temporal order and keep the procedure feasible.

The model that was finally selected was on the basis of the validation MAE.

---

## Evaluation Metrics

The models were checked using:

- Mean Absolute Error (MAE)

- Root Mean Squared Error (RMSE) when reported

- Coefficient of Variation of RMSE (CVRMSE)

- Mean Absolute Percentage Error (MAPE)

The criterion used was Mae for selecting the best model.

In the case of BDG2, MAPE was used with care since real building data can have small but non-zero values which can result in very high percentage errors even when the absolute errors are small.

---

## Unsupervised Analysis

The K-Means clustering method was applied in order to gain some insight into the forecasting.

### UCI

The analysis of the test identified three patterns in the use of energy and weather.

### NREL ResStock

For each of the 250 home samples the average energy usage over a 24-hour period was calculated.

The shape of the use of each building was adjusted so as to emphasize it rather than the amount of use.

The last group included four patterns relating to residential use, with operating times in the morning, in the afternoon, in the evening and more generally.

---

## Repository Layout

```text

.

├── data/

│ ├── processed/

│ └─ raw/

├── notebooks/

├── outputs/

│ ├── figures/

│ ├── models/

│ ├── predictions/

│ └── tables/

├── src/

│ ├── data/

│ ├── evaluation/

│ ├── features/

│ └── models/

├── requirements.txt

└── README.md

## Reproducing the Analysis

I've tested this project out on Windows using Python and PowerShell.

Once you've cloned the repo and installed the dependencies as listed in the file requirements.txt you'll want to run the analysis from the root of the repo.

### Required Source Data

Several source datasets are too large to keep in the GitHub repository and you'll need to obtain them separately.

The project uses the following datasets:

- UCI Appliances Energy Prediction

- NREL ResStock

- NREL ComStock

- Building Data Genome Project 2 (BDG2)

- EIA-930 California grid data

The local project structure expects the major source files and repositories in locations such as:

```text

appliances+energy+prediction/

building-data-genome-project-2/

data/

Region_CAL.xlsx

```

### `src/data/`

Carrying out data inspection, obtaining data, preparing the data, taking samples, carrying out hourly aggregation, integrating weather data and carrying out chronological splitting.

### `src/features/`

Forecast feature engineering.

### `src/models/`

Baseline evaluation. Modeling. Hyperparameter tuning. Final model evaluation. ComStock to BDG2 transfer modeling.

### `src/evaluation/`

Utilities for shared evaluation, model diagnostics, cross-dataset synthesis, and reproducibility checks.

### `outputs/`

Tables, graphs, predictions and the fitted model items were generated.

---

## Code and Reproducibility

All source code and generated project outputs are available in this GitHub repository:

https://github.com/jdelvil4/602

### Repository Contents

- `src/`: project source code

- `data/`: data dictionaries, metadata and processed data locations used by the project

- `outputs/tables/`: generated evaluation tables and summaries

- `outputs/figures/`: generated project figures

- `outputs/models/`: fitted model artifacts that are retained

- `outputs/predictions/`: saved prediction outputs that are retained

- `requirements.txt`: the Python packages Dependencies needed by the project

## How to Reproduce the Results

The project was developed and tested on Windows with the use of Python and PowerShell.

### 1. Copy the repository

```powershell

git clone https://github.com/jdelvil4/602.git

cd 602

```

### 2. Set up the Python environment

```powershell

python -m venv.venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt

```

The project makes use of various packages including Pandas, NumPy, Scikit-Learn, Matplotlib, Joblib, PyArrow, OpenPyXL and Boto3.

### 3. Get the source data

Some of the source datasets are so large that they cannot be stored in the GitHub repository and must therefore be obtained separately.

Required datasets are:

- UCI Appliances Energy Prediction

- NREL ResStock

- NREL ComStock

- Building Data Genome Project 2

- EIA-930 California hourly data

The local project structure expects major source data in locations such as:

```text

appliances+energy+prediction/

building-data-genome-project-2/

data/

Region_CAL.xlsx

```

For BDG2 expected source files include:

```text

building-data-genome-project-2/data/metadata/metadata.csv

building-data-genome-project-2/data/meters/cleaned/electricity_cleaned.csv

building-data-genome-project-2/data/weather/weather.csv

```

NREL metadata files are expected under:

```text

data/resstock_metadata.parquet

data/comstock_metadata.parquet

```

The UCI source file is:

```text

appliances+energy+prediction/energydata_complete.csv

```

The individual-building profiles and weather files from NREL are downloaded and then processed using the projects' data-acquisition and preparation scripts.

### 4. Carry out the workflow

1. Have a look at and get the source datasets ready.

2. Change the electricity measurements to a hourly resolution.

3. Combine weather data and building information.

4. Generate features for forecasting that are ahead by one hour.

Create the training, validation and test sets.

6. Set aside some buildings that are not part of the main group for use in evaluating buildings across the group.

7. Assess the Seasonal Naive Linear Regression and Lasso baseline models.

8. Train and tune the tree-based models.

9. Assess the locked models using temporal and held-out-building tests.

Carry out the ComStock-to-BDG2 simulated-to-real transfer experiment.

11. Carry out an analysis of the California grid context as specified in EIA-930.

Produce synthesis outputs across datasets.

Carry out the reproducibility audit.

Source code is organized by stage:

```text

src/data/        data preparation, inspection, sampling and splitting

src/features/    forecasting feature engineering

src/models/      baseline models, tree models, tuning and final evaluation

src/evaluation/  metrics, diagnostics, synthesis and reproducibility checks

```

### 5.. Transfer Workflow

The later‑stage BDG2 and simulated‑to‑real workflow can be run from the repository root with:

```powershell

python -m src.data.prepare_bdg2

python -m src.data.select_bdg2_modeling_sample

python -m src.data.prepare_bdg2_modeling_panel

python -m src.features.bdg2_forecast_features

python -m src.data.split_bdg2

python -m src.models.evaluate_bdg2_baselines

python -m src.models.evaluate_bdg2_trees

```

```bash

python -m src.models.tune_bdg2_random_forest

python -m src.models.evaluate_bdg2_final

python -m src.models.train_comstock_transfer_model_light

python -m src.models.evaluate_comstock_to_bdg2_transfer

python -m src.data.analyze_eia930_context

python -m src.evaluation.build_final_synthesis

```

The UCI, ResStock and ComStock stages involve the same steps: preparation, feature engineering, splitting, modeling, tuning and final evaluation, the scripts for each of these stages being found in the `src/` directory.

### 6. Reproducibility Settings

```python

RANDOM_STATE = 42

TRAIN_FRACTION = 0.70

VALIDATION_FRACTION = 0.15

TEST_FRACTION = 0.15

FORECAST_HORIZON_HOURS = 1

```

The datasets for forecasting are divided in the order of time at random.

For the multi‑building analyses:

The model is developed using 200 buildings.

- Fifty buildings are designated as held-out buildings.

- The buildings that are held out are not employed in the process of model fitting or hyperparameter tuning.

- When it applies, the final forecast period is used in both temporal and held-out evaluations.

### 7. Generated Outputs

Processed datasets are written to:

```text

data/processed/

```

Model results and artifacts are written under:

```text

outputs/tables/

outputs/models/

outputs/predictions/

```

Figures are written to:

```text

outputs/figures/

```

Important final outputs include:

```text

outputs/tables/final_model_performance_summary.csv

outputs/tables/final_mae_improvement_vs_seasonal.csv

outputs/tables/final_transfer_comparison.csv

outputs/tables/final_project_key_findings.csv

outputs/figures/final_mae_improvement_vs_seasonal.png

outputs/figures/final_simulated_vs_real_transfer.png

```

### 8. Final Verification

After the analysis is run:

```powershell

python -m src.evaluation.final_project_audit

```

The most recent completed project audit reported:

```text

PASS: 131

FAIL: 0

```

The last Git working-tree warning that was discovered as a result of the audit was dealt with by making the project changes and synchronizing the repository with origin/main.

The entire workflow involves millions of building-hour observations and uses tree-based ensemble models, so it may require a large amount of disk space, memory and time.

## Final Cross‑Dataset Synthesis

The final project‑level comparison can be regenerated with:

```powershell

python -m src.evaluation.build_final_synthesis

```

The synthesis evaluates the performance of the models by comparing them with the Seasonal Naive method and looks at the raw MAE for each dataset.

This is important since the datasets include electricity-consumption scales.

The primary cross‑dataset figure is:

```text

outputs/figures/final_mae_improvement_vs_seasonal.png

```

The simulated‑to‑real transfer comparison is presented separately in:

```text

outputs/figures/final_simulated_vs_real_transfer.png

```

## Interpretation

The project supports four conclusions:

The information needed to forecast electricity consumption one hour in advance includes historical consumption data, variables related to time, and weather conditions.

2. Ensemble models based on trees usually produce forecasts that are better than simple baseline models, but the extent of the improvement varies considerably depending on the dataset and the way in which it is evaluated.

3. It is possible to make much better forecasts when using data from buildings that are familiar than when forecasting for buildings that have never been seen before.

4. The fact that the model performs well on building data does not imply that it will work well on actual meters, since a direct transfer from ComStock to BDG2 failed without domain adaptation or training in the real world.

## Limitations

Several limitations should be considered when interpreting the results:

- A small number of the various NREL and BDG2 buildings were used in the models.

The number of hyperparameter searches was kept small due to the size of the dataset, the available computation time, and the limited system memory.

The evaluation of the building that is being held back assumes that each building has past electricity readings for the lagged predictors and is therefore not a cold-start test with no history.

-The direct transfer from ComStock to BDG2 made use of a shared raw-scale feature contract. Neither a clear local BDG2 unit declaration was found, so any doubts regarding the unit documentation should be borne in mind when examining the transfer experiment.

- MAPE may become unstable if the actual electricity consumption is near zero.

The EIA-930 operates at the grid level and cannot therefore be regarded as directly comparable to building-level forecasting models.

- Certain raw datasets are so large that they cannot be shared in this repository and must be obtained separately.

## Future Work

Potential extensions include:

Domain adaptation happens between real building data.

- Training that involves the use of both simulated and actual meter observations.

- It seems to me that more geographic regions should be included.

- I observe that the number of buildings is increasing.

In my opinion, it is essential to have cold-start forecasting for buildings when there is no historical meter data.

I know that probabilistic forecasting and uncertainty estimation are important.

I think live meter integration is essential.

- I believe that aggregating building forecasts at a community or grid level is useful.

## Repository

GitHub:

https://github.com/jdelvil4/602
