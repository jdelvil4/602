# Optimizing Energy Usage with Machine Learning

The project involves the use of machine learning techniques in order to predict building electricity consumption one hour in advance, based on both simulated and real energy data.

The study uses data from both homes and offices, including information on weather conditions, time-related features, previous electricity consumption, unsupervised clustering, and details about the California power grid.

A main aim is not just to check how accurate the predictions are, but to see how well the models can apply to different situations:

1. Is it possible for a model to predict the electricity usage of buildings which were included in the training set?

2. Would this approach be applicable to buildings that have never been seen before?

3. Can a model which has been trained on data relating to offices actually handle real electricity data from offices?

The project involves the use of Python and Scikit-Learn, with a time-based splitting method, cross-validation that is time-aware and which tests on buildings not seen during training, and full documentation of the modelling decisions.

---

## Key Results

The findings indicate that tree-based models generally assist with predictions made one hour in advance; the degree of their performance is very much dependent on the kind of generalisation under consideration.

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