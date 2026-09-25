# Data

Large source datasets not stored directly in this repository

## UCI Appliances Energy Prediction

Source: UCI Machine Learning Repository

Original 10 minute dataset processed by:

`src/data/prepare_uci.py`

Hourly dataset written to:

`data/processed/uci_hourly.csv`

## NREL ResStock

NREL ResStock load-profile data downloaded separately.

Data acquisition utilities located in:

`scripts/download_nrel_data.py`

Raw and processed NREL files excluded from Git because of their size.

## Building Data Genome Project 2

BDG2 data obtained from the project's public repository/source, not duplicated in this repository.

## EIA-930

EIA electric grid operating data downloaded separately from the U.S. Energy Information Administration.