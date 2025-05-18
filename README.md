# Hybrid-Model-for-PV-Power-Forecasting

## Abstract
Accurate prediction of photovoltaic (PV) power relies on meteorological forecasting techniques, which is critical for the secure operation of power systems. To enhance ultra-short-term PV power forecasting performance, this study proposes a hybrid deep learning framework that integrates meteorological model-prediction with station-measured data.

## Directory Overview
- **data/**: Contains scripts for data standardization of raw data
  - `DLdate.py`: Iterate by date and split training and testing sets
  - `EC_preprocess.py`: EC data preprocessing module 
  - `fusion_pre.py`: Multi-source data fusion preprocessing module, aligning EC data with station-measured data

- **data/Output/**: Contains the actual power generation data of PV stations
 
- **Model/**: Contains proposed deep learning models
  - `CNN_BiLSTM.py`: CNN-BiLSTM short-term PV power forecasting model
  - `CNN_Transformer.py`: CNN-Transformer ultra-short-term PV power forecasting model

- **Module/**: Contains utility modules
  - `Time_alignment.py`: Time series alignment tool
  - `data_clean.py`: Data cleaning and standardization module
  - `feat_engineer.py`: Feature engineering processing module

- **model_saved/**: Saves trained model weights

- **main.ipynb**: Main execution notebook for preprocessing, training, testing, and related plotting
- **README.md**: Project documentation


