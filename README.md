# ALLMIND

ALLMIND is a machine learning trading research framework that builds technical indicator models and a meta-model to predict price movements in cryptocurrency markets.

The system performs the following pipeline:
Exchange Data → Candles → Indicator Features → Indicator Models → Meta Features → Meta Model → Predictions

The goal of the project is to research whether a combination of technical indicators can be used to reliably predict short-term and long-term price movements.

---

# Architecture

The project is separated into clear functional layers.

Data Layer
- Fetches OHLCV market data from the exchange using CCXT
- Stores candles locally as CSV files

Indicator Layer
- Builds technical indicator features such as:
  - EMA
  - RSI
  - SMA
  - Support / Resistance
  - VWAP

ML Layer
- Trains a model per indicator
- Each model predicts whether price will increase within a defined horizon

Meta Layer
- Combines predictions from ALL indicator models
- Trains a meta-model to determine overall probability of price movement

Prediction Layer
- Evaluates current market conditions using the meta-model
- Determines if the current price is a potential buy or sell opportunity

---

# Installation

Clone the repository:
> git clone https://github.com/yourusername/allmind.git

> cd allmind

Create a virtual environment:
> python -m venv venv

Install dependencies:
> pip install -r requirements.txt

---

# Running the Project

Main pipeline is ran every midnight using GitHub Actions.

To run the main pipeline manually:
> python main.py

The pipeline will:
1. Load/update candle data
2. Build indicator features
3. Train indicator models
4. Generate meta features
5. Train the meta model
6. Evaluate current market conditions

Analysis and predictions are appended to a file:
"ANALYSIS.txt"

---

# Data Storage

Generated data is stored inside the `property` directory.

property.candles (historical OHLCV data)  
property.meta_models (trained meta models)  
property.models (trained indicator models)  

These files are generated automatically during training.

---

# Design Principles

The project follows several strict design principles. Using a deterministic data pipeline, each stage produces validated outputs that become inputs for the next stage (each having it's own responsibilities). Errors are raised early when corrupted data is detected.

Data Layer:  
Responsible for retrieving and storing market data.

Indicator Layer:  
Responsible for transforming price data into feature vectors.

Model Layer:  
Responsible for training and evaluating ML models.

Meta Layer:  
Responsible for aggregating model predictions.

Prediction Layer:  
Responsible for final trading signals.
