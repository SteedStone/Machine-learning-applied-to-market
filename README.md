# Machine Learning Stock Market Prediction Project

⚠️ **WORK IN PROGRESS** - This project is currently unfinished and under development.

## Overview

This project explores stock market prediction using machine learning techniques, specifically focusing on LSTM (Long Short-Term Memory) neural networks and Random Forest classifiers. The project has been inspired by various YouTube tutorials and educational content on financial time series forecasting.

## Features

### Current Implementations

- **LSTM-based Stock Prediction**: Time series forecasting using deep learning
  - Multi-step ahead predictions
  - Stock price trend prediction
  - Feature scaling and preprocessing
  - Recursive prediction capabilities

- **Random Forest Market Classification**: Binary classification for market direction
  - S&P 500 index prediction
  - Technical indicators integration
  - Backtesting framework
  - Precision-based model evaluation

### Datasets

The project works with multiple financial datasets:
- Individual stocks (AAPL, MSFT, MS)
- Market indices (S&P 500, CAC 40, DAX)
- Historical stock data via Yahoo Finance API

## Project Structure

```
├── Version_pas_mal.ipynb          # Main LSTM implementation
├── market_prediction.ipynb        # Random Forest market prediction
├── helping_functions.py           # Utility functions for plotting and analysis
├── *.csv files                    # Historical stock data
├── *.png files                    # Generated plots and visualizations
└── README.md                      # This file
```

## Key Components

### LSTM Model (`Version_pas_mal.ipynb`)
- Data preprocessing with StandardScaler
- Window-based time series preparation
- Multi-layer LSTM architecture with dropout
- Future price prediction with recursive forecasting
- Visualization of training, validation, and test results

### Random Forest Model (`market_prediction.ipynb`)
- Binary classification for market direction
- Technical indicators (moving averages, trend analysis)
- Backtesting with walk-forward validation
- Precision score optimization

## Dependencies

```python
pandas
numpy
tensorflow/keras
scikit-learn
yfinance
matplotlib
plotly
seaborn
```

## Current Status

🚧 **This project is incomplete and contains several issues:**

- Some code cells have errors and need debugging
- Model performance needs optimization
- Code structure requires refactoring
- Documentation is partial
- Testing framework is missing

## Inspiration

This project has been developed following concepts and techniques learned from YouTube educational videos on:
- Machine learning for finance
- LSTM neural networks
- Stock market prediction
- Technical analysis with Python

## Future Work

- [ ] Fix existing bugs and errors
- [ ] Improve model accuracy
- [ ] Add more sophisticated features
- [ ] Implement proper model validation
- [ ] Add comprehensive documentation
- [ ] Create deployment pipeline

## Disclaimer

This projected has not been finished yet and has been develloped during my holiday of first year, at that time I didn't had enough knowledge to finish it. Work to be finished

---

