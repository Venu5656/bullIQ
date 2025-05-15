# BullIQ: A Machine Learning Framework for Weekly Stock Return Prediction and Portfolio Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) This repository contains the code, notebooks, and resources for the **BullIQ** project. BullIQ is a comprehensive pipeline designed to predict weekly stock returns by integrating financial market data with sentiment analysis from news articles, and subsequently constructs optimized investment portfolios.

## Associated Publication

This work is detailed in our paper:

**"BullIQ: A Machine-Learning Framework for Weekly Stock Return Prediction and Portfolio Optimization"**

The paper provides an in-depth discussion of the methodology, experimental setup, results (including key figures like feature importance, prediction accuracy, and portfolio performance), and limitations. Performance metrics reported in the paper include a test MSE of approximately 0.00232, an annualized Sharpe ratio around 17.1, and an expected annual return of about 23.4% for the optimized portfolio (achieved by blending predicted returns with historical means and applying Ledoit-Wolf covariance shrinkage).

## Project Overview

The BullIQ pipeline encompasses several stages:

1.  **Data Collection & Preparation:**
    * Historical weekly stock prices (OHLCV) are loaded.
    * Daily news articles (snippets, publication dates) are processed.
2.  **Feature Engineering:**
    * **Market-based features:** Log returns, rolling and EWMA volatility, technical indicators (SMA, EMA, RSI, MACD, Volume Change) are calculated.
    * **News Embeddings:** News snippets are converted into dense vector representations using `sentence-transformers` (`all-MiniLM-L6-v2`). These daily embeddings are aggregated weekly with a time-decay weighting.
    * **PCA on Embeddings:** Principal Component Analysis (PCA) is applied to reduce the dimensionality of the weekly news embeddings.
    * **Sentiment Scores:** News snippets are analyzed using FinBERT (`ProsusAI/finbert`) to derive positive, neutral, and negative sentiment probabilities. A net sentiment score is calculated and aggregated weekly with time-decay.
    * **Feature Smoothing:** Key features like sentiment EWMA, volatility EWMA, RSI, and volume change are smoothed using EWMA or rolling means.
3.  **Predictive Modeling:**
    * A LightGBM regressor is trained to predict the next week's log return.
    * The feature set includes smoothed market indicators, sentiment scores, and top PCA components from news embeddings.
    * A time-aware train/test split is used (e.g., training up to the end of 2022, testing on 2023-2024 data).
    * Predictions are clipped (e.g., to $\pm1\%$) to manage risk from extreme forecasts.
4.  **Portfolio Optimization:**
    * Predicted returns (potentially blended with historical means) serve as expected returns ($\mu$).
    * A Ledoit-Wolf shrunk covariance matrix ($\Sigma$) is estimated from historical returns.
    * `PyPortfolioOpt` is used to construct long-only portfolios, maximizing the Sharpe ratio, subject to weight constraints (e.g., max 15% per asset).
5.  **Performance Evaluation & Backtesting:**
    * The model's predictive accuracy (e.g., MSE) is evaluated.
    * Portfolio performance (annualized return, volatility, Sharpe ratio, cumulative returns) is simulated under different rebalancing frequencies (weekly vs. monthly).
    * Stress testing scenarios are considered to evaluate portfolio robustness.

## Dataset

The project utilizes two main data sources, with initial collection potentially handled by `dataset.ipynb` and primary processing/modeling in `bullIQ_.ipynb`:

1.  **Financial Market Data:**
    * **Input File (for `bullIQ_.ipynb`):** `weekly_stock_price_data.csv`
    * **Content:** Historical weekly Open, High, Low, Close (OHLC) prices, and Volume.
    * **Original Source:** Yahoo Finance (via `yfinance` library in `dataset.ipynb`).
    * **Symbols:** Sourced from `financial_symbols.csv` (e.g., S&P 500 constituents, ETFs).
    * **Period (in `dataset.ipynb`):** January 1, 2015 - January 1, 2025.

2.  **News Data:**
    * **Input File (for `bullIQ_.ipynb`):** `news_articles_with_dates_2014_2025.csv`
    * **Content:** News articles (titles, URLs, snippets, publication dates).
    * **Original Source:** Google Custom Search API (queried in `dataset.ipynb` for articles published roughly between 2020-2025, though the filename suggests a broader collection from 2014).
    * **Date Extraction (in `bullIQ_.ipynb`):** Primarily via regex from snippets, with `newspaper3k` used in `dataset.ipynb` for initial attempts.

## Repository Structure
