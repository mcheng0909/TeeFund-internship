import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from datetime import datetime

# Expanded stock symbols list (100 companies across different sectors)
stock_symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'IBM', 'APLE', 'TSLA', 'NVDA', 'ORCL', 'JNJ', 'PFE', 'MRK', 'JPM', 
    'BAC', 'WFC', 'KO', 'PG', 'NKE', 'XOM', 'CVX', 'AMZN', 'WMT', 'TGT', 'META', 'NFLX', 'ADBE', 
    'INTC', 'CSCO', 'QCOM', 'AMD', 'TXN', 'AVGO', 'MU', 'CRM', 'SHOP', 'PYPL', 'SBUX', 'PEP', 'MCD', 
    'V', 'MA', 'AXP', 'GS', 'MS', 'BLK', 'SCHW', 'TROW', 'LMT', 'BA', 'RTX', 'NOC', 'GD', 'FDX', 
    'UPS', 'CAT', 'DE', 'MMM', 'GE', 'HON', 'GM', 'F', 'NIO', 'LI', 'XPEV', 'RIVN', 'SPOT', 'DIS', 
    'CMCSA', 'VZ', 'T', 'TMUS', 'CL', 'KMB', 'MDLZ', 'GIS', 'COST', 'LOW', 'HD', 'BBY', 'BKNG', 
    'EXPE', 'DAL', 'UAL', 'LUV', 'AAL', 'AIG', 'PRU', 'MET', 'TRV', 'C', 'WFC', 'USB', 'PNC', 'TFC', 
    'HBAN', 'KEY', 'STT', 'BK', 'SYF', 'COF', 'DFS'
]

# Download data for the selected stock symbols
data = yf.download(stock_symbols, start="2023-01-01", end="2024-01-01", threads=True)

# Ensure that data is not missing any columns
data = data.dropna(axis=1, how='all')

if 'Adj Close' not in data.columns or 'Volume' not in data.columns:
    raise ValueError("Missing required data columns: 'Adj Close' or 'Volume'")

stock_prices = data['Adj Close']
stock_returns = stock_prices.pct_change().dropna()
composite_scores = []

# Calculate composite score for each stock
def calculate_composite_score(symbol_data, symbol_volume):
    df = symbol_data.copy()

    # Calculate Price and Volume Trend (PVT) using vectorized operations
    pvt = ((df.diff() / df.shift(1)) * symbol_volume).cumsum().fillna(0)
    pvt_change = pvt.iloc[-1] - pvt.iloc[0]

    # Calculate Rate of Change (ROC) using Pandas methods
    roc_period = 14
    roc = df.pct_change(periods=roc_period) * 100
    roc_avg = roc.mean()

    # Calculate On-Balance Volume (OBV) using vectorized operations
    obv = np.where(df > df.shift(1), symbol_volume, np.where(df < df.shift(1), -symbol_volume, 0)).cumsum()
    obv_change = obv[-1] - obv[0]

    # Calculate Composite Score
    composite_score = pvt_change + roc_avg + obv_change
    return composite_score

# Calculate composite scores for all stock symbols
for symbol in stock_symbols:
    if symbol not in stock_prices.columns or symbol not in data['Volume'].columns:
        continue
    symbol_data = stock_prices[symbol]
    symbol_volume = data['Volume'][symbol]
    score = calculate_composite_score(symbol_data, symbol_volume)
    composite_scores.append({'Symbol': symbol, 'Score': score})

# Combine results into a DataFrame and rank by score
results_df = pd.DataFrame(composite_scores)
results_df = results_df.sort_values(by='Score', ascending=False).reset_index(drop=True)

# Sector Performance Comparison
sector_mapping = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'IBM': 'Tech', 'APLE': 'Real Estate', 'TSLA': 'Tech',
    'NVDA': 'Tech', 'ORCL': 'Tech', 'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare',
    'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'KO': 'Consumer Goods',
    'PG': 'Consumer Goods', 'NKE': 'Consumer Goods', 'XOM': 'Energy', 'CVX': 'Energy',
    'AMZN': 'Retail', 'WMT': 'Retail', 'TGT': 'Retail', 'META': 'Tech', 'NFLX': 'Tech', 'ADBE': 'Tech', 
    'INTC': 'Tech', 'CSCO': 'Tech', 'QCOM': 'Tech', 'AMD': 'Tech', 'TXN': 'Tech', 'AVGO': 'Tech', 
    'MU': 'Tech', 'CRM': 'Tech', 'SHOP': 'Tech', 'PYPL': 'Tech', 'SBUX': 'Consumer Goods', 
    'PEP': 'Consumer Goods', 'MCD': 'Consumer Goods', 'V': 'Financial', 'MA': 'Financial', 'AXP': 'Financial', 
    'GS': 'Financial', 'MS': 'Financial', 'BLK': 'Financial', 'SCHW': 'Financial', 'TROW': 'Financial', 
    'LMT': 'Defense', 'BA': 'Aerospace', 'RTX': 'Defense', 'NOC': 'Defense', 'GD': 'Defense', 'FDX': 'Transportation', 
    'UPS': 'Transportation', 'CAT': 'Industrial', 'DE': 'Industrial', 'MMM': 'Industrial', 'GE': 'Industrial', 
    'HON': 'Industrial', 'GM': 'Auto', 'F': 'Auto', 'NIO': 'Auto', 'LI': 'Auto', 'XPEV': 'Auto', 
    'RIVN': 'Auto', 'SPOT': 'Tech', 'DIS': 'Entertainment', 'CMCSA': 'Entertainment', 
    'VZ': 'Telecom', 'T': 'Telecom', 'TMUS': 'Telecom', 'CL': 'Consumer Goods', 'KMB': 'Consumer Goods', 
    'MDLZ': 'Consumer Goods', 'GIS': 'Consumer Goods', 'COST': 'Retail', 'LOW': 'Retail', 
    'HD': 'Retail', 'BBY': 'Retail', 'BKNG': 'Travel', 'EXPE': 'Travel', 'DAL': 'Airline', 
    'UAL': 'Airline', 'LUV': 'Airline', 'AAL': 'Airline', 'AIG': 'Insurance', 'PRU': 'Insurance', 
    'MET': 'Insurance', 'TRV': 'Insurance', 'C': 'Financial', 'WFC': 'Financial', 'USB': 'Financial', 
    'PNC': 'Financial', 'TFC': 'Financial', 'HBAN': 'Financial', 'KEY': 'Financial', 
    'STT': 'Financial', 'BK': 'Financial', 'SYF': 'Financial', 'COF': 'Financial', 'DFS': 'Financial'
}

results_df['Sector'] = results_df['Symbol'].map(sector_mapping)

# Calculate Sector Average Score
sector_avg_scores = results_df.groupby('Sector')['Score'].mean().reset_index()
print("\nSector Average Scores:\n", sector_avg_scores)

# Risk-Adjusted Performance (Sharpe Ratio Calculation)
def sharpe_ratio(returns, risk_free_rate=0.01):
    mean_return = returns.mean()
    std_dev = returns.std()
    if std_dev == 0:
        return np.nan
    return (mean_return - risk_free_rate) / std_dev

# Calculate Sharpe Ratios for all stocks simultaneously
sharpe_ratios = stock_returns.apply(lambda x: sharpe_ratio(x))
sharpe_ratios_df = sharpe_ratios.reset_index().rename(columns={0: 'Sharpe Ratio'}).sort_values(by='Sharpe Ratio', ascending=False)
print("\nSharpe Ratios for Stocks:\n", sharpe_ratios_df)

# Factor analysis (calculate IC for additional stocks)
def calculate_ic(stock_symbols, stock_returns):
    factor_scores = {}

    # Iterate through each stock symbol to calculate factors
    for symbol in stock_symbols:
        if symbol not in stock_returns.columns:
            continue
        stock_data = stock_returns[symbol]

        # Calculate ROC for each stock
        roc = stock_data.rolling(window=14).mean()
        # Calculate OBV for each stock
        obv = stock_data.cumsum()

        # Store ROC and OBV scores
        factor_scores[symbol] = {
            'ROC': roc,
            'OBV': obv
        }

    # Calculate IC (Information Coefficient) for each factor
    ics = {}
    for symbol, factors in factor_scores.items():
        ic_values = {}
        for factor, values in factors.items():
            next_period_returns = stock_returns[symbol].shift(-1)
            
            # Align the factor values and returns
            valid_index = values.dropna().index.intersection(next_period_returns.dropna().index)
            aligned_returns = next_period_returns.loc[valid_index]
            aligned_values = values.loc[valid_index]

            # Calculate Spearman rank correlation
            if len(aligned_values) > 0 and len(aligned_returns) > 0:
                ic, _ = spearmanr(aligned_values, aligned_returns)
                ic_values[factor] = ic

        ics[symbol] = ic_values

    return ics

# Compute IC values for factors
ic_values = calculate_ic(stock_symbols, stock_returns)

# Display the composite score, sector analysis, Sharpe ratios, and factor IC results
print("Composite Scores for Stocks:\n", results_df)
print("\nInformation Coefficient (IC) for Factors by Stock:\n", ic_values)
print("\nSharpe Ratios for Stocks:\n", sharpe_ratios_df)

# Price Target Projections based on Historical Volatility, Growth Momentum, and Macroeconomic Indicators
def price_target_projection(symbol_data, current_price, volatility_multiplier=1.2, momentum_weight=0.5, macroeconomic_factor=0.03):
    historical_volatility = symbol_data.pct_change().std()
    momentum = symbol_data.pct_change().mean()
    projected_change = (historical_volatility * volatility_multiplier) + (momentum_weight * momentum) + macroeconomic_factor
    return current_price * (1 + projected_change)

price_targets = {}
for symbol in stock_symbols:
    if symbol not in stock_prices.columns:
        continue
    current_price = stock_prices[symbol].iloc[-1]
    symbol_data = stock_prices[symbol]
    price_targets[symbol] = price_target_projection(symbol_data, current_price)

price_targets_df = pd.DataFrame(list(price_targets.items()), columns=['Symbol', 'Price Target']).sort_values(by='Price Target', ascending=False)
print("\nPrice Targets for Stocks:\n", price_targets_df)
