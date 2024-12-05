from yahooquery import Ticker

# Define the ticker for Nvidia
ticker = 'NVDA'

# Fetch the Nvidia data using yahooquery
nvidia = Ticker(ticker)

# Use .get() with a default value of None to avoid KeyErrors if fields are missing
price_data = nvidia.price.get(ticker, {})
summary_data = nvidia.summary_detail.get(ticker, {})
financial_data = nvidia.financial_data.get(ticker, {})
valuation_measures = nvidia.valuation_measures.get(ticker, {})

# Retrieve data points with error handling
current_price = price_data.get('regularMarketPrice')
target_price = valuation_measures.get('targetMeanPrice', None)
market_cap = price_data.get('marketCap')
pe_ratio = summary_data.get('trailingPE')
forward_pe = summary_data.get('forwardPE')
eps_ttm = summary_data.get('trailingEps')

# Get financial statements for historical revenue and EBITDA
quarterly_revenue = financial_data.get('totalRevenue')
quarterly_ebitda = financial_data.get('ebitda')

# Analyst estimates for revenue and EPS growth
analysis_data = valuation_measures  # This may contain forward estimates if available

# Display collected data
results = {
    "Current Price": current_price,
    "Target Price": target_price,
    "Market Cap": market_cap,
    "PE Ratio (TTM)": pe_ratio,
    "Forward PE": forward_pe,
    "EPS (TTM)": eps_ttm,
    "Quarterly Revenue": quarterly_revenue,
    "Quarterly EBITDA": quarterly_ebitda,
    "Analyst Estimates": analysis_data,
}

# Print the results
print(results)



# Provided Data
current_price = 136.05  # Current stock price
market_cap = 3337306505216  # Market Cap
pe_ratio = 63.279068  # Trailing P/E Ratio
forward_pe = 33.509853  # Forward P/E Ratio
quarterly_revenue = 96307003392  # Quarterly revenue (TTM)
quarterly_ebitda = 61184000000  # Quarterly EBITDA (TTM)

# Assumptions (replace None with values if found manually)
target_price = 148.87  # Replace if manually retrieved
eps_ttm = None  # Replace if manually retrieved

# Calculate EBITDA Margin
if quarterly_revenue and quarterly_ebitda:
    ebitda_margin = (quarterly_ebitda / quarterly_revenue) * 100
else:
    ebitda_margin = None

# Calculate Upside Percentage if target price is available
if target_price and current_price:
    upside_percentage = ((target_price - current_price) / current_price) * 100
else:
    upside_percentage = None

# Display results
calc_results = {
    "EBITDA Margin (%)": ebitda_margin,
    "Upside Percentage to Target Price (%)": upside_percentage,
}

print(calc_results)
