import yfinance as yf
import numpy as np
from datetime import datetime, timedelta


def VaR_historical_method(days =252, confidence_level = 0.95, data=[]):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)#1 year

    S = yf.download(data, start = start_date , end = end_date)['Adj Close'] #Adjusted price with dividends and splits
    daily_returns = ( (S - S.shift(1)) / S.shift(1) ).dropna() #Calculates simple daily returns and removes missing values

    VaR = np.percentile(daily_returns, (1 - confidence_level) * 100)
    return f"VaR 95% 1 day: {VaR:.4%}"


def VaR_monte_carlo_method(days=252, number_simulations=10000, confidence_level = 0.95, data=[]):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)  #1 year

    """Historical data and returns"""
    S = yf.download(data, start=start_date, end=end_date)['Adj Close']  #Adjusted price with dividends and splits
    daily_returns = ((S - S.shift(1)) / S.shift(1)).dropna()  #Calculates simple daily returns and removes missing values

    """Simulated returns for 1 day"""
    simulated_returns = np.random.normal(np.mean(daily_returns), np.std(daily_returns),(number_simulations, 1)) #Return simulation for 1 day
    simulated_prices = S[-1] * np.exp(simulated_returns)  #Lognormal distribution from last price

    """Calculate daily returns from simulated prices"""
    simulated_daily_returns = (simulated_prices - S[-1]) / S[-1]  #Calculate 1-day return for each simulation

    """VaR Calculation"""
    all_simulated_returns = simulated_daily_returns.flatten()  #Flatten to 1D array for all returns
    VaR = np.percentile(all_simulated_returns, (1 - confidence_level) * 100)
    return f"VaR 95% 1 day: {VaR:.4%}"


print("VaR historical method: ",VaR_historical_method(data=['^FCHI']))
print("VaR monte carlo method: ",VaR_monte_carlo_method(data=['^FCHI']))