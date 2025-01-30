import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

"""The Greeks - Black-Scholes model"""

def call_BS(S,r,K,T,sigma):
    d1 = math.log(S*math.exp(r*T)/K) / (sigma*math.sqrt(T)) + 0.5*sigma*math.sqrt(T)
    d2 = d1 - sigma*math.sqrt(T)
    N1 = norm.cdf(d1)
    N2 = norm.cdf(d2)
    call_price = S*N1 - K*math.exp(-r*T)*N2
    return round(call_price,4)

def put_BS(S,r,K,T,sigma):
    d1 = math.log(S*math.exp(r*T)/K) / (sigma*math.sqrt(T)) + 0.5*sigma*math.sqrt(T)
    d2 = d1 - sigma*math.sqrt(T)
    N1 = norm.cdf(-d1)
    N2 = norm.cdf(-d2)
    put_price =  K*math.exp(-r*T)*N2 - S*N1
    return round(put_price,4)

def call_delta(S,r,K,T,sigma):
    d1 = math.log(S*math.exp(r*T)/K) / (sigma*math.sqrt(T)) + 0.5*sigma*math.sqrt(T)
    return norm.cdf(d1)

def put_delta(S,r,K,T,sigma):
    d1 = math.log(S*math.exp(r*T)/K) / (sigma*math.sqrt(T)) + 0.5*sigma*math.sqrt(T)
    return -norm.cdf(-d1)

def call_gamma(S,r,K,T,sigma):
    d1 = math.log(S*math.exp(r*T)/K) / (sigma*math.sqrt(T)) + 0.5*sigma*math.sqrt(T)
    return norm.pdf(d1) / (S*sigma*math.sqrt(T))

def put_gamma(S,r,K,T,sigma):
    return call_gamma(S,r,K,T,sigma)

def call_theta(S,r,K,T,sigma):
    d1 = math.log(S*math.exp(r*T)/K) / (sigma*math.sqrt(T)) + 0.5*sigma*math.sqrt(T)
    d2 = d1 - sigma*math.sqrt(T)
    return -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r*K*math.exp(-r*T)*norm.cdf(d2)

def put_theta(S,r,K,T,sigma):
    d1 = math.log(S*math.exp(r*T)/K) / (sigma*math.sqrt(T)) + 0.5*sigma*math.sqrt(T)
    d2 = d1 - sigma*math.sqrt(T)
    return -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r*K*math.exp(-r*T)*norm.cdf(-d2)

def call_vega(S,r,K,T,sigma):
     d1 = math.log(S*math.exp(r*T)/K) / (sigma*math.sqrt(T)) + 0.5*sigma*math.sqrt(T)
     return S*norm.pdf(d1)*math.sqrt(T)

def put_vega(S,r,K,T,sigma):
     return call_vega(S,r,K,T,sigma)

def call_rho(S,r,K,T,sigma):
    d2 = math.log(S*math.exp(r*T)/K) / (sigma*math.sqrt(T)) - 0.5*sigma*math.sqrt(T)
    return K*T*math.exp(-r*T)*norm.cdf(d2)

def put_rho(S,r,K,T,sigma):
    d2 = math.log(S*math.exp(r*T)/K) / (sigma*math.sqrt(T)) - 0.5*sigma*math.sqrt(T)
    return -K*T*math.exp(-r*T)*norm.cdf(-d2)

def Monte_carlo_european_call (S,r,K,T,sigma,N,n):
    # Function to simulate stock paths and calculate European call option price
    # Parameters:
    # S: Initial stock price
    # r: Risk-free rate
    # K: Strike price
    # T: Time to maturity
    # sigma: Volatility
    # N: Number of simulated paths
    # n: Number of time steps

    dt = T / n #Time step size
    Z = np.random.standard_normal((N,n))
    ST = S * np.exp(np.cumsum(( r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z,axis=1)) #Calculate stock price paths using geometric Brownian motion
    call_price = np.exp(-r * T) * np.maximum(ST[: ,-1] - K , 0) #Calculate discounted payoff for call option at maturity
    ST = np.insert(ST, 0, S,1) #Add initial stock price to the paths

    """Plot all simulated stock price paths"""
    t = np.linspace(0, T, n+1)
    plt.plot(t, ST.T)
    plt.xlabel("Time t")
    plt.ylabel("Stock Price ($S_t$)")
    plt.title(f"Simulated Stock Price Paths\n$S_0={S}, r={r}, \sigma={sigma}$")
    plt.show()

    return f"Estimated price of the European call with Monte Carlo method: {np.mean(call_price):.4f}"

S= 10
r= 0.05
K= 11
T= 1
sigma = 0.2
N=50000
n=1000

print("call:" +str(call_BS(S,r,K,T,sigma)))
print("call delta:" +str(round(call_delta(S,r,K,T,sigma),4)))
print("call gamma:" +str(round(call_gamma(S,r,K,T,sigma),4)))
print("call theta:" +str(round(call_theta(S,r,K,T,sigma),4)))
print("call rho:" +str(round(call_rho(S,r,K,T,sigma),4)))
print("call vega:" +str(round(call_vega(S,r,K,T,sigma),4)))

print("put:" +str(put_BS(S,r,K,T,sigma)))
print("put delta:" +str(round(put_delta(S,r,K,T,sigma),4)))
print("put gamma:" +str(round(put_gamma(S,r,K,T,sigma),4)))
print("put theta:" +str(round(put_theta(S,r,K,T,sigma),4)))
print("put rho:" +str(round(put_rho(S,r,K,T,sigma),4)))
print("put vega:" +str(round(put_vega(S,r,K,T,sigma),4)))

print(Monte_carlo_european_call (S,r,K,T,sigma,N,n))
