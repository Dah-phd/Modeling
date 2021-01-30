#STATISTIC MODULES TIME SERIES

A library of statistical analysis, circling around prediction
It includes:
============
modules.predict:
    Linear projection
    Autoregressive model
    Moving average model
    ARIMA model

modules.concurrent:
    ARIMA
    useful only wiht lags over 90 periods
    for AR, MA models the multiple launch of ProcessPool makes concurrency useless
    
work in progress, plans:
    ATM all prediction functions are operational but not tested enough
    so multiple calc could be done at one time, such as solving ARIMA or AutoRs ... etc