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
    working on cython version, mainly for expirience but will be usefull especially for AutoReg and MovingAvg as they get quite time complex