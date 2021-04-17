#STATISTIC MODULES TIME SERIES

A library of statistical analysis, circling around prediction


It includes:

prediction models:
    Linear projection;
    Autoregressive model;
    Moving average model;
    ARIMA model;

regression models:
    linear_regression (statsmodel, there for convinience);
    rolling (Rolling regression), determins changes in the regression statistics over time;
    causality (Grainger causality test), determines causality in relation and direction;

modules.concurrent:
    ARIMA
    useful only wiht lags over 90 periods
    for AR, MA models the multiple time needed to launch ProcessPool makes concurrency useless
    