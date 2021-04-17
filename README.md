#STATISTIC MODULES TIME SERIES

A library of statistical analysis, circling around prediction

It includes:

prediction models:\n
    \tLinear projection;\n
    \tAutoregressive model;\n
    \tMoving average model;\n
    \tARIMA model;\n

regression models:\n
    \tlinear_regression (statsmodel, there for convinience);\n
    \trolling (Rolling regression), determins changes in the regression statistics over time;\n
    \tcausality (Grainger causality test), determines causality in relation and direction;\n

modules.concurrent:
    ARIMA
    useful only wiht lags over 90 periods
    for AR, MA models the multiple time needed to launch ProcessPool makes concurrency useless
    