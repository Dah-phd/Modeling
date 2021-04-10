import numpy as np
from scipy.stats import f as fdist
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
try:
    from TS_analysis.modules import data_tests
except Exception:
    from modules import data_tests


class causality:
    def __init__(self, Y, X):
        """
        Solves Grainger causality test to derive if fariable is causing the change in another.
        Data should start from the newst to the oldest observation.
            Y [list of float or numpy array] - the varaible row that is to be affcted, first value could be label if string;
            X [list of float or numpy array] - the factor variable row, first value could be label if string;
            self.fit() - builds the model;
            self.result - contains the results;
            (if self.fit(reverse = True)) 
                self.reversed_xy - contains the results for effects of Y on X, or reversed factor and result
        """
        self.Y = Y
        self.X = X
        self.YL, self.XL = 'Y', 'X'

    def fit(self, test_lags=5, labels_yx=(None, None), integrate=True, reverse=False):
        """
        Fits the data to the model.
            test_lags [int, default=5] - lags to test for causality;
            labels_yx [(tuple, string) default=(None,None)] - tuple of two strings - label_y and label_x;
            integrate [boolean, default=True] - integrates data to reach stationarity;
            reverse [boolean, default=False] - test reverse causality.
        """
        self.fix_data()
        if integrate:
            self.integrations, self.Y, self.X = data_tests.stationarity.forceSTATxy(
                self.Y, self.X)
            print(self.integrations)
            if self.integrations == None:
                return
        if isinstance(labels_yx, tuple):
            if labels_yx[0] != None and labels_yx[1] != None:
                self.YL, self.XL = str(labels_yx[0]), str(labels_yx[1])
        self.result = self.build(test_lags)
        if reverse:
            self.X, self.Y = self.Y, self.X
            self.XL, self.YL = self.YL, self.XL
            self.reversed_xy = self.build(test_lags)
            self.Y, self.X = self.X, self.Y
            self.XL, self.YL = self.YL, self.XL

    def build(self, test_lags):
        results = {'lags': [], self.XL+' => '+self.YL: [], 'value': [],
                   'full_model RSQ': [], 'used_datapoints': []}
        for lag in range(1, test_lags+1):
            lag_counts = lag
            passed_lags = 0
            base = np.copy(self.Y)[:-lag]
            auto_regressor = np.copy(self.Y)[lag:]
            regressor = np.hstack((auto_regressor, self.X[lag:]))
            while lag_counts > 1:
                lag_counts -= 1
                passed_lags -= 1
                auto_regressor = np.hstack(
                    (auto_regressor, self.Y[lag_counts:passed_lags]))
                regressor = np.hstack(
                    (regressor, self.Y[lag_counts:passed_lags], self.X[lag_counts:passed_lags]))
            # auto_regressor is first!!!
            auto_model = OLS(
                base,
                add_constant(auto_regressor),
                missing='drop'
            )
            full_model = OLS(
                base,
                add_constant(regressor),
                missing='drop'
            )
            auto_reg = auto_model.fit()
            caus_reg = full_model.fit()
            N = len(base)
            value = self._test(
                caus_reg.rsquared, auto_reg.rsquared, N, lag
            )
            results['lags'].append(lag)
            results['causality'].append(True if value < 0.05 else False)
            results['value'].append(value)
            results['full_model RSQ'].append(caus_reg.rsquared)
            results['used_datapoints'].append(N)
        return results

    def _test(self, r_full, r_reduced, N, df):
        # df is the sama as doubled the lags
        return 1 - fdist.cdf(
            (r_full-r_reduced)*(N-df*2-1)/(1-r_full)/df,
            df,
            (N-df*2-1)/(1-r_full)
        )

    def fix_data(self):
        if isinstance(self.Y[0], str):
            self.YL = self.Y[0]
            self.Y.pop(0)
        if isinstance(self.X[0], str):
            self.XL = self.X[0]
            self.X.pop(0)
        self.X = np.array(self.X).reshape(-1, 1)
        self.Y = np.array(self.Y).reshape(-1, 1)


class rolling():
    """
    Makes rolling regressions for the dataset, testing changes in
    beta ceficient,
    R_squared,

    """

    def __init__(self, Y, X):
        self.Y = Y
        self.X = X


def linear_regression(Y, X, alfa=True, fix_nan=True):
    """
    Using a package, it rebuild here for ease of use.
    It is 100% statsmodels OLS.
    Y (list[float]) - dependent variable;
    X ([list[float]|float]) - independent variable;
    """
    valid = check_X(X, len(Y))
    if not valid:
        return False
    if valid == 'stack':
        Y = np.array(Y).reshape(-1, 1)
        X = np.array(X)
        shape = X.shape
        X = X.reshape(shape[1], shape[0])
    if alfa:
        X = add_constant(X)
    model = OLS(Y, X, missing='drop' if fix_nan else 'none')
    return model.fit()


def check_X(X, values):
    if isinstance(X[0], list):
        if len(X[0]) == values:
            return 'stack'
    elif len(X) != values:
        return False
    return True

    # class dummy var regression
    # class rolling regression
