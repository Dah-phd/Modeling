import numpy as np
from scipy.stats import f as fdist
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
try:
    from TS_analysis import data_tests
except Exception:
    import data_tests


def linear_regression(Y, X, multiple_X=1, fix_nan=True, alfa=False, integrate=False):
    """
    Using a package, it rebuild here for ease of use.
    It is 100 % statsmodels OLS.
    Y(list[float]) - dependent variable;
    X([list[float] | float]) - independent variable;
    multiple_X [int, default=1] if there are mutiple factors (Xes), set the number of factor columns
    """
    Y = np.array(Y).reshape(-1, 1)
    X = np.array(X).reshape(-1, multiple_X)
    if integrate:
        integrations, Y, X = data_tests.stationarity.forceSTATxy(Y, X)
    if alfa:
        X = add_constant(X)
    model = OLS(Y, X, missing='drop' if fix_nan else 'none').fit()
    return model if not integrate else (model, integrations)


class causality:
    def __init__(self, Y, X):
        """
        Solves Grainger causality test to derive if fariable is causing the change in another.
        Data should start from the newst to the oldest observation.
            Y[list of float or numpy array] - the varaible row that is to be affcted, first value could be label if string;
            X[list of float or numpy array] - the factor variable row, first value could be label if string;
            self.fit() - builds the model;
            self.result - contains the results;
            (if self.fit(reverse=True))
                self.reversed_xy - contains the results for effects of Y on X, or reversed factor and result
        """
        self.Y = Y
        self.X = X
        self.YL, self.XL = 'Y', 'X'

    def fit(self, test_lags=5, labels_yx=(None, None), integrate=True, reverse=False):
        """
        Fits the data to the model.
            test_lags [int, default= 5] - lags to test for causality;
            labels_yx [(tuple, string) default = (None, None)] - tuple of two strings - label_y and label_x;
            integrate [boolean, default= True] - integrates data to reach stationarity;
            reverse [boolean, default= False] - test reverse causality.
        """
        self.fix_data()
        if integrate:
            self.integrations, self.Y, self.X = data_tests.stationarity.forceSTATxy(
                self.Y, self.X)
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
            ).fit()
            full_model = OLS(
                base,
                add_constant(regressor),
                missing='drop'
            ).fit()
            N = len(base)
            value = self._test(
                full_model.rsquared, auto_model.rsquared, N, lag
            )
            results['lags'].append(lag)
            results[self.XL+' => ' + self.YL].append(
                True if value < 0.05 else False)
            results['value'].append(value)
            results['full_model RSQ'].append(full_model.rsquared)
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

    def __init__(self, Y, X, multiple_X=1):
        """
        Y [list/array of float] - first value could be label, else it will be called Y;
        X [list/array of float/list of float] - first value could be label, else it will be X if many X1, X2 ... ;
        multiple_X [int, default = 1] - if you use list of lists for X, set to number of factors (Xes);
        """
        self.Y = Y
        self.X = X
        self.XL, YL = ['X'], 'Y'
        self.multiple_X = multiple_X

    def fit(self, length=40, integrate=True, alfa=False):
        self.fix_data()
        self.result = {'R_squared': [], 'N_observations': []}
        for label in self.XL:
            self.result[label] = []
        if alfa:
            self.result['alfa'] = []
        position = 0
        base_Y = self.Y[0:length]
        base_X = self.X[0:length]
        if alfa:
            base_X = add_constant(base_X)
        while len(base_X) >= length and len(base_Y) == length:
            model = OLS(base_Y, base_X).fit()
            self.result['R_squared'].append(model.rsquared)
            self.result['N_observations'].append(length)
            position += 1
            base_Y = self.Y[position:length+position]
            base_X = self.X[position:length+position:]
            if alfa:
                base_X = add_constant(base_X)
                for i, label in enumerate(self.XL, start=1):
                    self.result[label].append(model.params[i])
                self.result['alfa'].append(model.params[0])
            else:
                for i, label in enumerate(self.XL):
                    self.result[label].append(model.params[i])

    def fix_data(self):
        if isinstance(self.Y[0], str):
            self.YL = self.Y[0]
            self.Y.pop(0)
        if self.multiple_X == 1:
            if isinstance(self.X[0], str):
                self.XL = self.X[0]
                self.X.pop(0)
            self.X = np.array(self.X).reshape(-1, 1)
        else:
            self._fix_Xes()
        self.Y = np.array(self.Y).reshape(-1, 1)

    def _fix_Xes(self):
        self.XL = [('X' + str(x)) for x in range(self.multiple_X)]
        for x in range(self.multiple_X):
            if isinstance(self.X[0][0], str):
                self.XL.append[self.X[0][0]]
                self.X[0].pop(0)
        self.X = np.array(self.X).reshape(-1, self.multiple_X)
