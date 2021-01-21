import numpy as np
import data_tests
from sklearn import linear_model

# ORDER IS FROM NEWEST (UP) TO OLDEST (DOWN) DATA


class ARIMA:
    current_models = []

    def __init__(self, data, lags=31):
        self.current_models.append(self)
        self.integrations = 0
        self.all_models = {}
        self.data = data
        self.lags = lags
        self._test_data()
        self._turn_to_np()
        self.inform()
        self.best_model = self.build()
        self.prediction = self.predict()

    def _test_data(self):
        # checks stationarity
        self.integrations, self.data = data_tests.forceSTAT(self.data)

    def _turn_to_np(self):
        ''' transform into numpy array for ease of use and makes a
        backup of the base data (potentially)'''
        self.data = np.array(self.data).reshape(-1, 1)
        self.base = np.copy(self.data)

    def _moving_averages(self, lag):
        # returns np array list of moving averages
        result = []
        for x, _ in enumerate(self.data[lag:]):
            result.append(np.mean(self.data[1+x:lag+x]))
        return np.array(result).reshape(-1, 1)

    def _equlize(self, AR, t1, MA, t):
        # balances the lenght of imput data
        if t > t1:
            base = np.array[:-t]
            AR = AR[:-(t-t1)]
        elif t1 >= t:
            base = np.array[:-t1]
            MA = MA[:-(t1-t)]
        else:
            return 'Somethin went wrong'
        factor = np.hstack(AR, MA)
        return base, factor

    def _check_all_models(self):
        # returns the best model
        key, spec = 'model', {'R': 0}
        for t in self.all_models.items():
            if spec < t[1]['R']:
                key, spec = t[0], t[1]
        if key != 'model':
            return {key: spec}

    def build(self):
        # finds the best model
        for t in range(2, self.lags):
            MA = self._moving_averages(t)
            for t1 in range(1, self.lags):
                AR = self.data[t1:]
                base, factor = self._equlize(AR, t1, MA, t)
                model = linear_model.LinearRegression(
                    fit_intercept=False).fit(factor, base)
                regresion_values = {'R': model.score(factor, base)**2,
                                    'AR': model.coef_[0][0],
                                    'MA': model.coef_[0][1]}
            self.all_models['AR'+str(t1) +
                            'I'+str(self.integrations) +
                            'MA'+str(t)
                            ] = regresion_values
        best = self._check_all_models()
        return best

    def _decode_key(self, key):
        if 'I' in key:
            AR, MA = key.split('I')
            AR = int(AR[2:])
            MA = int(MA.split('MA')[-1])
            key = (AR, MA)
        elif 'MA' in key:
            AR, MA = key.split('MA')
            AR
            key = (AR, MA)
        else:
            key = ('broken key', 'broken key')
        return key

    def _key_integrity(self, key):
        # checks for expected mistakes in the key and corrects
        # if possible and close to normal
        if 'I' in key:
            key = key.split('I')
            MA = key[1].split('MA')
            key = key[0]+'I'+str(self.integrations)+'MA'+MA[1]
        elif 'MA' in key:
            key = key.split('MA')
            key = key[0]+'I'+str(self.integrations)+key[1]
        else:
            key = 'broken key'

    def predict(self, model='best', periods=31):
        # retrunt periods t+n .... t and same predictions
        if model == 'best':
            # Not very pretty but it works, look for more elegant way ... if po
            key = next(iter(self.best_model.keys()))
            model_dict = self.best_model[key]
        else:
            key = self._key_integrity(model)
            model_dict = self.all_models[key]
        AR, MA = self._decode_key(key)
        data = AR if AR > MA else MA
        data = self.data[:data]
        periods = []
        for t in range(periods):
            periods.insert(0, t)
            result = data[AR-1]*model_dict['AR'] + \
                np.mean(data[:MA]*model_dict['MA'])
            data = np.insert(data, 0, result, 0)
        return {'key': key,
                'periods(t)': np.array(periods).reshape(-1, 1),
                'prediction': data[:periods+1]}
        # + 1 due to indexing not including final value

    def __str__(self):
        # print the R2 of the best model and the model itself
        self.best_model

    @staticmethod
    def infrom():
        # show some information about the current class (base functions)
        text = 'information'
        print(text)


class linearProjection:
    current_models = []

    def __init__(self, data, integrate=False):
        self.current_models.append(self)
        self.data = data
        self.periods = len(self.data)+1
        self.integrations = 0
        self._turn_to_np(integrate)
        self.build()
        self.prediction = self.predict()
        self.inform()

    def _turn_to_np(self, integrate):
        # convert to numpy and makes a backup
        if integrate:
            self.integrations, self.data = data_tests.forceSTAT(self.data)
        self.data = np.array(self.data).reshape(-1, 1)
        self.base = np.copy(self.data)

    def build(self):
        factor = [t for t in range(1, self.periods)]
        factor.reverse()
        factor = np.array(factor).reshape(-1, 1)
        model = linear_model.LinearRegression().fit(factor, self.data)
        self.rsq, self.intercept, self.beta = (model.score(factor,
                                                           self.data**2),
                                               model.intercept_[0],
                                               model.coef_[0][0])

    def predict(self, periods=31):
        x = self.periods
        t = []
        predictions = []
        for t in periods:
            y = x*self.beta+self.intercept
            predictions.insert(0, y)
            t.insert(0, x)
            x += 1
        predictions = np.array(predictions).reshape(-1, 1)
        t = np.array(t).reshape(-1, 1)
        return {'R': self.rsq, 'periods(t)': t, 'prediction': predictions}

    @ staticmethod
    def inform():
        text = 'information'
        print(text)


class _simple_lag:
    current_models = []
    # parent class

    def __init__(self, data, lags=30, n_factors=3, integrate=True):
        self.current_models.append(self)
        self.data = data
        self.lags = lags+1
        self.n_factors = n_factors
        self._turn_to_np

    def _turn_to_np(self, integrate):
        # convert to numpy and makes a backup
        if integrate:
            self.integrations, self.data = data_tests.forceSTAT(self.data)
        self.data = np.array(self.data).reshape(-1, 1)
        self.base = np.copy(self.data)

    def _check_all_models(self):
        # under the condition that we use dict starting with the model than R
        self.best_model = {'model': '', 'R': 0}
        for t in self.all_models:
            if t['R'] > self.best_model['R']:
                self.best_model = t
        if self.best_model['model'] == '':
            print('No models to check')

    def _cascade(self):
        # should be recurring function
        for t in range(1, self.lags):
            pass
            # something to think about


class AutoReg(_simple_lag):
    # first child using multiple autoregressive factors (3 def)
    # make predictions
    def __init__(self, data, lags=31, n_factors=3, integrate=True):
        super().__init__(data, lags=31, n_factors=3, integrate=True)
        self.build()

    def build(self):
        self._cascade(self.n_factors)
