import numpy as np
from sklearn import linear_model

if __name__ == "__main__":
    import data_tests
else:
    from modules import data_tests
# ORDER IS FROM NEWEST (UP) TO OLDEST (DOWN) DATA


class ARIMA:
    """
    Class ARIMA is used to predict time series data.

    The model combines AR - autoregression and MA - moving averages,
    also if the initial data is not stationary it integrates it until so.

    The model is brute-forced, aka it will calculate all the possible models,
    such as AR1MA2, AR2MA2, AR3MA2 ... AR(lags)MA(lags),
    then select the one with least historic error.

    ####Params####:

        data: imput data list of variables, starting from newst to oldest data,
        could be numpy array.

        lags: by default 31, could be changed,
        it determines how much possible lags will be tested.
        THE CALCULATION GROW EXPONENTIALLY!

    Every instance of the class will be stored in list current_models
    (call with any self.current_models)
    """
    current_models = []

    def __init__(self, data, lags=31):
        """
        Constructor params(could be invoked):
        -------------------------------------
            self.integrations: automated - returns the integrations done
            to make the data stationary.

            self.all_models: stores information on all models solved.

            self.data: input - initial data.

            self.lags: input - lags to be tested.

            self._test_data() and self._turn_to_np() called private methods.

            self.base: generated, backup of the used data, prior numpy.

            self.best_model: generated - returns the best model,
            after testing all generated.

            self.prediction: generated - returns initial prediction
            of the found best model.

        """
        self.current_models.append(self)
        self.integrations = 0
        self.all_models = {}
        self.data = data
        self.lags = lags
        self._test_data()
        self._turn_to_np()
        self.prediction = self.predict()

    def _test_data(self):
        # Private: checks stationarity by calling other module.
        self.integrations, self.data = data_tests.forceSTAT(self.data)

    def _turn_to_np(self):
        # Private: transform into numpy array with proper shape/bachup.
        self.data = np.array(self.data).reshape(-1, 1)
        self.base = np.copy(self.data)

    def _moving_averages(self, lag):
        # returns np array list of moving averages base on the lag.
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
        '''
        Trigering the build function solves all models in order to
        find the best model, by score, then returns it as a result.
        Also generates self.all_models and self.best to store the information.
        '''
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
        self.best = self._check_all_models()
        return self.best

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
        '''
        Predict function:
        -----------------
        Use to predict expected values for the class, normal use:
        First use build, then invoke and it will generated 30 predictions.
        Starting with the most furthest and reachin to the closes (t+30...t+1).

        Params:
        -------
            model: default 'best', use the self.best model,
            could be given specific model.
            Example AR1MA1 or AR1I1MA1 in that case the function
            could be triggered prior build and make prediction.
            ### DATA STILL WILL BE INTEGRATED BY self.integration ###
        '''
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
        if self.best:
            self.best_model
        else:
            'Model builder'


class linearProjection:
    '''
    Linear projection is modeling the expected values of a timeseries
    by using as factor the change of the period i.e. (1,2,3,4,5...n).

    current_models (invoked by self.current_models) - lists all instances
    of the class.

    #######
    self.build() is used to trigger the solution of the model.
        generates:
            self.rsq = score of the model
            self.intercept = constant queficient
            self.beta = coeficient of the relation
        !!! The model dose not consider the possibility of 0 intercept !!!

    self.predict() is used to generate predictions.

    #######
    Required params:

    data: The input data should include list or numpy array starting
    with the newest data to the oldest.

    integrate: optional bool, default = False.
    Determines should predict force stationarity.
    '''
    current_models = []

    def __init__(self, data, integrate=False):
        '''
        Constructor params:
        ------------------
        self.data: input data - list/np.array

        self.periods: generated, to track lenght of the timeseries

        self.integrations: generated, states the number of
        integrations if any.

        self._turn_to_np(): private method
        buils self.base - copy of the initial data.
        '''
        self.current_models.append(self)
        self.data = data
        self.periods = len(self.data)+1
        self.integrations = 0
        self._turn_to_np(integrate)

    def _turn_to_np(self, integrate):
        # convert to numpy and makes a backup
        if integrate:
            self.integrations, self.data = data_tests.forceSTAT(self.data)
        self.data = np.array(self.data).reshape(-1, 1)
        self.base = np.copy(self.data)

    def build(self):
        '''
        self.build(), no params is used to initialize the solution of the
        linear model, if not triggered the predict function will automate it.
        '''
        factor = [t for t in range(1, self.periods)]
        factor.reverse()
        factor = np.array(factor).reshape(-1, 1)
        model = linear_model.LinearRegression().fit(factor, self.data)
        self.rsq, self.intercept, self.beta = (model.score(factor,
                                                           self.data**2),
                                               model.intercept_[0],
                                               model.coef_[0][0])

    def predict(self, periods=30):
        '''
        self.predict is used to generate predictions from the build model,
        if no model is build it will build it. Also stores the predictions in
        self.prediction

        Params:
        periods: int, how long should the prediction be, default is 30 periods
        '''
        if not self.rsq:
            self.build()
        x = self.periods
        t = []
        predictions = []
        for t in range(periods+1):
            y = x*self.beta+self.intercept
            predictions.insert(0, y)
            t.insert(0, x)
            x += 1
        predictions = np.array(predictions).reshape(-1, 1)
        t = np.array(t).reshape(-1, 1)
        self.prediction = {'R': self.rsq,
                           'periods(t)': t, 'prediction': predictions}
        return self.prediction


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
