import logging

# module_logger = logging.getLogger('model')


class FuzzyVolatilityModel:
    def __init__(self, train_data, test_data=None, n_test=None):
        self.logger = logging.getLogger('model.FuzzyVolatilityModel')
        self.logger.info('creating an instance of FuzzyVolatilityModel')

        self.train_data = train_data.copy()

        if test_data is not None:
            self.test_data = test_data.copy()
        else:
            if n_test is not None:
                self.test_data = self.train_data.iloc[-n_test:].copy()
                self.train_data = self.train_data.iloc[:-n_test].copy()

        self.logger.debug(f'test_data: {test_data}\n'
                          f'train_data: {train_data}')

    def push(self, observation):
        pass

    def forecast(self, test_data):
        pass
