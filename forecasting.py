import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from PyQt5 import QtWidgets, uic
import sys


class data_class:

    @staticmethod
    def set_start():
        global nyears
        global lags
        global indicators
        global nindicators
        global target_variable
        global countries
        global ncountries
        global start_year
        global end_year
        global data
        global y, x
        global app, window, imp

        nyears = 10
        lags = 5

        indicators = {"gdp": "NY.GDP.MKTP.CD",
                      "population": "SP.POP.TOTL",
                      "inflation": "FP.CPI.TOTL.ZG"}
        nindicators = len(indicators)

        target_variable = "gdp"

        countries = ['us', 'gb']
        ncountries = len(countries)

        # Start and end year for the data set
        start_year = 1976
        end_year = 2019

        app = QtWidgets.QApplication(sys.argv)  # Create an instance of QtWidgets.QApplication
        window = Ui()
        imp = importData
        data = imp.get_data_from_API()
        y, x = importData.set_logarifmical(data)



    def update_data():
        global data
        global y
        global x
        data = imp.get_data_from_API()
        y, x = importData.set_logarifmical(data)


class decorate:

    def decorator_func(func):
        def wrapper(*args, **kwargs):
            Ui.print_log('Raising function...')
            Ui.print_log('{} has been raised'.format(func.__name__))
            Ui.print_log('Entering the function...')
            func(*args, **kwargs)
            Ui.print_log('Function has been succeeded')

        return wrapper


class importData(object):

    @staticmethod
    def get_data_from_API():
        template_url = "http://api.worldbank.org/v2/countries/{0}/indi"
        template_url += "cators/{1}?date={2}:{3}&format=json&per_page=999"
        # Countries should be ISO identifiers separated by semi-colon
        raw_data = pd.DataFrame()
        country_str = ';'.join(countries)
        for label, indicator in indicators.items():
            # Fill in the template URL
            url = template_url.format(country_str, indicator,
                                      start_year, end_year)

            # Request the data
            json_data = requests.get(url)

            # Convert the JSON string to a Python object
            json_data = json_data.json()
            json_data = json_data[1]

            for data_point in json_data:
                country = data_point['country']['id']

                # Create a variable for each country and indicator pair
                item = country + '_' + label

                year = data_point['date']

                value = data_point['value']

                # Append to data frame
                new_row = pd.DataFrame([[item, year, value]],
                                       columns=['item', 'year', 'value'])
                raw_data = raw_data.append(new_row)

        upd_data = raw_data.pivot('year', 'item', 'value')

        return upd_data

    def set_logarifmical(d):
        # (Runtime warning expected due to NaN)
        d = np.log(d).diff().iloc[1:, :]
        # Set NaN to zero
        d.fillna(0, inplace=True)
        # Subtract the mean from each series
        d = d - d.mean()
        # Convert to date type
        d.index = pd.to_datetime(d.index, format='%Y')
        # Put the target variable into a separate data frame
        target = d[[x for x in data.columns
                    if x.split("_")[-1] == target_variable]]

        return target, d

    def count_errors(target):
        ncountries = len(countries)
        errors = target.iloc[-nyears:] - target.shift().iloc[-nyears:]
        # Root mean squared error
        rmse = errors.pow(2).sum().sum() / (nyears * ncountries) ** .5

        # target_data  <=== target Var_modeling(set_logarifmical)


class graph:
    @decorate.decorator_func
    def plotting(data):
        for lab in indicators.keys():
            indicator = data[[x for x in data.columns
                              if x.split("_")[-1] == lab]]
            indicator.plot(title=lab)
            plt.show()



class modelling:

    @decorate.decorator_func
    def VAR_modeling(target_data):
        # Sum of squared errors
        sse = 0
        ncountries = len(countries)
        for t in range(nyears):
            # Create a VAR model
            model = VAR(target_data.iloc[t:-nyears + t], freq='AS')

            # Estimate the model parameters
            results = model.fit(maxlags=1)

            actual_values = target_data.values[-nyears + t + 1]

            forecasts = results.forecast(target_data.values[:-nyears + t], 1)
            forecasts = forecasts[0, :ncountries]
        sse += ((actual_values - forecasts) ** 2).sum()
        # Root mean squared error
        rmse = (sse / (nyears * ncountries)) ** .5

        Ui.print_log_result(rmse, 'VAR')


    @decorate.decorator_func
    def gaussian_model(target_data, da, a, r):
        ncountries = len(countries)
        gpr = GaussianProcessRegressor(kernel=RBF(r), alpha=a)
        # Number of data points for estimation/fitting for each forecast
        ndata = target_data.shape[0] - nyears - lags
        # Sum of squared errors
        sse = 0
        for t in range(nyears):

            # Observations for the target variables
            y = np.zeros((ndata, ncountries))
            # Observations for the independent variables
            X = np.zeros((ndata, lags * ncountries * nindicators))

            for i in range(ndata):
                y[i] = target_data.iloc[t + i + 1]
                X[i] = da.iloc[t + i + 2:t + i + 2 + lags].values.flatten()

            gpr.fit(X, y)

            x_test = np.expand_dims(da.iloc[t + 1:t + 1 + lags].values.flatten(), 0)
            forecast = gpr.predict(x_test)

            sse += ((target_data.iloc[t].values - forecast) ** 2).sum()
        rmse = (sse / (nyears * ncountries)) ** .5
        Ui.print_log_result(rmse, 'GPR')


class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()  # Call the inherited classes __init__ method
        uic.loadUi('forecasting.ui', self)  # Load the .ui file
        self.logs = self.findChild(QtWidgets.QTextBrowser, 'textBrowser')

        self.spin_box = self.findChild(QtWidgets.QDoubleSpinBox, 'doubleSpinBox')
        self.spin_box.setMinimum(0)
        self.spin_box.setMaximum(100)
        self.spin_box.setValue(0.1)

        self.spin_box_RBF = self.findChild(QtWidgets.QDoubleSpinBox, 'doubleSpinBox_2')
        self.spin_box_RBF.setMinimum(0)
        self.spin_box_RBF.setMaximum(20)
        self.spin_box_RBF.setValue(0.1)

        self.button_GPR = self.findChild(QtWidgets.QPushButton, 'pushButton')
        self.button_GPR.clicked.connect(lambda: window.check_boxes())
        self.button_GPR.clicked.connect(lambda: modelling.gaussian_model(y, x, self.spin_box.value(), self.spin_box_RBF.value()))

        self.button_build_VAR = self.findChild(QtWidgets.QPushButton, 'pushButton_2')
        self.button_build_VAR.clicked.connect(lambda: window.check_boxes())
        self.button_build_VAR.clicked.connect(lambda: modelling.VAR_modeling(y))

        self.button_show = self.findChild(QtWidgets.QPushButton, 'pushButton_3')
        self.button_show.clicked.connect(lambda: window.check_boxes())
        self.button_show.clicked.connect(lambda: graph.plotting(data))

        self.us = self.findChild(QtWidgets.QCheckBox, 'us')
        self.ru = self.findChild(QtWidgets.QCheckBox, 'ru')
        self.au = self.findChild(QtWidgets.QCheckBox, 'au')
        self.ca = self.findChild(QtWidgets.QCheckBox, 'ca')
        self.de = self.findChild(QtWidgets.QCheckBox, 'de')
        self.es = self.findChild(QtWidgets.QCheckBox, 'es')
        self.fr = self.findChild(QtWidgets.QCheckBox, 'fr')
        self.gb = self.findChild(QtWidgets.QCheckBox, 'gb')
        self.jp = self.findChild(QtWidgets.QCheckBox, 'jp')
        self.at = self.findChild(QtWidgets.QCheckBox, 'at')
        self.cn = self.findChild(QtWidgets.QCheckBox, 'cn')
        self.ae = self.findChild(QtWidgets.QCheckBox, 'ae')

        self.show()  # Show the GUI

    @staticmethod
    def check_boxes():
        c = []
        global countries
        if window.us.isChecked():
            c.append('us')
        if window.ru.isChecked():
            c.append('ru')
        if window.au.isChecked():
            c.append('au')
        if window.ca.isChecked():
            c.append('ca')
        if window.de.isChecked():
            c.append('de')
        if window.es.isChecked():
            c.append('es')
        if window.fr.isChecked():
            c.append('fr')
        if window.gb.isChecked():
            c.append('gb')
        if window.jp.isChecked():
            c.append('jp')
        if window.at.isChecked():
            c.append('at')
        if window.cn.isChecked():
            c.append('cn')
        if window.ae.isChecked():
            c.append('ae')
        if len(c) < 2:
            countries = ['us', 'gb']
            Ui.print_log('')
            Ui.print_log('')
            Ui.print_log('not enough countries! Entering US and GB as default...')
            Ui.print_log('')
            Ui.print_log('')
        else:
            countries = c
        data_class.update_data()


    def print_log_result(log, method):
        window.logs.append('')
        window.logs.append(str('\n\t' + '-' * 28))
        window.logs.append(str('\t| Error of ' + method + ' : ' + str(np.round(log, 4)) + '|'))
        window.logs.append(str('\t' + '-' * 28 + '\n'))
        window.logs.append('')


    def print_log(log):
        window.logs.append(log)


if __name__ == '__main__':
    data_class.set_start()
    app.exec_()  # Start the application



