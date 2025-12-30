from statsmodels.tsa.arima.model import ARIMA

def train_arima_model(series, order=(1, 1, 1)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit