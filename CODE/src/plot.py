#plot

import matplotlib.pyplot as plt

def plot_original_vs_log(df):
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    df["Close"].plot(ax=ax[0], title="Original Price")
    df["log_return"].plot(ax=ax[1], title="Log Return")
    plt.tight_layout()
    plt.show()

def plot_arima_fit(df, model_fit):
    pred = model_fit.predict(start=0, end=len(df)-1)
    df["predicted"] = pred
    df[["log_return", "predicted"]].plot(figsize=(12, 4), title="ARIMA Fit vs Real")
    plt.show()