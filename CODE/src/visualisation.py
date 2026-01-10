import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def plot_forecast_vs_actual(y_true, y_pred, title):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="orange")
    plt.title(title)

    # Calcolo metriche
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Mostra le metriche nel grafico
    metrics_text = f"MAE: {mae:.4f}  |  MSE: {mse:.4f}  |  RMSE: {rmse:.4f}"
    plt.suptitle(metrics_text, fontsize=10, y=0.93, color="gray")

    plt.xlabel("Time")
    plt.ylabel("Scaled Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()