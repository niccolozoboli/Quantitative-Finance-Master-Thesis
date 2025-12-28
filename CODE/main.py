from src.data_loader import download_commodity
from src.preprocessing import apply_log_return, scale_features
from src.models import train_arima_model, create_lagged_features, train_random_forest
from src.plot import plot_original_vs_log, plot_arima_fit

# STEP 1: Load
df = download_commodity("GC=F")  # Gold Futures

# STEP 2: Preprocessing
df = apply_log_return(df)
df = scale_features(df)

# STEP 3: Visuals
plot_original_vs_log(df)

# STEP 4: ARIMA
arima_model = train_arima_model(df["log_return"])
plot_arima_fit(df, arima_model)

# STEP 5: Random Forest
df_ml = create_lagged_features(df)
X = df_ml[[f"lag_{i}" for i in range(1, 6)]]
y = df_ml["log_return"]
rf_model = train_random_forest(X, y)