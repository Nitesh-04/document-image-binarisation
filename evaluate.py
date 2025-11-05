import pandas as pd

df = pd.read_csv('results_final.csv')

accuracy = (df['difference'].abs() <= 5).mean()
mae = df['difference'].abs().mean()
mse = (df['difference'] ** 2).mean()
rmse = mse ** 0.5

print("Final Metrics After Cleaning:")
print(f"Total samples: {len(df)}")
print(f"Accuracy (±5 tolerance): {accuracy * 100:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
