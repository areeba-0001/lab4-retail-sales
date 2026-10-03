import pandas as pd
import numpy as np
from scipy.stats import zscore

# 10-day retail sales dataset
data = {
    "Day": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Revenue": [45, 52, np.nan, 48, 390, 55, np.nan, 50, 47, 53]
}

df = pd.DataFrame(data)

# Report missing values
print("Missing values in Revenue:", df["Revenue"].isnull().sum())

# Fill missing values using median
median_revenue = df["Revenue"].median()
df["Revenue"] = df["Revenue"].fillna(median_revenue)

print("\nAfter median fill:")
print(df[["Day", "Revenue"]])

# Min-Max normalization
df["Revenue_MinMax"] = (
    (df["Revenue"] - df["Revenue"].min()) /
    (df["Revenue"].max() - df["Revenue"].min())
)

# Z-score standardization
df["Revenue_Zscore"] = zscore(df["Revenue"])

# IQR outlier detection
Q1 = df["Revenue"].quantile(0.25)
Q3 = df["Revenue"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df["Outlier"] = (
    (df["Revenue"] < lower_bound) |
    (df["Revenue"] > upper_bound)
)

print("\nFinal Table:")
print(df[["Day", "Revenue", "Revenue_MinMax", "Revenue_Zscore", "Outlier"]])

outlier_days = df.loc[df["Outlier"], "Day"].tolist()
print("\nIQR outliers detected on day(s):", outlier_days)