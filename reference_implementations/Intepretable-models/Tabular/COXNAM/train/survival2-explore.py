import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter



# Load dataset
df = pd.read_csv("../datasets/support2.csv")

# Check event distribution
print("Event Distribution (Death vs. Censored):")
print(df["death"].value_counts(normalize=True))  # Proportion of death vs. censored

# Summary statistics of survival times
print("\nTime to event (summary):")
print(df["d.time"].describe())

# Plot survival time distributions
plt.figure(figsize=(12, 5))

# Histogram of time-to-event
sns.histplot(df[df["death"] == 1]["d.time"], bins=30, kde=True, color="red", label="Death")
sns.histplot(df[df["death"] == 0]["d.time"], bins=30, kde=True, color="blue", label="Censored")

plt.xlabel("Time to Event (Days)")
plt.ylabel("Count")
plt.title("Distribution of Survival Time")
plt.legend()
plt.savefig("survival2-explore-01.png")


from lifelines.statistics import proportional_hazard_test
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Fit a basic Cox model
df_cox = df.drop(columns=["d.time", "death"])

# Identify categorical & numerical columns
categorical_cols = df_cox.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = df_cox.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Step 1: Impute missing values in categorical columns BEFORE encoding
cat_imputer = SimpleImputer(strategy="most_frequent")
df_cox[categorical_cols] = cat_imputer.fit_transform(df_cox[categorical_cols])

# Step 2: One-hot encoding
encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
df_cox_categorical = encoder.fit_transform(df_cox[categorical_cols])
categorical_feature_names = encoder.get_feature_names_out(categorical_cols)
df_cox_categorical = pd.DataFrame(df_cox_categorical, columns=categorical_feature_names, index=df_cox.index)

# Step 3: Impute missing values in numerical columns BEFORE scaling
num_imputer = SimpleImputer(strategy="median")
df_cox_numerical_imputed = pd.DataFrame(num_imputer.fit_transform(df_cox[numerical_cols]), columns=numerical_cols, index=df_cox.index)

# Step 4: Combine numerical and categorical features
df_cox = pd.concat([df_cox_numerical_imputed, df_cox_categorical], axis=1)
df_cox.drop(columns=['hospdead','dzgroup_Coma', 'surv6m', 'surv2m', 'dzclass_Coma'], inplace=True)




df_cox["duration"] = df["d.time"]
df_cox["event"] = df["death"]





cph = CoxPHFitter()
cph.fit(df_cox, duration_col="duration", event_col="event")

# Test PH assumption
ph_test_results = proportional_hazard_test(cph, df_cox, time_transform="rank")
print(ph_test_results.summary)
