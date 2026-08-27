# Notebook 5: Feature engineering.
# Fit every transformation on the training split only, then apply to val and test.
# Save every fitted object, not just the output table.

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

train = pd.read_csv("train.csv", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"])
val = pd.read_csv("val.csv", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"])
test = pd.read_csv("test.csv", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"])

# Only use information available at prediction time (order placed).
# Do not use order_delivered_customer_date or review score as a feature: they only exist
# after delivery and would leak the label.

def add_time_features(df):
    df = df.copy()
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    return df

train = add_time_features(train)
val = add_time_features(val)
test = add_time_features(test)

numeric_features = ["total_price", "total_freight", "n_items", "total_payment_value",
                     "n_payment_installments", "purchase_weekday", "purchase_month", "purchase_hour"]
categorical_features = ["customer_state", "payment_type", "product_category_name_english"]

# Impute numeric missing values using the training median
num_imputer = SimpleImputer(strategy="median")
train_num = num_imputer.fit_transform(train[numeric_features])
val_num = num_imputer.transform(val[numeric_features])
test_num = num_imputer.transform(test[numeric_features])

# Scale numeric features using training statistics
scaler = StandardScaler()
train_num = scaler.fit_transform(train_num)
val_num = scaler.transform(val_num)
test_num = scaler.transform(test_num)

# Impute categorical missing values with a constant, then one-hot encode
cat_imputer = SimpleImputer(strategy="constant", fill_value="unknown")
train_cat = cat_imputer.fit_transform(train[categorical_features])
val_cat = cat_imputer.transform(val[categorical_features])
test_cat = cat_imputer.transform(test[categorical_features])

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
train_cat_enc = encoder.fit_transform(train_cat)
val_cat_enc = encoder.transform(val_cat)
test_cat_enc = encoder.transform(test_cat)

feature_names = numeric_features + list(encoder.get_feature_names_out(categorical_features))

X_train = np.hstack([train_num, train_cat_enc])
X_val = np.hstack([val_num, val_cat_enc])
X_test = np.hstack([test_num, test_cat_enc])

y_train = train["is_late"].values
y_val = val["is_late"].values
y_test = test["is_late"].values

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("Number of features:", len(feature_names))

# Save the final feature tables
pd.DataFrame(X_train, columns=feature_names).assign(is_late=y_train).to_csv("features_train.csv", index=False)
pd.DataFrame(X_val, columns=feature_names).assign(is_late=y_val).to_csv("features_val.csv", index=False)
pd.DataFrame(X_test, columns=feature_names).assign(is_late=y_test).to_csv("features_test.csv", index=False)

# Save the fitted transformers, not just the output table.
# In production, the pipeline must load these same objects and never re-fit on new data.
joblib.dump(num_imputer, "num_imputer.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(cat_imputer, "cat_imputer.joblib")
joblib.dump(encoder, "encoder.joblib")

with open("feature_list.txt", "w") as f:
    f.write("\n".join(feature_names))

print("\nArtifacts saved: features_train.csv, features_val.csv, features_test.csv, "
      "num_imputer.joblib, scaler.joblib, cat_imputer.joblib, encoder.joblib, feature_list.txt")
