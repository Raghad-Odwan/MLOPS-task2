# Notebook 3: Train / validation / test split.
# Split randomly (stratified on the label) so all three sets share the same
# label ratio, as instructed. A random split is simpler to reason about here
# since the label ratio changes noticeably over time in this dataset (Olist's
# delivery performance improved over the two years covered), which would make
# a time-based split unstable for a first model.

import pandas as pd
from sklearn.model_selection import train_test_split

labeled = pd.read_csv("labeled_table.csv", parse_dates=["order_purchase_timestamp"])

print("Total rows:", labeled.shape)
print("Date range:", labeled["order_purchase_timestamp"].min(), "to", labeled["order_purchase_timestamp"].max())

# First split off the test set, then split the remainder into train and validation.
# stratify=labeled["is_late"] keeps the same late ratio in every split.
train_val, test = train_test_split(
    labeled, test_size=0.15, stratify=labeled["is_late"], random_state=42
)
train, val = train_test_split(
    train_val, test_size=0.1765, stratify=train_val["is_late"], random_state=42
)
# 0.1765 of the remaining 85% is about 15% of the original data, giving a 70/15/15 split.

print("\nTrain shape:", train.shape)
print("Val shape:", val.shape)
print("Test shape:", test.shape)

print("\nLabel balance in train:")
print(train["is_late"].value_counts(normalize=True))
print("\nLabel balance in val:")
print(val["is_late"].value_counts(normalize=True))
print("\nLabel balance in test:")
print(test["is_late"].value_counts(normalize=True))

print("\nDate range in train:", train["order_purchase_timestamp"].min(), "-", train["order_purchase_timestamp"].max())
print("Date range in val:", val["order_purchase_timestamp"].min(), "-", val["order_purchase_timestamp"].max())
print("Date range in test:", test["order_purchase_timestamp"].min(), "-", test["order_purchase_timestamp"].max())

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)
print("\nArtifacts saved: train.csv, val.csv, test.csv")