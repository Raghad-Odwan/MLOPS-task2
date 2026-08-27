# Notebook 2: Create the label (late vs on-time delivery).

import pandas as pd

ml_table = pd.read_csv("ml_table.csv", parse_dates=[
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
])

print("Loaded ml_table:", ml_table.shape)

# Only orders that were actually delivered have a real delivery date to compare.
print("\nOrder status counts:")
print(ml_table["order_status"].value_counts())

delivered = ml_table[ml_table["order_status"] == "delivered"].copy()
print("\nDelivered orders:", delivered.shape)
print("Missing delivered date among delivered orders:", delivered["order_delivered_customer_date"].isna().sum())

delivered = delivered.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"])

# Build the label: 1 = late, 0 = on time
delivered["is_late"] = (delivered["order_delivered_customer_date"] > delivered["order_estimated_delivery_date"]).astype(int)

# Sanity check the label on a few real orders
print("\nSample check:")
print(delivered[["order_id", "order_delivered_customer_date", "order_estimated_delivery_date", "is_late"]].head(10))

# Class distribution
print("\nClass distribution:")
print(delivered["is_late"].value_counts())
print(delivered["is_late"].value_counts(normalize=True))

late_ratio = delivered["is_late"].mean()
print(f"\nLate ratio: {late_ratio:.3f}")
if late_ratio < 0.2 or late_ratio > 0.8:
    print("This looks like an imbalanced classification problem.")
else:
    print("Classes are reasonably balanced.")

# Save artifact: the labeled table
delivered.to_csv("labeled_table.csv", index=False)
print("\nArtifact saved: labeled_table.csv")
print("Final labeled table shape:", delivered.shape)
