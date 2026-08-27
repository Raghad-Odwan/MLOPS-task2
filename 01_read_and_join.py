# Notebook 1: Read tables from the database and join them into one ML table.
# Goal: one row per order at the end.

import pandas as pd
from sqlalchemy import create_engine

DB_USER = "olist"
DB_PASSWORD = "olist123"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "olist_db"

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Read every table on its own
orders = pd.read_sql("SELECT * FROM orders", engine)
customers = pd.read_sql("SELECT * FROM customers", engine)
order_items = pd.read_sql("SELECT * FROM order_items", engine)
order_payments = pd.read_sql("SELECT * FROM order_payments", engine)
products = pd.read_sql("SELECT * FROM products", engine)
sellers = pd.read_sql("SELECT * FROM sellers", engine)
category_translation = pd.read_sql("SELECT * FROM category_translation", engine)

print("orders:", orders.shape)
print("customers:", customers.shape)
print("order_items:", order_items.shape)
print("order_payments:", order_payments.shape)
print("products:", products.shape)
print("sellers:", sellers.shape)

# Check row counts, keys, duplicates
print("\nUnique order_id in orders:", orders["order_id"].nunique())
print("Duplicate order_id in orders:", orders["order_id"].duplicated().sum())

print("\nRows per order in order_items:")
print(order_items.groupby("order_id").size().describe())

print("\nRows per order in order_payments:")
print(order_payments.groupby("order_id").size().describe())

# Aggregate order_items to one row per order
items_agg = order_items.groupby("order_id").agg(
    n_items=("order_item_id", "count"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
).reset_index()

main_product = (
    order_items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    .sort_values("price", ascending=False)
    .drop_duplicates(subset="order_id", keep="first")[["order_id", "product_category_name"]]
)
items_agg = items_agg.merge(main_product, on="order_id", how="left")
items_agg = items_agg.merge(category_translation, on="product_category_name", how="left")

print("\nitems_agg shape:", items_agg.shape)

# Aggregate order_payments to one row per order
payments_agg = order_payments.groupby("order_id").agg(
    total_payment_value=("payment_value", "sum"),
    n_payment_installments=("payment_installments", "max"),
).reset_index()

main_payment_type = (
    order_payments.sort_values("payment_value", ascending=False)
    .drop_duplicates(subset="order_id", keep="first")[["order_id", "payment_type"]]
)
payments_agg = payments_agg.merge(main_payment_type, on="order_id", how="left")

print("payments_agg shape:", payments_agg.shape)

# Join everything into one ML table, one row per order
ml_table = orders.merge(customers, on="customer_id", how="left")
ml_table = ml_table.merge(items_agg, on="order_id", how="left")
ml_table = ml_table.merge(payments_agg, on="order_id", how="left")

print("\nFinal ml_table shape:", ml_table.shape)
print("Unique orders in ml_table:", ml_table["order_id"].nunique())
print("\nColumns:", list(ml_table.columns))
print("\nSample rows:")
print(ml_table.head(3))

# Save artifact
ml_table.to_csv("ml_table.csv", index=False)
print("\nArtifact saved: ml_table.csv")
