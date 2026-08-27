# MLOps Training - Task 2

## From Tables to Notebooks

This project is the second task of the MLOps training track. The goal is to move from the raw database tables built in Task 1 to a complete first modeling pipeline, organized as six separate notebooks, each with a single responsibility and its own saved artifacts.

Repository link: https://github.com/Raghad-Odwan/MLOPS-task2

## Where We Started

From Task 1, the database was already running locally with all nine Olist tables loaded, and the target problem was defined: predict whether a delivered order will arrive after its estimated delivery date.

## Project Structure

```
01_read_and_join.py         Reads the database tables and builds one row per order
02_create_labels.py         Builds the late delivery label
03_train_val_test_split.py  Splits the data into train, validation, and test sets
04_eda.py                   Explores the training data in detail
05_feature_engineering.py   Builds and saves the final features and transformers
06_train_tune_evaluate.py   Trains, tunes, and evaluates the model
```

Each notebook only reads the artifacts produced by the notebook before it, and only writes artifacts for the notebook after it.

## Notebook 1: Read and Join the Tables

This notebook reads every table from the database on its own, checks row counts and duplicates, then aggregates order_items and order_payments to one row per order before joining everything together.

Key finding: both order_items and order_payments have more than one row for many orders, since an order can contain several products or be paid in several installments. These tables were aggregated (summed and counted) before joining, so the final table has exactly one row per order.

Result: a single ML table with 99,441 rows, one row per order, combining order, customer, product category, and payment information.

Artifact: ml_table.csv

## Notebook 2: Create the Labels

The label (is_late) was built by comparing order_delivered_customer_date to order_estimated_delivery_date, but only for orders with status "delivered", since orders that were never delivered have no real delivery date to compare against.

The label was checked manually against a sample of real orders before being trusted.

Class distribution:

| Label | Count | Percentage |
|---|---|---|
| On time (0) | 88,644 | 91.9% |
| Late (1) | 7,826 | 8.1% |

This is a clearly imbalanced classification problem, which shaped the choices made in later notebooks (evaluation metric, class weighting).

Artifact: labeled_table.csv

## Notebook 3: Train, Validation, Test Split

### First attempt: split by time

The first version of this notebook split the data by order_purchase_timestamp, using the oldest 70% of orders for training, the next 15% for validation, and the most recent 15% for testing. This is a common choice for a real deployment scenario, where a model is always trained on past orders and used to predict future ones.

However, this split produced a late ratio that was very different across the three sets:

| Split | Late ratio |
|---|---|
| Train | 9.0% |
| Validation | 5.3% |
| Test | 6.6% |

This mismatch turned out to cause a real problem in the final model (see Notebook 6 below).

### Final approach: random split, stratified by label

The task instructions specifically say to keep the same label ratio across splits when splitting randomly. Based on that, and on the imbalance found above, the final version of this notebook uses a random split stratified on the label, so all three sets share almost exactly the same late ratio.

Final split (70% train, 15% validation, 15% test):

| Split | Rows | Late ratio |
|---|---|---|
| Train | 67,526 | 8.11% |
| Validation | 14,473 | 8.11% |
| Test | 14,471 | 8.11% |

Artifacts: train.csv, val.csv, test.csv

## Notebook 4: Exploratory Data Analysis

This notebook works only on the training split.

Main points checked:

- Data types, shape, and memory usage of the training set
- Missing values: mostly in the product category columns (about 1,000 rows out of 67,526), coming from orders whose items could not be matched to a product
- Numerical columns (price, freight, payment value, number of items) are all right-skewed, with a small number of very high-value orders
- Categorical columns: credit card is by far the most common payment type, and SP (Sao Paulo) is the most common customer state by a large margin
- Late ratio by payment type: fairly similar across payment types, with boleto slightly higher than the rest
- Late ratio by customer state: noticeably different across states, ranging from about 5% in Parana and Minas Gerais to about 13-14% in Bahia and Rio de Janeiro
- Delivery time: about 12 days on average, but with a long tail up to 209 days for a few extreme cases
- Late ratio by weekday: relatively stable across the week, with a slightly higher rate for orders placed on Monday

Main takeaway: customer location (state) and order value/size look like the most promising features for the model, while payment type and weekday show a weaker relationship with the label.

Artifacts: eda_hist_*.png (histograms for each numerical column), eda_late_ratio_by_state.png, eda_findings.txt

## Notebook 5: Feature Engineering

Features were built using only information available at the time the order is placed. The following were explicitly excluded from the feature set to avoid data leakage: the actual delivery date, the review score, and anything else that only becomes known after delivery.

Steps:

- Added time-based features from the purchase timestamp: weekday, month, hour
- Numerical features (price, freight, item count, payment value, installments, and the new time features) were imputed with the median and scaled, both fitted on the training split only
- Categorical features (customer state, payment type, product category) were imputed with a constant value and one-hot encoded, again fitted on the training split only
- The same fitted imputers, scaler, and encoder were then applied to the validation and test splits without refitting

This matches the rule that in production, the pipeline must reuse the same fitted objects and never refit on new data.

Final feature table: 111 columns (mostly from one-hot encoding customer states and product categories).

Artifacts: features_train.csv, features_val.csv, features_test.csv, num_imputer.joblib, scaler.joblib, cat_imputer.joblib, encoder.joblib, feature_list.txt

## Notebook 6: Train, Tune, Evaluate

A simple baseline (always predict "on time", the majority class) was used first, giving an F1 score of 0 on the late class, since it never predicts a late order at all.

A Random Forest classifier with class_weight="balanced" was then trained to account for the imbalance, and tuned by comparing a few values of max_depth on the validation split. ROC-AUC and F1 were used as the main metrics instead of accuracy, since accuracy would be misleading on an imbalanced problem like this one.

### Result with the time-based split (first attempt)

| Metric | Validation | Test |
|---|---|---|
| F1 | 0.14 | 0.065 |
| ROC-AUC | 0.60 | 0.38 |

A test ROC-AUC below 0.50 means the model performed worse than random guessing on the test set. This was caused by the mismatch in label ratio between the time-based splits described in Notebook 3: the model learned patterns from a period with a higher late ratio and was evaluated on a period with a different, lower late ratio and likely different underlying causes of delay.

### Result with the random stratified split (final version)

| Metric | Validation | Test |
|---|---|---|
| F1 | 0.28 | 0.27 |
| Precision | 0.17 | 0.20 |
| Recall | 0.51 | 0.39 |
| ROC-AUC | 0.70 | 0.70 |

Switching to a random, stratified split fixed the problem: validation and test performance are now close to each other, and a ROC-AUC of about 0.70 shows the model is meaningfully better than random guessing at identifying late orders, which is a reasonable result for a first baseline model.

Best model: Random Forest with max_depth=15, class_weight="balanced".

Artifacts: final_model.joblib, results_summary.txt

## Conclusion

All six notebooks run in order from a clean start, each one reading the artifacts of the step before it and saving its own artifacts for the step after it. The train/validation/test split decision was revisited after discovering a real problem (a large difference in label ratio across time-based splits, leading to an unreliable test score), and switching to a random split stratified on the label solved it, in line with the task instructions. The final model reaches a validation and test ROC-AUC of about 0.70, clearly better than the majority-class baseline, and provides a reasonable first result to build on in the next stages of the project.
