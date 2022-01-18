import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import re

data = pd.read_csv("data/output/revised/single_year/0118_1718_claims_weight_db_matched.csv")

# Data for dumpster loads only
truck_mask = (data["subcategory_prev"] == "GENERAL DEMOLITION") & data["item_description"].str.contains("DUMPSTER LOAD")
truck_data = data[truck_mask].copy()

# Mask for items associated with valid pentatonic IDs
id_matched_mask = data["pentatonic_id"].notna() & (data["pentatonic_id"] != "")

data["weight_lbs"] = data["weight_lbs"].astype('str').str.replace(',', '.')
data["weight_ustons"] = data["weight_ustons"].astype('str').str.replace(',', '.')
data["volume_cf"] = data["volume_cf"].astype('str').str.replace(',', '.')
data["volume_cy"] = data["volume_cy"].astype('str').str.replace(',', '.')
data["item_quantity"] = data["item_quantity"].astype('str').str.replace(',', '.')

# Estimated weight of claim items
data["item_weight_lbs"] = data["weight_lbs"].astype('float') * data["item_quantity"].astype('float')
data["item_weight_ustons"] = data["weight_ustons"].astype('float') * data["item_quantity"].astype('float')
data["item_volume_cf"] = data["volume_cf"].astype('float') * data["item_quantity"].astype('float')
data["item_volume_cy"] = data["volume_cy"].astype('float') * data["item_quantity"].astype('float')


# Dumpster load items have patterns, e.g. "DUMPSTER LOAD APPROX 12 YARDS 1-3 TONS DEBRIS" or DUMPSTER LOAD APPROX 20 YARDS 4 TONS DEBRIS
# In order to calculate weight for each truck, numbers before TONS are extracted

# Mask for items with ranged numbers, such as 1-3 TONS, 5-7 TONS, etc. 
range_mask = truck_data["item_description"].str.contains('\s+\d\s*-\s*\d\s+TONS', flags=re.IGNORECASE, regex=True)

# Mask for items with exact numbers, such as 4 TONS, etc. 
precise_mask = truck_data["item_description"].str.contains('(\s+\d\s+TONS)(?!\s+\d\s*-\s*\d\s+TONS)', flags=re.IGNORECASE, regex=True)

# Since for majority of trucks range of weights is given, the corresponding columns are introduced
# In case of single number, lower and upper weights are the same
truck_data[["lower_truck_weight", "upper_truck_weight"]] = 0
truck_data.loc[range_mask, ["lower_truck_weight", "upper_truck_weight"]] = truck_data.loc[range_mask, "item_description"].str.extract(r'\s+(\d)\s*-\s*(\d)\s+TONS', flags=re.IGNORECASE).values
truck_data.loc[precise_mask, "lower_truck_weight"] = truck_data.loc[precise_mask, "upper_truck_weight"] = truck_data.loc[precise_mask, "item_description"].str.extract(r'\s+(\d)\s+TONS', flags=re.IGNORECASE).values

# Calculate average weight for each truck and mulptiply by number of trucks
truck_data["average_truck_weight"] = 0.5*(truck_data["lower_truck_weight"].astype(float) + truck_data["upper_truck_weight"].astype(float))*truck_data["item_quantity"]

# Total weight of all trucks per claim ID
truck_weight_data = truck_data.groupby("claim_id").agg(total_truck_weight=("average_truck_weight", "sum")).reset_index()

# Merge total weight truck data with main claim data
data = data.merge(truck_weight_data[["claim_id", "total_truck_weight"]], on="claim_id", how="left")

# Function to calculate fraction of items matched with weights DB
def matching_fraction(col):
    
    mask = col.notna() & (col != 0) & (col != "")
    
    return round(mask.sum()/col.shape[0], 2)

# Aggregate claim data for each claim ID
claim_id_data = data[~truck_mask].groupby(["claim_id"]).agg(
    total_claims=("item_description", "count"),
    total_items=("item_quantity", "sum"), #changed from item_quantity to count
    total_truck_weight=("total_truck_weight", "last"),
    weight_estimation_lbs=("item_weight_lbs", "sum"),
    weight_estimation_ustons=("item_weight_ustons", "sum"),
    volume_estimation_cf=("item_volume_cf", "sum"),
    volume_estimation_cy=("item_volume_cy", "sum"),
    ID_matched_fraction=("pentatonic_id", matching_fraction),
    matched_fraction=("item_weight_lbs", matching_fraction)
)

claim_id_data.to_csv(("data/output/count_temp_save.csv"))

# Calculate difference between total truck weight and total estimated weight of claims in US tonnes
claim_id_data["excessive_truck_weight"] = claim_id_data["total_truck_weight"].astype('float') - claim_id_data["weight_estimation_ustons"].astype('float')
claim_id_data["excessive_truck_weight_percentage"] = claim_id_data["excessive_truck_weight"]/claim_id_data["total_truck_weight"]*100
claim_id_data["abs_excessive_truck_weight_percentage"] = claim_id_data["excessive_truck_weight_percentage"].abs()
average_deviation = claim_id_data["abs_excessive_truck_weight_percentage"].mean()

# Saving data to Excel
claim_id_data.to_excel(("data/output/0118_claim_id_weight_comparison.xlsx"))

plot = claim_id_data["excessive_truck_weight_percentage"].hist(bins=100, range=(-100, 100), grid=True)

plt.xlabel('Deviation in Truck Weight Estimation (in percent)', fontsize=30)
plt.ylabel('Frequency', fontsize=30)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)

plt.savefig('data/output/0118_complete_percent_count_excessive_truck_weight_frequency.png')

plt.yscale("log")

plt.savefig('data/output/0118_complete_percent_count_excessive_truck_weight_frequency_logscale.png')

lines = ['Overall Average Deviation: ' + str(average_deviation)]
with open('data/output/metrics.txt', 'w') as f:
    f.writelines(lines)