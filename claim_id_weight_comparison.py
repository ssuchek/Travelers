import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

from config import config

import numpy as np
import pandas as pd

import re

# Path to input matched data
filepath = "{0}/{1}/claims_weight_db_matched_weights_{1}.csv".format(config["data"]["output_dir"], config["data"]["db_version"])

data = pd.read_csv(filepath)

# Data for dumpster loads
truck_mask = (data["subcategory_prev"] == "GENERAL DEMOLITION") & data["item_description"].str.contains("DUMPSTER LOAD")
truck_data = data[truck_mask].copy()

# Data for pickup trucks loads
pickup_mask = (data["subcategory_prev"] == "GENERAL DEMOLITION") & data["item_description"].str.contains("HAUL DEBRIS - PER PICKUP TRUCK LOAD")
pickup_data = data[pickup_mask].copy()

# Mask for items associated with valid pentatonic IDs
id_matched_mask = data["pentatonic_id"].notna() & (data["pentatonic_id"] != "")

# Weight columns naming convention is "weigh_ustons_<suffix>"
weight_columns = [col for col in data.columns if "ustons" in col]

print("Available weight columns: {}".format(weight_columns))

# Estimated weight of claim items
for col in weight_columns:
    data[col] = data[col] * data["count"]
# data["item_weight_lbs"] = data["weight_lbs"] * data["count"]
# data["item_volume_cf"] = data["volume_cf"] * data["count"]
# data["item_volume_cy"] = data["volume_cy"] * data["count"]

# Dumpster load items have patterns, e.g. "DUMPSTER LOAD APPROX 12 YARDS 1-3 TONS DEBRIS" or DUMPSTER LOAD APPROX 20 YARDS 4 TONS DEBRIS
# In order to calculate weight for each truck, numbers before TONS are extracted

# Mask to find weight patterns as ranged numbers, such as 1-3 TONS, 5-7 TONS, etc. 
range_mask = truck_data["item_description"].str.contains('\s+\d\s*-\s*\d\s+TONS', flags=re.IGNORECASE, regex=True)

# Mask to find weight patterns as single numbers, such as 4 TONS, etc. 
precise_mask = truck_data["item_description"].str.contains('(\s+\d\s+TONS)(?!\s+\d\s*-\s*\d\s+TONS)', flags=re.IGNORECASE, regex=True)

# Upper and lower truck weights are introduced since the majority of dumpster loads contain range of weights
# In case of single numbers, lower and upper weights are equal 
truck_data[["lower_truck_weight", "upper_truck_weight"]] = 0
truck_data.loc[range_mask, ["lower_truck_weight", "upper_truck_weight"]] = truck_data.loc[range_mask, "item_description"].str.extract(r'\s+(\d)\s*-\s*(\d)\s+TONS', flags=re.IGNORECASE).values
truck_data.loc[precise_mask, "lower_truck_weight"] = truck_data.loc[precise_mask, "upper_truck_weight"] = truck_data.loc[precise_mask, "item_description"].str.extract(r'\s+(\d)\s+TONS', flags=re.IGNORECASE).values

# Calculate average weight for each truck and multiply by number of trucks
truck_data["average_truck_weight"] = 0.5*(truck_data["lower_truck_weight"].astype(float) + truck_data["upper_truck_weight"].astype(float))*truck_data["count"]

# Assumptions based on data we got for our app
pickup_data['weight'] = pickup_data['item_quantity'] * 2.4 

# Total weight of all dumpster loads per claim ID
truck_weight_data = truck_data.groupby("claim_id").agg(total_truck_weight=("average_truck_weight", "sum")).reset_index()

# Total weight of all pickup trucks per claim ID
pickup_weight_data = pickup_data.groupby("claim_id").agg(total_pickup_weight=("weight", "sum")).reset_index()

# Merge dumpster loads and pickup trucks data
all_truck_weight_data = pd.merge(truck_weight_data, pickup_weight_data, how='outer')

# Fill NaNs in total weight with 0
all_truck_weight_data['total_pickup_weight'] = all_truck_weight_data['total_pickup_weight'].fillna(0)
all_truck_weight_data['total_truck_weight'] = all_truck_weight_data['total_truck_weight'].fillna(0)

# Total weight of dumpster loads and pickup trucks
all_truck_weight_data['total_weight'] = all_truck_weight_data['total_truck_weight'] + all_truck_weight_data['total_pickup_weight']

# Merge truck data with main claim data
data = data.merge(all_truck_weight_data[["claim_id", "total_weight"]], on="claim_id", how="left")

# Function to calculate fraction of items matched with weights DB
def matching_fraction(col):
    
    mask = col.notna() & (col != 0) & (col != "")
    
    return round(mask.sum()/col.shape[0], 2)

# Introduce data aggregations per claim ID
aggregate_args = {
    "total_claims" : ("item_description", "count"),
    "total_items" : ("count", "sum"),
    "total_truck_weight" : ("total_weight", "last"),
    "ID_matched_fraction" : ("pentatonic_id", matching_fraction),
    "matched_fraction" : ("weights_primary_desc", matching_fraction)
}

for col in weight_columns:
    aggregate_args["weight_estimation_ustons" + col.partition("ustons")[-1]] = (col, "sum")

# Aggregate claim data for each claim ID
# claim_id_data = data[~truck_mask].groupby(["claim_id"]).agg(
#     total_claims=("item_description", "count"),
#     total_items=("count", "sum"),
#     total_truck_weight=("total_truck_weight", "last"),
#     # weight_estimation_lbs=("item_weight_lbs", "sum"),
#     weight_estimation_ustons=("item_weight_ustons", "sum"),
#     # volume_estimation_cf=("item_volume_cf", "sum"),
#     # volume_estimation_cy=("item_volume_cy", "sum"),
#     ID_matched_fraction=("pentatonic_id", matching_fraction),
#     matched_fraction=("item_weight_lbs", matching_fraction)
# )
truck_data_mask = truck_mask | pickup_mask

claim_id_data = data[~truck_data_mask].groupby(["claim_id"]).agg(**aggregate_args)

# # Calculate difference between total truck weight and total estimated weight of claims in US tonnes
# claim_id_data["excessive_truck_weight"] = claim_id_data["total_truck_weight"] - claim_id_data["weight_estimation_ustons"]

# Saving estimation data to Excel
db_version = config["data"]["db_version"]
output_path = "{0}/{1}".format(config["data"]["output_dir"], db_version)

claim_id_data.to_excel("{}/claim_id_weight_comparison_{}_test.xlsx".format(output_path, db_version))

hist_colors = list(mcolors.TABLEAU_COLORS)
line_colors = list(mcolors.BASE_COLORS)

weight_columns = [col for col in claim_id_data.columns if "ustons" in col]
#plot_columns = [col for col in weight_columns if "zero" in col]
plot_columns = weight_columns
labels = []

# Absolute excessive truck weight plots
for col in plot_columns:

    index = plot_columns.index(col)

    col_suffix = col.partition("ustons")[-1]
    plt_label = ",".join(list(filter(None, col_suffix.split("_"))))
    new_col = "excessive_truck_weight" + col_suffix


    claim_id_data[new_col] = claim_id_data["total_truck_weight"] - claim_id_data[col]

    y = claim_id_data[new_col].dropna()

    bins = np.linspace(-400, 400, 401)
    # hist = y.hist(bins=bins, density=True)
    points, _, _ = plt.hist(y, bins=bins, density=True, histtype="step", linewidth=2, color=hist_colors[index])

    bin_width = bins[1] - bins[0]
    zero_bin = np.where(bins==0)[0][0]
    positive_integral = bin_width * sum(points[zero_bin:-1])

    labels.append("({}) data: {:.1f}% excessive weight>0".format(plt_label, positive_integral*100))

    # points, _, _ = plt.hist(y, bins=bins, density=True, histtype="barstepstacked", color=hist_colors[index], label="({}) data".format(plt_label))

    # mu, std = norm.fit(y)

    # xbins = np.linspace(-400, 400, 40001)
    # x = [0.5 * (xbins[i] + xbins[i+1]) for i in range(len(xbins)-1)]

    # fit = norm.pdf(x, mu, std)
    # plt.plot(x, fit, 'm-', linewidth=3, label=r'({}): gaussian fit $(\mu={:.2f},\sigma={:.2f})$'.format(plt_label, mu, std))

    # kde = gaussian_kde(y).pdf(x)
    # plt.plot(x, kde, '{}-'.format(line_colors[index]), linewidth=1.5, label="{} Gaussian KDE".format(plt_label))

plt.grid(True)

plt.xlabel("Excessive truck weight in tons", fontsize=30)
plt.ylabel("Frequency", fontsize=30)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(labels=labels, fontsize=16)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)

plt.xlim(-50,50)
plt.savefig('{}/excessive_truck_weight_frequency_{}.png'.format(output_path, db_version))

plt.clf()

# Relative excessive truck weight plots
for col in plot_columns:

    index = plot_columns.index(col)

    col_suffix = col.partition("ustons")[-1]
    plt_label = ",".join(list(filter(None, col_suffix.split("_"))))
    new_col = "excessive_truck_weight" + col_suffix

    claim_id_data[new_col] = claim_id_data["total_truck_weight"] - claim_id_data[col]

    valid_data = claim_id_data.dropna(subset=[new_col, "total_truck_weight"])

    y = valid_data[new_col]/valid_data["total_truck_weight"]*100

    bins = np.linspace(-100, 100, 101)
    points, _, _ = plt.hist(y, bins=bins, density=True, histtype="step", linewidth=2, color=hist_colors[index])

    bin_width = bins[1] - bins[0]
    zero_bin = np.where(bins==0)[0][0]
    positive_integral = bin_width * sum(points[zero_bin:-1])

    labels.append("({}) data: {:.1f}% excessive weight>0".format(plt_label, positive_integral*100))

plt.grid(True)

plt.xlabel("Relative excessive truck weight [%]", fontsize=30)
plt.ylabel("Frequency", fontsize=30)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(labels=labels, fontsize=16, loc='upper center')

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)

plt.savefig('{}/relative_excessive_truck_weight_frequency_{}.png'.format(output_path, db_version))

# bins = np.linspace(-400, 400, 401)
# # hist = y.hist(bins=bins, density=True)
# points, _, _ = plt.hist(y, bins=bins, density=True)

# plt.xlim(-400,400)
# plt.yscale("log")

# plt.savefig('{}/excessive_truck_weight_frequency_logscale_{}.png'.format(output_path, db_version))

# bin_width = bins[1] - bins[0]
# zero_bin = np.where(bins==0)[0][0]
# four_bin = np.where(bins==4)[0][0]
# eight_bin = np.where(bins==8)[0][0]

# print(bins[four_bin])
# print(bins[eight_bin])

# positive_integral = bin_width * sum(points[zero_bin:-1])
# print("Positive excessive weight integral: {}".format(positive_integral))

# negative_integral = bin_width * sum(points[0:zero_bin])
# print("Negative excessive weight integral: {}".format(negative_integral))

# zero_to_four_integral = bin_width * sum(points[zero_bin:four_bin])
# print("Excessive weight 0-4 integral: {}".format(zero_to_four_integral))

# zero_to_eight_integral = bin_width * sum(points[zero_bin:eight_bin])
# print("Excessive weight 0-8 integral: {}".format(zero_to_eight_integral))





