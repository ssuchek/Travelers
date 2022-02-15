import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

from config import config
import config as constants

import importlib
importlib.reload(constants)

import numpy as np
import pandas as pd
from pandas import ExcelWriter

import re
import time

db_version = config["data"]["db_version"]
# Path to input matched data
filepath = "{0}/{1}/claims_weight_db_matched_weights_{1}.csv".format(config["data"]["output_dir"], db_version)

data = pd.read_csv(filepath)

# Data for dumpster loads
truck_id = "DUMPSTER LOAD"

# Keep claim IDs with only dumpster load collection method

no_dumpster_load_mask = data["subcategory_prev"].str.contains("GENERAL DEMOLITION", flags=re.IGNORECASE, regex=True) & \
                        (data["item_description"].str.contains(truck_id, flags=re.IGNORECASE, regex=True) == False)
no_dumpster_load_ids = data.loc[no_dumpster_load_mask, "claim_id"].unique().tolist()

total_ids_initial = len(data["claim_id"].unique().tolist())

data = data[~data["claim_id"].isin(no_dumpster_load_ids)]

print("Total {}/{} claim IDs contain only dumpster load collection method".format(len(no_dumpster_load_ids),
                                                                                 total_ids_initial)
     )

truck_mask = data["subcategory_prev"].str.contains("GENERAL DEMOLITION", flags=re.IGNORECASE, regex=True) & \
            data["item_description"].str.contains(truck_id, flags=re.IGNORECASE, regex=True)
truck_data = data[truck_mask].copy()

# Data for pickup trucks loads
pickup_id = "PICKUP TRUCK LOAD"
pickup_mask = data["subcategory_prev"].str.contains("GENERAL DEMOLITION", flags=re.IGNORECASE, regex=True) & \
            data["item_description"].str.contains(pickup_id, flags=re.IGNORECASE, regex=True)
pickup_data = data[pickup_mask].copy()

axle_id = "AXLE DUMP"
axle_mask = data["subcategory_prev"].str.contains("GENERAL DEMOLITION", flags=re.IGNORECASE, regex=True) & \
            data["item_description"].str.contains(axle_id, flags=re.IGNORECASE, regex=True)

# Mask for items associated with valid pentatonic IDs
id_matched_mask = data["pentatonic_id"].notna() & (data["pentatonic_id"] != "")

# Weight columns naming convention is "weigh_ustons_<suffix>"
weight_columns = [col for col in data.columns if "ustons" in col]

print("Available weight columns: {}".format(weight_columns))

# Estimated weight of claim items
for col in weight_columns:
    data[col] = data[col] * data["item_quantity"]
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
# truck_data["truck_weight"] = 0.5*(truck_data["lower_truck_weight"].astype(float) + truck_data["upper_truck_weight"].astype(float))*truck_data["item_quantity"]
truck_data["truck_weight"] = truck_data["upper_truck_weight"].astype(float)*truck_data["item_quantity"]
truck_data["truck_cost"] = 0

for size in constants.TRUCK_COST_MAP[truck_id]:
    if not "YARDS" in size:
        truck_size_mask = truck_data["item_description"].astype(str).str.contains("YARDS", flags=re.IGNORECASE, regex=True)
        truck_data.loc[~truck_size_mask, "truck_cost"] = constants.TRUCK_COST_MAP[truck_id]["TRUCK"] * truck_data["item_quantity"]
    else:
        truck_size_mask = truck_data["item_description"].astype(str).str.contains(size, flags=re.IGNORECASE, regex=True)
        truck_data.loc[truck_size_mask, "truck_cost"] = constants.TRUCK_COST_MAP[truck_id][size] * truck_data["item_quantity"]

# Assumptions based on data we got for our app
print("Using pickup truck weight: {} tons".format(constants.PICKUP_TRUCK_WEIGHT))
print("Using pickup truck cost: {}$".format(constants.TRUCK_COST_MAP[pickup_id]))
pickup_data["pickup_weight"] = pickup_data["item_quantity"] * constants.PICKUP_TRUCK_WEIGHT
pickup_data["pickup_cost"]   = pickup_data["item_quantity"] * constants.TRUCK_COST_MAP[pickup_id] 

def to_list(col, to_int=False):
    
    values = col.values.tolist()
    
    if to_int:
        values = [int(v) for v in values]
    
    return ",".join(str(v) for v in values)

# Total weight of all dumpster loads per claim ID
truck_weight_data = truck_data.groupby("claim_id").agg(total_truck_weight=("truck_weight", "sum"),
                                                       all_truck_weights=("upper_truck_weight", to_list),
                                                       all_truck_qty=("item_quantity", to_list),
                                                    total_truck_cost=("truck_cost", "sum"),
                                                    ).reset_index()

# Total weight of all pickup trucks per claim ID
pickup_weight_data = pickup_data.groupby("claim_id").agg(total_pickup_weight=("pickup_weight", "sum"),
                                                        total_pickup_cost=("pickup_cost", "sum"),
                                                        ).reset_index()
# Merge dumpster loads and pickup trucks data
all_truck_weight_data = pd.merge(truck_weight_data, pickup_weight_data, how='outer')

# Fill NaNs in total weight with 0
all_truck_weight_data['total_truck_weight'] = all_truck_weight_data['total_truck_weight'].fillna(0)
all_truck_weight_data['total_truck_cost'] = all_truck_weight_data['total_truck_cost'].fillna(0)
all_truck_weight_data['total_pickup_weight'] = all_truck_weight_data['total_pickup_weight'].fillna(0)
all_truck_weight_data['total_pickup_cost'] = all_truck_weight_data['total_pickup_cost'].fillna(0)

# Total weight of dumpster loads and pickup trucks
all_truck_weight_data['total_weight'] = all_truck_weight_data['total_truck_weight'] + all_truck_weight_data['total_pickup_weight']
all_truck_weight_data['total_cost'] = all_truck_weight_data['total_truck_cost'] + all_truck_weight_data['total_pickup_cost']

truck_data_mask = (truck_mask | pickup_mask | axle_mask)
data = data[~truck_data_mask]

# Merge truck data with main claim data
data = data.merge(all_truck_weight_data[["claim_id", "total_weight", "all_truck_weights", "all_truck_qty", "total_cost", ]], on="claim_id", how="left")

# Function to calculate fraction of items matched with weights DB
def total_matching_fraction(col, ext_mask=None):
    
    mask = col.notna() & (col != 0) & (col != "")
    
    return round(mask.sum()/col.shape[0], 2)

def single_matching_fraction(col, ext_mask=None):
    
    mask = col.notna() & (col != 0) & (col != "") & (~col.astype(str).str.contains(","))
    
    return round(mask.sum()/col.shape[0], 2)

def multiple_matching_fraction(col, ext_mask=None):
    
    mask = col.notna() & (col != 0) & (col != "") & (col.astype(str).str.contains(","))
    
    return round(mask.sum()/col.shape[0], 2)

# Introduce data aggregations per claim ID
aggregate_args = {
    "date" : ("ls_date", "last"),
    "zip" : ("zip", "last"),
    "total_claims" : ("item_description", "count"),
    "total_items" : ("count", "sum"),
    "total_truck_weight" : ("total_weight", "last"),
    "truck_weights" : ("all_truck_weights", "last"),
    "truck_qty" : ("all_truck_qty", "last"),
    "total_truck_cost" : ("total_cost", "last"),
    "total_id_matching_fraction" : ("pentatonic_id", total_matching_fraction),
    "single_id_matching_fraction" : ("pentatonic_id", single_matching_fraction),
    "multiple_id_matching_fraction" : ("pentatonic_id", multiple_matching_fraction),
    "primary_matching_fraction" : ("weights_primary_desc", total_matching_fraction)
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

claim_id_data = data.groupby(["claim_id"]).agg(**aggregate_args).reset_index()

weight_columns = [col for col in claim_id_data.columns if "weight_estimation_ustons" in col]

for col in weight_columns:
    col_suffix = col.partition("ustons")[-1]
    
    claim_id_data[col] = round(claim_id_data[col], 2)

    claim_id_data["excessive_truck_weight_ustons" + col_suffix] = claim_id_data["total_truck_weight"] - claim_id_data[col]
    claim_id_data["relative_excessive_truck_weight_ustons" + col_suffix] = round(claim_id_data["excessive_truck_weight_ustons" + col_suffix]/claim_id_data["total_truck_weight"]*100,2)

    
# Saving estimation data to Excel
output_path = "{0}/{1}".format(config["data"]["output_dir"], db_version)


suffixes = [col.partition("ustons_")[-1] for col in claim_id_data.columns if col.startswith("weight_estimation_ustons")]

columns = ['claim_id', 'zip', 'total_claims', 'total_items', 'total_truck_weight', 'total_truck_cost', 
           'total_id_matching_fraction', 'single_id_matching_fraction', 'multiple_id_matching_fraction', 
           'primary_matching_fraction']
# claim_id_data.to_excel("{}/claim_id_weight_comparison_{}_only_dumpster_loads.xlsx".format(output_path, db_version))


prices_data = pd.read_csv("data/collection_methods.csv")

dumpster_load_mask = (prices_data["type"] == "Dumpster")
prices_data = prices_data.loc[dumpster_load_mask, ["type", "price_per_collection", "weight_limit_tons"]]

agg_prices_data = prices_data.groupby("type").agg(price_per_collection=("price_per_collection",list),
                                        weight_limit_ustons=("weight_limit_tons",list)
                                        )

weights = agg_prices_data["weight_limit_ustons"].iloc[0]
costs = agg_prices_data["price_per_collection"].iloc[0]

# density, costs, weights = zip(*sorted((c/w, c, w) for c, w in zip(costs, weights)))

# print(density)
print(weights)
print(costs)

def minimize_estimation_cost(row, weights, costs, method):
    
    weights = [float(w) for w in weights]
    costs   = [float(c) for c in costs]
    
    max_weight = float(row["weight_estimation_ustons" + method])

    if max_weight == 0:
        return 0

    def fit_weights(min_cost, min_combination, coeffs, in_costs, in_weights, reference_weight):
        """
            Function that calculates the best packing of a given reference weight into trucks of given sizes
            with the minimum cost.

            Best cost value is calculated based on the price per ton (PPT) for each truck (cost/weight). 
            1. First, trucks are ranked from the lowest PPT to the highest. 
            2. The maximum amount n0 of lowest PPT trucks which fill the largest possible weight 
                below reference weight is obtained as quotient of reference_weight/truck_weight.
                In case all trucks are heavier than reference weight, the truck with the minimum cost is chosen.
            3. First candidate for minimal cost combination is (n0+1)*C_0 of lowest PPT trucks.
            4. In case of next lowest PPT trucks with weights W_i above the remainder weight 
                (reference_weight mod truck_weight) and costs C_i, minimal cost combinations n0*C_0+C_i are
                tested against previous minimum cost (n0+1)*C_0. If n0*C_0+C_i < (n0+1)*C_0, it becomes new minimum cost.
                These trucks are then excluded from the future iterations. 
            5. For the trucks that weigh less than reference weight, repeat steps 1-4 until no trucks are left.

            Input parameters:
            :param coeffs:                 keep coefficients for weights (=number of trucks)
            :param in_costs:               input costs of each available truck
            :param in_weights:             input weights of each available truck
            :param reference_weight:       weight to pack into input trucks
            Output parameters:
            :param min_cost:               minimum cost                           
            :param min_combination:        combination of truck weights with the minimum cost
        """
    
        if not in_weights or not in_costs:
            print("No input weights detected. Switching to next iteration...")
            return 
        
        previous_cost = sum([c*costs[weights.index(w)] for w,c in coeffs.items()])

        if min(in_weights) > reference_weight:
            for c,w in zip(in_costs, in_weights):
                if previous_cost + c < min_cost:
                    min_cost = previous_cost + c
                    min_combination = {**coeffs, **{w:1}}
            return min_cost, min_combination
        elif max(in_weights) > reference_weight:
            above_costs, above_weights = zip(*((c, w) for c, w in zip(in_costs, in_weights)
                                                  if w > reference_weight))
            for c,w in zip(above_costs, above_weights):
                if previous_cost + c < min_cost:
                    min_cost = previous_cost + c
                    min_combination = {**coeffs, **{w:1}}
                in_costs.remove(c)
                in_weights.remove(w)
        
        
        _, valid_costs, valid_weights = zip(*sorted((c/w, c, w) for c, w in zip(in_costs, in_weights)))

        for c,w in zip(valid_costs, valid_weights):
            new_coeff = reference_weight // w
            
            if previous_cost + (new_coeff + 1) * c < min_cost:
                min_combination = {**coeffs, **{w:new_coeff+1}}
                min_cost = previous_cost + (new_coeff + 1) * c
            
            remaining_costs = [cc for cc in valid_costs if cc != c]
            remaining_weights = [ww for ww in valid_weights if ww != w]
            
            if len(remaining_weights) == 0:
                continue

            min_cost, min_combination = fit_weights(min_cost, min_combination, 
                        {**coeffs, **{w:new_coeff}}, 
                        remaining_costs, 
                        remaining_weights, 
                        reference_weight % w)

            previous_cost = sum([c*costs[weights.index(w)] for w,c in coeffs.items()])
            
        return min_cost, min_combination
            
    min_combination = {}
    min_cost = 100000000000
    coeffs = {}
    
    min_cost, min_combination = fit_weights(min_cost, min_combination, coeffs, costs, weights, max_weight)        
        
    min_weights = ",".join(str(w) for w,i in min_combination.items() if i > 0)
    min_nitems  = ",".join(str(int(i)) for w,i in min_combination.items() if i > 0)
    
    return pd.Series([round(min_cost,2), min_nitems, min_weights])


for col in weight_columns:
    col_suffix = col.partition("ustons")[-1]
    
    new_col = "estimated_cost" + col_suffix
    
    # Without density sorting
    claim_id_data[[new_col, "estimated_truck_qty" + col_suffix, "estimated_truck_weights" + col_suffix]] = claim_id_data.apply(lambda row: minimize_estimation_cost(row, weights, costs, col_suffix), axis=1)
    claim_id_data["excessive_cost" + col_suffix] = claim_id_data["total_truck_cost"] - claim_id_data[new_col]
    claim_id_data["relative_excessive_cost" + col_suffix] = round(claim_id_data["excessive_cost" + col_suffix]/claim_id_data["total_truck_cost"]*100,2)
    
#     # With density sorting
#     density, costs_, weights_ = zip(*sorted((c/w, c, w) for c, w in zip(costs, weights)))
#     claim_id_data[new_col+"_sorted"] = claim_id_data.apply(lambda row: minimize_estimation_cost(row, weights_, costs_, col_suffix), axis=1)
#     claim_id_data["excessive_cost_sorted" + col_suffix] = claim_id_data["total_truck_cost"] - claim_id_data[new_col+"_sorted"]
#     claim_id_data["relative_excessive_cost_sorted" + col_suffix] = round(claim_id_data["excessive_cost" + col_suffix]/claim_id_data["total_truck_cost"]*100,2)

suffixes = [col.partition("ustons_")[-1] for col in claim_id_data.columns if col.startswith("weight_estimation_ustons")]

columns = ['claim_id', 'zip', 'total_claims', 'total_items', 'total_truck_weight', 'truck_weights', 'truck_qty',
           'total_truck_cost', 
           'total_id_matching_fraction', 'single_id_matching_fraction', 'multiple_id_matching_fraction', 
           'primary_matching_fraction']

excel_name = "{}/claim_id_weight_comparison_{}_only_dumpster_loads.xlsx".format(output_path, db_version)

plot_mask = (claim_id_data["total_truck_cost"] > 0) & (claim_id_data["estimated_cost_median_median"] > 0) 
    
total_excessive_costs = claim_id_data.loc[plot_mask, "excessive_cost_median_median"].dropna().sum()
# total_excessive_costs_sorted = claim_id_data.loc[plot_mask, "excessive_cost_sorted_median_median"].dropna().sum()

print("Total excessive costs: {:.1f} $".format(total_excessive_costs))
# print("Total sorted excessive costs: {:.1f} $".format(total_excessive_costs_sorted))

start_time = time.time()

with ExcelWriter(excel_name, engine='xlsxwriter') as writer:
    for suffix in suffixes:
        save_cols = columns + [col for col in claim_id_data.columns if suffix in col]
        save_df = claim_id_data[save_cols]
        
        save_df.to_excel(writer, sheet_name=suffix, index=False)
        
        worksheet = writer.sheets[suffix]
        for idx, col in enumerate(save_df):  # loop through all columns
            series = save_df[col]
            max_len = max((
                series.astype(str).map(len).max(),  # len of largest item
                len(str(series.name))  # len of column name/header
                )) + 1  # adding a little extra space
            worksheet.set_column(idx, idx, max_len)  # set column width
    writer.save()
            
print("Total time elapsed: {} s".format(time.time()-start_time))

id_matching_fractions = [0, 0.25, 0.5, 0.75, 1]
id_col = "single_id_matching_fraction"


valid_weight_mask = (claim_id_data["total_truck_weight"] > 0) & (claim_id_data["weight_estimation_ustons_median_median"] > 0) & \
                            (claim_id_data[id_col] > 0.75) & \
                            (claim_id_data[id_col] <= 1.)

final_data = claim_id_data[valid_weight_mask]

columns = ['claim_id', 'date', 'total_id_matching_fraction', 'total_items', 
           'total_truck_weight', 'truck_weights', 'truck_qty', 'total_truck_cost']

excel_name = "{}/final_only_positive_claim_id_weight_comparison_{}_only_dumpster_loads.xlsx".format(output_path, db_version)

plot_mask = (claim_id_data["total_truck_cost"] > 0) & (claim_id_data["estimated_cost_median_median"] > 0) 
    
total_excessive_costs = claim_id_data.loc[plot_mask, "excessive_cost_median_median"].dropna().sum()
# total_excessive_costs_sorted = claim_id_data.loc[plot_mask, "excessive_cost_sorted_median_median"].dropna().sum()

print("Total excessive costs: {:.1f} $".format(total_excessive_costs))
# print("Total sorted excessive costs: {:.1f} $".format(total_excessive_costs_sorted))

start_time = time.time()

suffixes = ["median_median"]

with ExcelWriter(excel_name, engine='xlsxwriter') as writer:
    for suffix in suffixes:
        save_cols = columns+[c for c in final_data.columns if "median_median" in c]
        add_mask = (final_data["excessive_truck_weight_ustons_median_median"] > 0)
        save_df = final_data.loc[add_mask, save_cols]
        
        final_data['total_id_matching_fraction'] = final_data['total_id_matching_fraction'] * 100
        
        save_df.to_excel(writer, sheet_name=suffix, index=False)
        
        worksheet = writer.sheets[suffix]
        for idx, col in enumerate(save_df):  # loop through all columns
            series = save_df[col]
            max_len = max((
                series.astype(str).map(len).max(),  # len of largest item
                len(str(series.name))  # len of column name/header
                )) + 1  # adding a little extra space
            worksheet.set_column(idx, idx, max_len)  # set column width
    writer.save()            
print("Total time elapsed: {} s".format(time.time()-start_time))



import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

from scipy import integrate

valid_truck_weight_mask = (claim_id_data["total_truck_weight"] > 0)

hist_colors = list(mcolors.TABLEAU_COLORS)
line_colors = list(mcolors.BASE_COLORS)

total_excessive_weights = {}

plot_columns = [col for col in claim_id_data.columns if col.startswith("excessive_truck_weight")]
labels = []

down_thres = None

# Absolute excessive truck weight plots
for col in plot_columns:
    
    index = plot_columns.index(col)

    col_suffix = col.partition("ustons")[-1]
    plt_label = ",".join(list(filter(None, col_suffix.split("_"))))
    
    
    valid_weight_mask = valid_truck_weight_mask & (claim_id_data["weight_estimation_ustons" + col_suffix] > 0)
    
    if down_thres is not None:
        thres_mask = (claim_id_data[col] > down_thres)
        print("{}, total entries below {} t: {}".format(plt_label, down_thres, (valid_weight_mask & ~thres_mask).sum()))
        valid_weight_mask &= thres_mask

    y = claim_id_data.loc[valid_weight_mask, col].dropna()

    bins = np.linspace(-400, 400, 401)
    points, _, _ = plt.hist(y, bins=bins, density=True, histtype="step", linewidth=2, color=hist_colors[index])

    positive_integral = y[y > 0].shape[0]/y.shape[0]
    total_excessive_weight = y.sum()
    total_excessive_weights[col_suffix] = total_excessive_weight

    labels.append("({}): {:.1f}\% positive ({:.1f} tons)".format(plt_label, positive_integral*100, total_excessive_weight))

plt.grid(True)

plt.title("Excessive weight estimation for claim IDs with only dumpster loads", fontsize=30)


plt.xlabel("Excessive truck weight in tons", fontsize=30)
plt.ylabel("Frequency", fontsize=30)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(labels=labels, fontsize=20)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)

plt.xlim(-50,50)

file_label = "_{}t".format(abs(down_thres)) if down_thres is not None else ""
plt.savefig('{}/excessive_truck_weight_frequency_only_dumpster_loads{}_{}.png'.format(output_path, file_label, db_version), facecolor='w')




import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

from scipy import integrate

valid_truck_weight_mask = (claim_id_data["total_truck_weight"] > 0)

hist_colors = list(mcolors.TABLEAU_COLORS)
line_colors = list(mcolors.BASE_COLORS)

# weight_columns = [col for col in claim_id_data.columns if "ustons" in col and col.startswith("excessive")]
#plot_columns = [col for col in weight_columns if "zero" in col]
plot_columns = [col for col in claim_id_data.columns if col.startswith("excessive_truck_weight") and "median_median" in col]
labels = []

down_thres = None

# Absolute excessive truck weight plots
for col in plot_columns:
    
    for i in range(len(id_matching_fractions)-1):
    
        index = i

        col_suffix = col.partition("ustons")[-1]
        plt_label = ",".join(list(filter(None, col_suffix.split("_"))))

        valid_weight_mask = valid_truck_weight_mask & (claim_id_data["weight_estimation_ustons" + col_suffix] > 0) & \
                            (claim_id_data[id_col] > id_matching_fractions[i]) & \
                            (claim_id_data[id_col] <= id_matching_fractions[i+1])

        if down_thres is not None:
            thres_mask = (claim_id_data[col] > down_thres)
            print("{}, total entries below {} t: {}".format(plt_label, down_thres, (valid_weight_mask & ~thres_mask).sum()))
            valid_weight_mask &= thres_mask

        y = claim_id_data.loc[valid_weight_mask, col].dropna()

        bins = np.linspace(-400, 400, 401)
        points, _, _ = plt.hist(y, bins=bins, density=True, histtype="step", linewidth=2, color=hist_colors[index])

        positive_integral = y[y > 0].shape[0]/y.shape[0]
        total_excessive_weight = y.sum()

        labels.append("{}-{}\% ({}): {:.1f}\% positive ({:.1f} tons)".format(int(id_matching_fractions[i]*100),
                                                                             int(id_matching_fractions[i+1]*100),
                                                                             y.shape[0],
                                                                             positive_integral*100, 
                                                                             total_excessive_weight))

plt.grid(True)

plt.title("Excessive weight estimation for claim IDs with only dumpster loads for different single ID matching", fontsize=24)


plt.xlabel("Excessive truck weight in tons", fontsize=30)
plt.ylabel("Frequency", fontsize=30)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(labels=labels, fontsize=20)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)

plt.xlim(-50,50)

file_label = "_{}t".format(abs(down_thres)) if down_thres is not None else ""
plt.savefig('{}/excessive_truck_weight_frequency_only_dumpster_loads_IDmatching{}_{}.png'.format(output_path, file_label, db_version), facecolor='w')




plt.clf()

plot_columns = [col for col in claim_id_data.columns if col.startswith("relative_excessive_truck_weight")]
labels = []

# Relative excessive truck weight plots
for col in plot_columns:
    
    valid_weight_mask = valid_truck_weight_mask & (claim_id_data[col] > 0)

    index = plot_columns.index(col)

    col_suffix = col.partition("ustons")[-1]
    plt_label = ",".join(list(filter(None, col_suffix.split("_"))))
    
    valid_weight_mask = valid_truck_weight_mask & (claim_id_data["weight_estimation_ustons" + col_suffix] > 0)

    y = claim_id_data.loc[valid_weight_mask, col].dropna()

    bins = np.linspace(-1000, 100, 276)
    points, _, _ = plt.hist(y, bins=bins, density=True, histtype="step", linewidth=2, color=hist_colors[index])

    positive_integral = y[y > 0].shape[0]/y.shape[0]

    labels.append("({}): {:.1f}\% positive ({:.1f} tons)".format(plt_label, positive_integral*100, total_excessive_weights[col_suffix]))

plt.grid(True)

plt.title("Relative excessive weight estimation for claim IDs with only dumpster loads", fontsize=30)

plt.xlabel("Relative excessive truck weight [\%]", fontsize=30)
plt.ylabel("Frequency", fontsize=30)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(labels=labels, fontsize=20, loc='upper left')

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)

plt.xlim(-100,100)

plt.savefig('{}/relative_excessive_truck_cost_frequency_only_dumpster_loads{}_{}.png'.format(output_path, file_label, db_version), facecolor='w')








