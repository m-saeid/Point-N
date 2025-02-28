import os
import sys
import pandas as pd

# Path setup
project_path = os.path.abspath(".")
sys.path.append(project_path)


def group_and_average(df, columns, target="acc_cos"):
    # Check if all columns are in the DataFrame
    if not all(col in df.columns for col in columns + [target]):
        raise ValueError(
            f"One or more columns {columns + [target]} are not present in the DataFrame"
        )

    # Group by the specified columns and calculate the mean of the target
    grouped = df.groupby(columns)[target].mean().reset_index()

    return grouped


def group_and_max(df, columns, target="acc_cos"):
    # Check if all columns are in the DataFrame
    if not all(col in df.columns for col in columns + [target]):
        raise ValueError(
            f"One or more columns {columns + [target]} are not present in the DataFrame"
        )

    # Group by the specified columns and calculate the max of the target
    grouped = df.groupby(columns)[target].max().reset_index()

    return grouped


dataset = "scanobjectnn"  # "modelnet40", "scanobjectnn"

if dataset == "scanobjectnn":
    file_names = {
        "PB_T50_RS": "pointgn_cls_scanobject_nFalse_sPB_T50_RS",
        "OBJ_ONLY": "pointgn_cls_scanobject_nFalse_sOBJ_ONLY",
        "OBJ_BG": "pointgn_cls_scanobject_nFalse_sOBJ_BG",
    }
elif dataset == "modelnet40":
    file_names = {"MODELNET40": "pointgn_cls_modelnet40_nTrue"}

# Initialize an empty DataFrame
all_data = pd.DataFrame()

# Load all CSV files and concatenate them
for key, file_name in file_names.items():
    file_path = os.path.join("eval", f"{file_name}.csv")
    df = pd.read_csv(file_path)
    df["dataset"] = key  # Optionally add a column to identify the dataset
    all_data = pd.concat([all_data, df], ignore_index=True)

# Group and calculate the average and max for various columns
grouped = group_and_average(all_data, ["sigma", "fdim", "stage", "k"])

# Compute and print group-wise averages and max values for each column
def print_grouped_data(grouped, column_name):
    mean_values = group_and_average(grouped, [column_name])
    max_values = group_and_max(grouped, [column_name])
    
    print(f"\n{'-'*40}")
    print(f"Group-wise Average of '{column_name}':")
    print(mean_values.to_string(index=False))  # Avoid index for cleaner output
    print(f"\n{'-'*40}")
    print(f"Group-wise Max of '{column_name}':")
    print(max_values.to_string(index=False))  # Avoid index for cleaner output
    print(f"{'-'*40}\n")


print_grouped_data(grouped, "fdim")
print_grouped_data(grouped, "stage")
print_grouped_data(grouped, "k")
print_grouped_data(grouped, "sigma")

sorted_values = grouped.sort_values(by="acc_cos", ascending=False).head(10)
print(sorted_values.to_string(index=False)) 


