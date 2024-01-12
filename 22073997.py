#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 19:34:44 2024

@author: apple
"""
#import of libraries

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import errors as er  



#using function read_csv_and_set_index to read and set index for my dataframe
def read_csv_and_set_index(file_name):
    """
    Read a CSV file into a DataFrame and set the 'Country Name' column as 
    the index.

    Parameters:
    - file_name (str): The name of the CSV file.

    Returns:
    - pd.DataFrame: The DataFrame with the 'Country Name' column set as the 
    index.
    """

    df1 = pd.read_csv(file_name, index_col='Country Name')
    tranposed_df = df1.T

    return df1, tranposed_df


file_name = 'ads2.csv'
df1, tranposed_df = read_csv_and_set_index(file_name)

# display my data
print(df1)
print(tranposed_df)

#removing the first column for from the data
df1 = df1.iloc[1:]
df1 = df1.reset_index(drop=True)  # index is not copied into index column

print(df1)

#the statistical infomation about my data
print(df1.describe())

#creating another varaible called corr to check my data correlation
corr = df1.corr()
print(corr.round(3))

# fig, ax = plt.subplots(figsize=(8, 8))
plt.figure(figsize=[8, 8])

# this prouces an image
plt.imshow(corr)
plt.colorbar()
annotations = df1.columns[:6]  # extract relevant headers

# Set tick positions and labels
plt.xticks(ticks=range(len(annotations)), labels=annotations, rotation=45,
           ha='right', fontsize=12)
plt.yticks(ticks=range(len(annotations)), labels=annotations, fontsize=12)

# Adjust layout for better visibility
plt.tight_layout()
plt.show()

# Create scatter matrix plot
scatter_matrix = pd.plotting.scatter_matrix(df1, figsize=(10, 10), s=10)

# Adjust layout to provide space for labels
plt.tight_layout()

# Increase the space between the plots to make room for the labels
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Adjust x-axis and y-axis labels
for ax in scatter_matrix.flatten():
    ax.set_xlabel(ax.get_xlabel(), fontsize=10, rotation=45)
    ax.set_ylabel(ax.get_ylabel(), fontsize=10, rotation=0, ha='right')

plt.show()
plt.figure(figsize=(8, 8))
plt.scatter(df1["Population growth (annual %)"],
    df1["Renewable energy consumption (% of total final energy consumption)"],
            10, marker="o")
plt.xlabel("Population growth (annual %)")
plt.ylabel("Renewable energy consumption (% of total final energy consumption)")
plt.show()


# Setup a scaler object
scaler = pp.RobustScaler()

# Extract columns
df_clust = df1[["Population growth (annual %)",
        "Renewable energy consumption (% of total final energy consumption)"]]

# Set up the scaler
scaler.fit(df_clust)

# Apply the scaling
df_norm = scaler.transform(df_clust)

# Print the result (now a NumPy array)
print(df_norm)

# Scatter plot of the normalized data
plt.figure(figsize=(8, 8))
plt.scatter(df_norm[:, 0], df_norm[:, 1], 10, marker="o")
plt.xlabel("Population growth (annual %)")
plt.ylabel("Renewable energy consumption (% of total final energy consumption)")
plt.show()


# Calculate silhouette score for 2 to 10 clusters
def one_silhouette(xy, n):
    """Calculates silhouette score for n clusters"""
    # Set up the clusterer with the number of expected clusters
    kmeans = KMeans(n_clusters=n, n_init=20, random_state=42)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)
    labels = kmeans.labels_
    # Calculate the silhouette score
    score = silhouette_score(xy, labels)
    return score



# Calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhouette(df_norm, ic)
    print(f"The silhouette score for {ic:3d} clusters is {score:7.4f}")

# Set up the clusterer with the number of expected clusters
kmeans = KMeans(n_clusters=3, n_init=20, random_state=0)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm)  # fit done on x, y pairs

# Extract cluster labels
labels = kmeans.labels_

# Extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen_original_scale = scaler.inverse_transform(cen)
xkmeans = cen_original_scale[:, 0]
ykmeans = cen_original_scale[:, 1]

# Extract x and y values of data points
x = df_clust["Population growth (annual %)"]
y = df_clust["Renewable energy consumption (% of total final energy consumption)"]

plt.figure(figsize=(8.0, 8.0))

# Plot data with kmeans cluster number
plt.scatter(x, y, c=labels, s=10, cmap='viridis', marker="o")

# Show cluster centers
plt.scatter(xkmeans, ykmeans, c='k', s=45, marker="d", label='Cluster Centers')

plt.xlabel("Population growth (annual %)")
plt.ylabel("Renewable energy consumption (% of total final energy consumption)")
plt.title('clusters between the 2 indicators')
plt.legend()
plt.show()
print(cen)


# Applying KMeans

kmeans = KMeans(n_clusters=3, n_init=20, random_state=0)
cluster_labels = kmeans.fit_predict(df_norm)

# Add the cluster labels to the dataset
df_clust["Cluster"] = cluster_labels
print(df_clust)
# Create subplots for each indicator
indicators = df_clust.columns[:-1]  # Exclude the 'Cluster' column
num_indicators = len(indicators)

fig, axes = plt.subplots(nrows=num_indicators, ncols=1, 
                         figsize=(12, 5 * num_indicators))

# Plot mean distribution for each indicator
for i, indicator in enumerate(indicators):
    sns.barplot(data=df_clust, x="Cluster", y=indicator,
                palette="viridis", ax=axes[i])

    axes[i].set_title(f"Mean Distribution for {indicator} across Clusters")
    axes[i].set_ylabel("Mean Value")
    axes[i].set_xlabel("Cluster")

plt.tight_layout()
plt.show()

temp = df_clust.groupby("Cluster").mean()
print(temp)

# giving the labels new name
labels = ["Cluster 0", "Cluster 1", "Cluster 2"]

# Check if the length of labels matches the number of clusters
if len(labels) == len(temp):
    temp.index = labels
    temp.plot(kind="bar", figsize=(12, 5))
    plt.title("Mean of Each Indicator with respect to each Cluster")
    plt.ylabel("Mean Value")
    plt.xlabel("Indicator")
    plt.show()


# Selecting one indicator from each cluster
selected_indicators = df_clust.groupby("Cluster").last().reset_index()

# Print the selected countries DataFrame
print(selected_indicators)


# Define the exponential function


def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 
    and growth rate g."""
    if isinstance(t, list) or isinstance(t, np.ndarray):
        t = [val - 2000 for val in t]
    else:
        t = t - 2000
    f = n0 * np.exp(g * np.array(t))
    return f


# Load the data
china_df = pd.read_csv("china_co2.csv")

# Use curve_fit to find the best parameters
param, covar = curve_fit(exponential, china_df["year"], china_df["co2"])

# Extract the optimal scale factor from the parameters
best_scale_factor = param[0]

print("Optimal Scale Factor:", best_scale_factor)

# Plot the original data and the best fit
plt.figure()
plt.plot(china_df["year"], exponential(
    china_df["year"], *param), label="Best Fit")
plt.plot(china_df["year"], china_df["co2"], label="Original Data")
plt.xlabel("Year")
plt.ylabel("CO2 Level")
plt.title('co2 emmsion in china')
plt.legend()
plt.show()



# Predict values for ten and twenty years in the future
future_years = np.linspace(1990, 2040, 100)
predicted_values = exponential(future_years, *param)


# Calculate confidence intervals for the fitted parameters
# Using the standard errors from the covariance matrix
std_dev = np.sqrt(np.diag(covar))
print(std_dev)
# You can adjust the multiplier based on your desired confidence level
conf_interval = 1.96

lower_limits = param - conf_interval * std_dev
upper_limits = param + conf_interval * std_dev

# Use error_prop to calculate 1 sigma error ranges

sigma_values = er.error_prop(future_years, exponential, param, covar)

# Plot confidence intervals with explicit label
plt.fill_between(future_years,
                 exponential(future_years, *lower_limits),
                 exponential(future_years, *upper_limits),
                 color='gray', alpha=0.2, label='Confidence Interval')

# Print and plot predicted values
plt.plot(china_df["year"], china_df["co2"], label="Original Data")
plt.plot(future_years, predicted_values, color='red', label='Predicted Values')

plt.title('co2 emmsion prediction for china')
# Add a legend
plt.legend()

plt.show()
