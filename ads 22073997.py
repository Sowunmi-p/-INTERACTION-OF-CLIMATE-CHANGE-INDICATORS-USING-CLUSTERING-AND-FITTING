#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:51:22 2023

@author: apple
"""
# import of libraries

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import errors as er


# using function read_csv_and_set_index to read and set index for my dataframe
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

    df = pd.read_csv(file_name, index_col='Country Name')
    tranposed_df = df.T

    return df, tranposed_df


# using function read_csv to read my data
def read_csv(file_name):
    """
    Read a CSV file into a DataFrame

    Parameters:
    - file_name (str): The name of the CSV file.

    Returns:
    - pd.DataFrame: The DataFrame.
    """

    df = pd.read_csv(file_name)
    return df


# Use the function to read the file
fitting_df = read_csv("adsfitting.csv")

# Display the DataFrame
print(fitting_df)


# Load the data
file_name = 'ads2.csv'
df1, transposed_df = read_csv_and_set_index(file_name)


# display my data
print(df1)
print(transposed_df)

# Drop null values from the transposed DataFrame
transposed_df = transposed_df.dropna()
print(transposed_df)

# removing the first column for from the data
df1 = df1.iloc[1:]
print(df1)

df1 = df1.reset_index(drop=True)  # index is not copied into index column
print(df1)


print(df1.columns)

# the statistical infomation about my data
print(df1.describe())

# creating another varaible called corr to check my data correlation
corr = df1.corr()
print(corr.round(3))

# fig, ax = plt.subplots(figsize=(8, 8))
plt.figure(figsize=[8, 8])

# heatmap
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

# plot the unnormalized data
plt.figure(figsize=(8, 8))
plt.scatter(df1["Population growth"],
            df1["Renewable energy consumption"],
            10, marker="o")
plt.xlabel("Population growth", fontsize=15, fontweight='bold', color='black')
plt.ylabel("Renewable energy consumption", fontsize=15, fontweight='bold',
           color='black')
plt.title('Unnormalized data for the 2 indicators',
          fontsize=20, fontweight='bold', color='black')
plt.show()


# Setup a scaler object
scaler = pp.RobustScaler()

# Extract columns
df_clust = df1[["Population growth",
                "Renewable energy consumption"]]
# print the clustered df
print(df_clust)

# Set up the scaler
scaler.fit(df_clust)

# Apply the scaling
df_norm = scaler.transform(df_clust)

# Print the result (now a NumPy array)
print(df_norm)

# Scatter plot of the normalized data
plt.figure(figsize=(8, 8))
plt.scatter(df_norm[:, 0], df_norm[:, 1], 10, marker="o")
plt.xlabel("Population growth", fontsize=15, fontweight='bold', color='black')
plt.ylabel("Renewable energy consumption", fontsize=15, fontweight='bold',
           color='black')
plt.title('Normalized data for the 2 indicators',
          fontsize=20, fontweight='bold', color='black')
plt.show()


# Define a function to calculate silhouette score for 2-10 number of
# clusters
def calculate_silhouette_score(data, num_clusters):
    """
    Calculate silhouette score for a given number of clusters.

    Parameters:
    - data: The input data for clustering
    - num_clusters: Number of clusters to form

    Returns:
    - score: Silhouette score for the clustering
    """
    # Set up the KMeans clusterer with the specified number of clusters
    kmeans = KMeans(n_clusters=num_clusters, n_init=20, random_state=42)
    
    # Fit the data to the KMeans model, and store the results in the kmeans 
    #object
    kmeans.fit(data)
    
    # Get the cluster labels assigned to each data point
    labels = kmeans.labels_
    
    # Calculate the silhouette score for the clustering
    score = silhouette_score(data, labels)
    
    # Return the calculated silhouette score
    return score


# Evaluate silhouette scores for different cluster numbers (2 to 10)
# using the one_silhouette function
# and print the results for analysis.
for ic in range(2, 11):
    score = calculate_silhouette_score(df_norm, ic)
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
x = df_clust["Population growth"]
y = df_clust["Renewable energy consumption"]

plt.figure(figsize=(8.0, 8.0))

# Plot data with kmeans cluster number
plt.scatter(x, y, c=labels, s=10, cmap='viridis', marker="o")

# Show cluster centers
plt.scatter(xkmeans, ykmeans, c='k', s=45, marker="d", label='Cluster Centers')

plt.xlabel("Population growth", fontsize=15, fontweight='bold',
           color='black')
plt.ylabel("Renewable energy consumption",
           fontsize=15, fontweight='bold',
           color='black')
plt.title('clusters between the 2 indicators',
          fontsize=20, fontweight='bold', color='black')
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

    """
    Plot mean distribution for each indicator across clusters.

    Parameters:
   - indicators (list): List of indicator names to be plotted.
   - df_clust (pd.DataFrame): DataFrame containing clustered data.
   - axes (array): Array of subplot axes.

  Returns:
  None
  """

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
    """
   Check if the length of labels matches the number of clusters and 
   plot a bar chart.

   Parameters:
   - labels (list): List of cluster labels.
   - temp (pd.DataFrame): DataFrame containing mean values for 
   each indicator across clusters.

   Returns:
   None
   """

    temp.index = labels
    temp.plot(kind="bar", figsize=(12, 5))
    plt.title("Mean of Each Indicator with respect to each Cluster",
              fontsize=20, fontweight='bold', color='black')
    plt.ylabel("Mean Value", fontsize=15, fontweight='bold', color='black')
    plt.xlabel("clusters",  fontsize=15, fontweight='bold', color='black')
    plt.show()


# Selecting one indicator from each cluster
selected_indicators = df_clust.groupby("Cluster").last().reset_index()

# Print the selected countries DataFrame
print(selected_indicators)

# FITTING OF MY DATA

# Data cleaning/wrangling
# drop columns that are not needed(country code and series code)
fitting_df.drop(columns=['Country Code', 'Series Code'], inplace=True)
print(fitting_df)

# from my clustering i picked 3 countries from the cluster 1 for my fitting
#which are United kingdom, Brazil and south africa for Renewable enery
#consumption

#created a new dataframe for uk data
uk_df = fitting_df[fitting_df.loc[:, "Country Name"]
                   == 'United Kingdom']
print(uk_df)

#created a new dataframe for brazil data 
brazil_df = fitting_df[fitting_df.loc[:, "Country Name"]
                       == 'Brazil']
print(brazil_df)

#created a new dataframe for south africa data
South_Africa_df = fitting_df[fitting_df.loc[:, "Country Name"]
                             == 'South Africa']
print(South_Africa_df)
# Define the exponential function


def calculate_exponential(t, n0, growth_rate):
    """
    Calculates an exponential function with scale factor n0 and growth rate g.

    Parameters:
    - t: Time variable (can be a single value, list, or numpy array)
    - n0: Scale factor for the exponential function
    - growth_rate: Growth rate for the exponential function

    Returns:
    - f: Result of the exponential function
    """

    # Adjust time values if provided as a list or numpy array
    if isinstance(t, list) or isinstance(t, np.ndarray):
        t = [val - 2000 for val in t]
    else:
        t = t - 2000
    
    # Calculate the exponential function using numpy
    result = n0 * np.exp(growth_rate * np.array(t))
    
    # Return the calculated exponential function values
    return result


# Use curve_fit to find the best parameters for uk data
param1, covar1 = curve_fit(calculate_exponential, uk_df["year"],
                           uk_df["Renewable energy consumption"])

# Extract the optimal scale factor from the parameters
best_scale_factor = param1[0]

print("Optimal Scale Factor:", best_scale_factor)

# Plot the original data and the best fit UK data
plt.figure()
plt.plot(uk_df["year"], calculate_exponential(
    uk_df["year"], *param1), label="Best Fit")
plt.plot(uk_df["year"], uk_df["Renewable energy consumption"],
         label="Original Data")
plt.xlabel("Year", fontsize=10, fontweight='bold', color='black')
plt.ylabel("Renewable energy consumption Level", fontsize=10,
           fontweight='bold', color='black')
plt.title('Renewable energy consumption in UK',
          fontsize=15, fontweight='bold', color='black')
plt.legend()
plt.show()


# Use curve_fit to find the best parameters for Brazil data
param2, covar2 = curve_fit(calculate_exponential, brazil_df["year"],
                           brazil_df["Renewable energy consumption"])

# Extract the optimal scale factor from the parameters
best_scale_factor = param2[0]

print("Optimal Scale Factor:", best_scale_factor)

# Plot the original data and the best fit Brazil data
plt.figure()
plt.plot(brazil_df["year"], calculate_exponential(
    brazil_df["year"], *param2), label="Best Fit")
plt.plot(brazil_df["year"], brazil_df["Renewable energy consumption"],
         label="Original Data")
plt.xlabel("Year", fontsize=10, fontweight='bold', color='black')
plt.ylabel("Renewable energy consumption Level", fontsize=10,
           fontweight='bold', color='black')
plt.title('Renewable energy consumption in Brazil',
          fontsize=15, fontweight='bold', color='black')
plt.legend()
plt.show()

# Use curve_fit to find the best parameters for south africa data
param3, covar3 = curve_fit(calculate_exponential, South_Africa_df["year"],
                           South_Africa_df["Renewable energy consumption"])

# Extract the optimal scale factor from the parameters
best_scale_factor = param3[0]

print("Optimal Scale Factor:", best_scale_factor)

# Plot the original data and the best fit South Africa data
plt.figure()
plt.plot(South_Africa_df["year"], calculate_exponential(
    South_Africa_df["year"], *param3), label="Best Fit")
plt.plot(South_Africa_df["year"],
         South_Africa_df["Renewable energy consumption"],
         label="Original Data")
plt.xlabel("Year", fontsize=10, fontweight='bold', color='black')
plt.ylabel("Renewable energy consumption Level", fontsize=10,
           fontweight='bold', color='black')
plt.title('Renewable energy consumption in South Africa',
          fontsize=15, fontweight='bold', color='black')
plt.legend()
plt.show()

# Predict values for ten and twenty years in the future for United kingdom 
#renewable energy
future_years = np.linspace(1998, 2040, 100)
predicted_values = calculate_exponential(future_years, *param1)


# Calculate confidence intervals for the fitted parameters
# Using the standard errors from the covariance matrix
std_dev = np.sqrt(np.diag(covar1))
print(std_dev)

# confidence level
conf_interval = 1.95

lower_limits = param1 - conf_interval * std_dev
upper_limits = param1 + conf_interval * std_dev

# Use error_prop to calculate 1 sigma error ranges

sigma_values = er.error_prop(future_years, calculate_exponential, param1,
                             covar1)

# Plot confidence intervals with explicit label
plt.fill_between(future_years,
                 calculate_exponential(future_years, *lower_limits),
                 calculate_exponential(future_years, *upper_limits),
                 color='gray', alpha=0.2, label='Confidence Interval')

# Print and plot predicted values
plt.plot(uk_df["year"], uk_df["Renewable energy consumption"],
         label="Original Data")
plt.plot(future_years, predicted_values, color='red',
         label='Predicted Values')
plt.xlabel("Year", fontsize=10, fontweight='bold', color='black')
plt.ylabel("Renewable energy consumptionLevel", fontsize=10,
           fontweight='bold', color='black')
plt.title('Renewable energy consumption prediction for UK',
          fontsize=15, fontweight='bold', color='black')

# Add a legend
plt.legend()

plt.show()
