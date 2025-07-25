# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:50:10 2025

@author: Abhipsa
"""
# Customer Segmentation for a Retail Store - K - Means and Elbow Menthod
#Dataset - Customer Segmentation for a Retail Store
#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# Import Dataset, Read the CSV file
df = pd.read_csv("Mall_Customers.csv")

# Creating copy of the dataset
df_clean = df.copy()

# Understanding the Dataset

# First View of the dataset - Printing the first 5 Rows of the Dataset
print("\nThe first 5 Rows of the Dataset: ")
print("\n", df_clean.head())
print("\nData Types of each of the Columns/Features of the Dataset: ")
print(df_clean.columns.tolist())

# Dataset Rows & Columns count
print("\nNumber of Rows in the Dataset:", df_clean.shape[0])
print("\nNumber of Columns in the Dataset:", df_clean.shape[1])

# Dataset Information
print("\nDataset Information: ")
print(df_clean.info())

# Dataset Duplicate Value Count
print("\nNumber of duplicate Values of the Dataset:", df_clean.duplicated().sum())

# Missing Values of the Dataset
print("\nMissing Values present in each columns of the Dataset")
print(df_clean.isnull().sum())

print("Unique Values in each Columns of the Dataset: ")
print(df_clean.nunique())

# Data Pre - Processing
#Drop CustomerID as it's not useful for clustering
df_pp = df.drop('CustomerID', axis=1)

# Updated Dataset after Dropping the column
print("\n",df_pp.head())
print(df_pp.columns.tolist())
print("\nThe number of columns of the updated dataset: ",df_pp.shape[1])

# Encode 'Genre' column
le = LabelEncoder()
df_clean['Genre'] = le.fit_transform(df_clean['Genre'])  # Male=1, Female=0

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_clean)

# Convert scaled data to DataFrame
df_scaled = pd.DataFrame(scaled_features, columns=df_clean.columns)

# Elbow Method to find optimal k
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Apply KMeans with k=5
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_scaled)

# Add cluster labels to original data
df['Cluster'] = clusters

# Visualize clusters (Annual Income vs Spending Score)
plt.figure(figsize=(10, 6))
plt.scatter(df[df['Cluster'] == 0]['Annual Income (k$)'],
            df[df['Cluster'] == 0]['Spending Score (1-100)'],
            s=80, c='red', label='Cluster 1')

plt.scatter(df[df['Cluster'] == 1]['Annual Income (k$)'],
            df[df['Cluster'] == 1]['Spending Score (1-100)'],
            s=80, c='blue', label='Cluster 2')

plt.scatter(df[df['Cluster'] == 2]['Annual Income (k$)'],
            df[df['Cluster'] == 2]['Spending Score (1-100)'],
            s=80, c='green', label='Cluster 3')

plt.scatter(df[df['Cluster'] == 3]['Annual Income (k$)'],
            df[df['Cluster'] == 3]['Spending Score (1-100)'],
            s=80, c='cyan', label='Cluster 4')

plt.scatter(df[df['Cluster'] == 4]['Annual Income (k$)'],
            df[df['Cluster'] == 4]['Spending Score (1-100)'],
            s=80, c='magenta', label='Cluster 5')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, df_clean.columns.get_loc("Annual Income (k$)")],
            kmeans.cluster_centers_[:, df_clean.columns.get_loc("Spending Score (1-100)")],
            s=300, c='yellow', marker='X', label='Centroids')

plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.legend()
plt.grid(True)
plt.show()






