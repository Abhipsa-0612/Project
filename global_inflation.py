# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 12:37:56 2025

@author: Abhipsa
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#import warnings
#warnings.filterwarnings("ignore")

df = pd.read_csv("global_inflation_data.csv")

# Reshape from wide to long format
df_long = df.melt(id_vars=['country_name', 'indicator_name'],
                  var_name='year',
                  value_name='inflation_rate')

np.random.seed(42)
df_long['gdp'] = np.random.normal(25000, 15000, size=len(df_long))
df_long['unemployment_rate'] = np.random.normal(6.5, 3.0, size=len(df_long))

# Clean year column
df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce')
df_long.dropna(subset=['year', 'inflation_rate'], inplace=True)
df_long.reset_index(drop=True, inplace=True)

print("First 5 rows:")
print(df.head())

print("\nData Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Descriptive Stats & Distribution Plots
print(df_long['inflation_rate'].describe(),"\n\n")

# Histogram + KDE
plt.figure(figsize=(10, 5))
sns.histplot(df_long['inflation_rate'], bins=40, kde=True, color='skyblue')
plt.title("Distribution of Global Inflation Rates")
plt.grid(True)
plt.show()

# Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x='inflation_rate', data=df_long, color='salmon')
plt.title("Boxplot of Global Inflation Rates")
plt.grid(True)
plt.show()

# Global Inflation Trend Over Time
global_trend = df_long.groupby('year')['inflation_rate'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=global_trend, x='year', y='inflation_rate', marker='o')
plt.title("Global Average Inflation Rate Over Time")
plt.ylabel("Inflation Rate (%)")
plt.grid(True)
plt.show()

# Country-wise Comparison (Top 5 Inflation)
top_countries = df_long.groupby('country_name')['inflation_rate'].mean().nlargest(5).index.tolist()

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_long[df_long['country_name'].isin(top_countries)],
             x='year', y='inflation_rate', hue='country_name')
plt.title("Top 5 Countries with Highest Inflation (Trend Over Years)")
plt.grid(True)
plt.show()

# Outlier Detection
top_outliers = df_long.sort_values(by='inflation_rate', ascending=False).head(10)
print(top_outliers[['country_name', 'year', 'inflation_rate']])

# Correlation Heatmap (Simulated GDP & Unemployment)
plt.figure(figsize=(8, 5))
sns.heatmap(df_long[['inflation_rate', 'gdp', 'unemployment_rate']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation: Inflation, GDP, Unemployment")
plt.show()

# PCA + KMeans Clustering
# Aggregate to country level
grouped = df_long.groupby('country_name')[['inflation_rate', 'gdp', 'unemployment_rate']].mean()
grouped.dropna(inplace=True)

scaler = StandardScaler()
scaled = scaler.fit_transform(grouped)

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(pca_result)

# Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='Set2')
plt.title("Country Clusters Based on Inflation, GDP & Unemployment")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()

# Heatmap Matrix (Country × Year)
heat_df = df_long.pivot_table(index='country_name', columns='year', values='inflation_rate')
plt.figure(figsize=(18, 12))
sns.heatmap(heat_df, cmap="coolwarm", linewidths=0.5)
plt.title("Inflation Heatmap by Country and Year")
plt.xlabel("Year")
plt.ylabel("Country")
plt.show()

