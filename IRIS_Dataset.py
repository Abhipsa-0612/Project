# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 22:50:20 2025

@author: Abhipsa
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')
df.columns = [col.replace('Cm', '') for col in df.columns]
df['Species'] = pd.Categorical(df['Species'])

print("First 5 rows:")
print(df.head())

print("\nData Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe())

# Distribution Plots
numeric_cols = df.select_dtypes(include='number').columns
sns.set(style='whitegrid')
plt.figure(figsize=(15, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 2, i)  # Use 3 rows and 2 columns
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Box Plots by Species
plt.figure(figsize=(15, 10))  # Larger figure to fit all subplots
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 2, i)  # 3 rows and 2 columns = fits up to 6 plots
    sns.boxplot(data=df, x='Species', y=col, palette='pastel')
    plt.title(f'{col} by Species')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Groupby Statistics
print("\nMean values by species:")
print(df.groupby('Species').mean())

print("\nStandard deviation by species:")
print(df.groupby('Species').std())

# Pivot Table
pivot = df.pivot_table(index='Species', values=numeric_cols, aggfunc='mean')
print("\nPivot Table (Mean values per species):")
print(pivot)

# Visual Storytelling

# Pairplot
sns.pairplot(df, hue='Species', palette='Set2')
plt.suptitle("Pairplot of Iris Features by Species", y=1.02)
plt.show()

# Swarmplot Example
plt.figure(figsize=(15, 5))
sns.swarmplot(data=df, x='Species', y='PetalLength', palette='Set1')
plt.title("Petal Length Distribution per Species")
plt.show()

# Barplot of Mean Features
mean_df = df.groupby('Species').mean().reset_index().melt(id_vars='Species', var_name='Feature', value_name='Mean')
plt.figure(figsize=(15, 10))
sns.barplot(data=mean_df, x='Feature', y='Mean', hue='Species')
plt.title("Average Feature Values per Species")
plt.show()

