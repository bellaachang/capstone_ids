#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IDS Capstone: Assessing Professor Effectiveness (APE) with Data Science Methodologies
Authors: Bella Chang, Kristina Fujimoto, Emily Wang
"""

# 1. Import Libraries and Set Seed
import numpy as np 
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, roc_curve, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler

seed = 18738336
rng = np.random.default_rng(seed)

# 2. Load and Merge Data
capstone_num = pd.read_csv("rmpCapstoneNum.csv", names=[
    'Average Rating', 'Average Difficulty', 'Number of Ratings',
    'Received a Pepper?', '% of Students Who Would Retake',
    '# of Ratings from Online Classes', 'Male Prof?', 'Female Prof?'])

capstone_qual = pd.read_csv("rmpCapstoneQual.csv", names=[
    'Major/Field', 'University', 'US State'])

capstone_tags = pd.read_csv("rmpCapstoneTags.csv", names=[
    'Tough grader', 'Good feedback', 'Respected', 'Lots to read',
    'Participation matters', "Don't skip class or you will not pass",
    'Lots of homework', 'Inspirational', 'Pop quizzes!', 'Accessible',
    'So many papers', 'Clear grading', 'Hilarious', 'Test heavy',
    'Graded by few things', 'Amazing lectures', 'Caring', 'Extra credit',
    'Group projects', 'Lecture heavy'])

# Combine datasets and clean nulls
df = pd.concat([capstone_num, capstone_qual, capstone_tags], axis=1)
df = df.dropna(subset=['Average Rating']).reset_index(drop=True)
df = df.drop(columns=['% of Students Who Would Retake'])

# 3. Data Preprocessing
# Drop ambiguous gender
df = df[(df['Male Prof?'] != df['Female Prof?'])]

# Filter by ratings threshold
median_ratings = df['Number of Ratings'].median()
df = df[df['Number of Ratings'] >= median_ratings].reset_index(drop=True)

# Normalize tags
tag_cols = capstone_tags.columns
df['Total Tags'] = df[tag_cols].sum(axis=1)
df[tag_cols] = df[tag_cols].div(df['Total Tags'], axis=0).fillna(0)

# Prepare cleaned datasets
df_num = df[['Average Rating', 'Average Difficulty', 'Number of Ratings',
             'Received a Pepper?', '# of Ratings from Online Classes',
             'Male Prof?', 'Female Prof?']].copy()

df_tags = df[tag_cols].copy()
df_combined = pd.concat([df_num, df_tags], axis=1)

# (Further analysis and modeling steps follow in the full version...)


# 4. Exploratory Gender Bias Analysis

# A. Central Tendency Test - Ratings by Gender
male_ratings = df[df['Male Prof?'] == 1]['Average Rating']
female_ratings = df[df['Female Prof?'] == 1]['Average Rating']
u_stat, p_val_rating = stats.mannwhitneyu(male_ratings, female_ratings, alternative='greater')

# B. Variance Comparison via Permutation Test
obs_var_diff = abs(male_ratings.var() - female_ratings.var())
perm_diffs = []
combined = np.concatenate([male_ratings, female_ratings])

for _ in range(10000):
    np.random.shuffle(combined)
    perm_male = combined[:len(male_ratings)]
    perm_female = combined[len(male_ratings):]
    perm_diffs.append(abs(perm_male.var() - perm_female.var()))

p_val_variance = np.mean(np.array(perm_diffs) >= obs_var_diff)

# C. Effect Size Estimates (Cohen’s d)
def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std

d_rating = cohens_d(male_ratings, female_ratings)

# 5. Tag-Based Gender Differences
tag_pvals = {}
tag_effects = {}

for tag in tag_cols:
    m = df[df['Male Prof?'] == 1][tag]
    f = df[df['Female Prof?'] == 1][tag]
    stat, p = stats.mannwhitneyu(m, f)
    tag_pvals[tag] = p
    tag_effects[tag] = cohens_d(m, f)

# 6. Difficulty Analysis
male_diff = df[df['Male Prof?'] == 1]['Average Difficulty']
female_diff = df[df['Female Prof?'] == 1]['Average Difficulty']

ks_stat, ks_pval = stats.ks_2samp(male_diff, female_diff)
t_stat_diff, p_val_diff = stats.ttest_ind(male_diff, female_diff, equal_var=True)
d_diff = cohens_d(male_diff, female_diff)

# 7. Predictive Modeling
# A. Ridge Regression with Numeric Predictors
X_num = df[['Average Difficulty', 'Number of Ratings', '# of Ratings from Online Classes', 'Female Prof?']]
y_rating = df['Average Rating']
ridge_num = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5).fit(X_num, y_rating)
r2_num = ridge_num.score(X_num, y_rating)
rmse_num = np.sqrt(mean_squared_error(y_rating, ridge_num.predict(X_num)))

# B. Ridge Regression with Tags
X_tags = df_tags
ridge_tags = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5).fit(X_tags, y_rating)
r2_tags = ridge_tags.score(X_tags, y_rating)
rmse_tags = np.sqrt(mean_squared_error(y_rating, ridge_tags.predict(X_tags)))

# C. Ridge Regression for Difficulty
y_diff = df['Average Difficulty']
ridge_diff = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5).fit(X_tags, y_diff)
r2_diff = ridge_diff.score(X_tags, y_diff)
rmse_diff = np.sqrt(mean_squared_error(y_diff, ridge_diff.predict(X_tags)))

# D. Logistic Regression for Pepper
X_all = pd.concat([X_num, X_tags], axis=1)
y_pepper = df['Received a Pepper?']
X_train, X_test, y_train, y_test = train_test_split(X_all, y_pepper, test_size=0.4, random_state=seed)

logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_prob)

# 8. Exploratory: NY vs FL Positivity Bias
positive_tags = ['Caring', 'Inspirational', 'Amazing lectures', 'Good feedback', 'Respected']
df['Positive Tag Mean'] = df[positive_tags].mean(axis=1)
ny_pos = df[df['US State'] == 'NY']['Positive Tag Mean']
fl_pos = df[df['US State'] == 'FL']['Positive Tag Mean']

t_stat_pos, p_val_pos = stats.ttest_ind(ny_pos, fl_pos, equal_var=True)
d_pos = cohens_d(ny_pos, fl_pos)

# Print summary of findings (can be expanded into plots or reports)
print(f"Gender Bias - p-value (ratings): {p_val_rating:.5f}, Effect size (d): {d_rating:.3f}")
print(f"Gender Variance Diff - p-value: {p_val_variance:.5f}")
print(f"Difficulty Difference - p-value: {p_val_diff:.5f}, Effect size (d): {d_diff:.3f}")
print(f"R2 (numeric predictors): {r2_num:.3f}, RMSE: {rmse_num:.3f}")
print(f"R2 (tags): {r2_tags:.3f}, RMSE: {rmse_tags:.3f}")
print(f"R2 (difficulty from tags): {r2_diff:.3f}, RMSE: {rmse_diff:.3f}")
print(f"AUC (pepper classifier): {auc_score:.3f}")
print(f"Positivity Bias (NY vs FL) - p-value: {p_val_pos:.3f}, Effect size (d): {d_pos:.3f}")
