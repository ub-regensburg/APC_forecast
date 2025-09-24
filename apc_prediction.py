#!/usr/bin/python3
# -*- coding: utf-8 -*-
#

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit



# Read the csv file
data = pd.read_csv("apc_de.csv")

# drop not needed columns
data.drop("doi", axis=1, inplace=True)
data.drop("issn", axis=1, inplace=True)
data.drop("issn_print", axis=1, inplace=True)
data.drop("issn_electronic", axis=1, inplace=True)
data.drop("issn_l", axis=1, inplace=True)
data.drop("license_ref", axis=1, inplace=True)
data.drop("indexed_in_crossref", axis=1, inplace=True)
data.drop("pmid", axis=1, inplace=True)
data.drop("pmcid", axis=1, inplace=True)
data.drop("url", axis=1, inplace=True)
data.drop("doaj", axis=1, inplace=True)
data.drop("ut", axis=1, inplace=True)
data.drop("is_hybrid", axis=1, inplace=True)

data.head()

print( data.info())
print( data.describe() )

# plot the average APC values
fig, ax = plt.subplots(figsize=(12, 4))
average_week_demand = data.groupby(["period"])["euro"].mean()
average_week_demand.plot(ax=ax)
_ = ax.set(
    title="Durchnitt APCs",
    xticks=[2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,2019,2020,2021,2022,2023,2024],
    xlabel="Jahr",
    ylabel="Euro",
)

# Encode the columns 

le = LabelEncoder()
encoded = le.fit_transform(data['institution'])
data.drop("institution", axis=1, inplace=True)
data["institution"] = encoded

encoded = le.fit_transform(data['publisher'])
data.drop("publisher", axis=1, inplace=True)
data["publisher"] = encoded

encoded = le.fit_transform(data['journal_full_title'])
data.drop("journal_full_title", axis=1, inplace=True)
data["journal_full_title"] = encoded

# print information of the data
print( data.info())
print( data.describe() )

data.hist(bins=10, figsize=(12,10))
plt.show()

# assign y as normalized cost, X as data without costs 
y = data["euro"] / data["euro"].max() 
X = data.drop("euro", axis="columns")

X["publisher"].value_counts()

#aply machine learning
ts_cv = TimeSeriesSplit(
    n_splits=5,
    gap=48,
    max_train_size=100000,
    test_size=2000,
)

all_splits = list(ts_cv.split(X, y))
train_0, test_0 = all_splits[0]

print ( data.iloc[test_0] )
print (X.iloc[test_0])
print (X.iloc[train_0])





