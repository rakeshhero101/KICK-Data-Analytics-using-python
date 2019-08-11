# Importing the packages
import numpy as np
import pandas as pd
import pydot
from io import StringIO
from sklearn.tree import export_graphviz

def dataPrep():
    # Reading the data
    df = pd.read_csv("CaseStudyData.csv")
    
    # Removing the unwanted columns from the dataframe df
    df.drop(['PurchaseID', 'PurchaseTimestamp','PRIMEUNIT','AUCGUART','WheelTypeID', 'MMRCurrentRetailRatio', 'ForSale'], axis=1,       inplace=True)

    # Converting the '?' into nan value 
    df = df.replace('?', np.nan)

    # Changing the PurchaseDate series
    # Splitting it into year and month series
    df['PurchaseYear'] = pd.DatetimeIndex(df['PurchaseDate']).year
    df['PurchaseMonth'] = pd.DatetimeIndex(df['PurchaseDate']).month
    
    #Removing the PurchaseDate series
    df.drop(['PurchaseDate'], axis=1, inplace=True)

    # Rearranging the index
    cols = df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df = df.reindex(cols, axis = 1)

    # Filling the missing value of categorical column with the frequent value or mode
    df['Color'].fillna(df['Color'].mode()[0], inplace=True)
    df['Auction'].fillna(df['Auction'].mode()[0], inplace=True)
    df['Make'].fillna(df['Make'].mode()[0], inplace=True)
    df['WheelType'].fillna(df['WheelType'].mode()[0], inplace=True)
    df['Nationality'].fillna(df['Nationality'].mode()[0], inplace=True)
    df['Size'].fillna(df['Size'].mode()[0], inplace=True)
    df['VNST'].fillna(df['VNST'].mode()[0], inplace=True)
    df['TopThreeAmericanName'].fillna(df['TopThreeAmericanName'].mode()[0], inplace=True)

    # chnaging all letters to lowercase
    df['Transmission'] = df['Transmission'].str.lower()
    df['Transmission'] = df['Transmission'].replace('0', 'no')
    df['Transmission'].fillna(df['Transmission'].mode()[0], inplace=True)

    # Imputation of error values
    df['IsOnlineSale'] = df['IsOnlineSale'].replace('0.0', 0)
    df['IsOnlineSale'] = df['IsOnlineSale'].replace('-1', 1)
    df['IsOnlineSale'] = df['IsOnlineSale'].replace('1.0', 1)
    df['IsOnlineSale'] = df['IsOnlineSale'].replace('2.0', 1)
    df['IsOnlineSale'] = df['IsOnlineSale'].replace('4.0', 1)

    # Filling the missing value of numerical column with median value
    df['VehYear'].fillna(df['VehYear'].median(), inplace=True)
    df['VehOdo'].fillna(df['VehOdo'].median(), inplace=True)
    df['WarrantyCost'].fillna(df['WarrantyCost'].median(), inplace=True)

    # Since all the interval/numeric is an object type
    # We convert them to floattype for calculation purposes
    # we use .str to replace and then convert to float
    # fill the missing values
    df['MMRAcquisitionAuctionAveragePrice'] = df.MMRAcquisitionAuctionAveragePrice.str.replace('$', '').astype(float)
    df['MMRAcquisitionAuctionAveragePrice'].fillna(df['MMRAcquisitionAuctionAveragePrice'].median(), inplace=True)
    
    df['MMRAcquisitionAuctionCleanPrice'] = df.MMRAcquisitionAuctionCleanPrice.str.replace('$', '').astype(float)
    df['MMRAcquisitionAuctionCleanPrice'].fillna(df['MMRAcquisitionAuctionCleanPrice'].median(), inplace=True)

    df['MMRAcquisitionRetailAveragePrice'] = df.MMRAcquisitionRetailAveragePrice.str.replace('$', '').astype(float)
    df['MMRAcquisitionRetailAveragePrice'].fillna(df['MMRAcquisitionRetailAveragePrice'].median(), inplace=True)

    df['MMRAcquisitonRetailCleanPrice'] = df.MMRAcquisitonRetailCleanPrice.str.replace('$', '').astype(float)
    df['MMRAcquisitonRetailCleanPrice'].fillna(df['MMRAcquisitonRetailCleanPrice'].median(), inplace=True)

    df['MMRCurrentAuctionAveragePrice'] = df.MMRCurrentAuctionAveragePrice.str.replace('$', '').astype(float)
    df['MMRCurrentAuctionAveragePrice'].fillna(df['MMRCurrentAuctionAveragePrice'].median(), inplace=True)

    df['MMRCurrentAuctionCleanPrice'] = df.MMRCurrentAuctionCleanPrice.str.replace('$', '').astype(float)
    df['MMRCurrentAuctionCleanPrice'].fillna(df['MMRCurrentAuctionCleanPrice'].median(), inplace=True)

    df['MMRCurrentRetailAveragePrice'] = df.MMRCurrentRetailAveragePrice.str.replace('$', '').astype(float)
    df['MMRCurrentRetailAveragePrice'].fillna(df['MMRCurrentRetailAveragePrice'].median(), inplace=True)

    df['MMRCurrentRetailCleanPrice'] = df.MMRCurrentRetailCleanPrice.str.replace('$', '').astype(float)
    df['MMRCurrentRetailCleanPrice'].fillna(df['MMRCurrentRetailCleanPrice'].median(), inplace=True)

    df['VehBCost'] = df.VehBCost.str.replace('$', '').astype(float)
    df['VehBCost'].fillna(df['VehBCost'].median(), inplace=True)

    df['IsOnlineSale'] = df.IsOnlineSale.str.replace('$', '').astype(float)
    df['IsOnlineSale'].fillna(df['IsOnlineSale'].median(), inplace=True)

    #Replaciing the 0 value with np.nan
    df['MMRAcquisitionAuctionAveragePrice'].replace(0, np.nan ,inplace=True)
    df['MMRAcquisitionAuctionCleanPrice'].replace(0, np.nan ,inplace=True)
    df['MMRAcquisitionRetailAveragePrice'].replace(0, np.nan ,inplace=True)
    df['MMRAcquisitonRetailCleanPrice'].replace(0, np.nan ,inplace=True)
    df['MMRCurrentAuctionAveragePrice'].replace(0, np.nan ,inplace=True)
    df['MMRCurrentAuctionCleanPrice'].replace(0, np.nan ,inplace=True)
    df['MMRCurrentRetailAveragePrice'].replace(0, np.nan ,inplace=True)
    df['MMRCurrentRetailCleanPrice'].replace(0, np.nan ,inplace=True)
    
    #Filling np.nan with median value
    df['MMRAcquisitionAuctionAveragePrice'].fillna(df['MMRAcquisitionAuctionAveragePrice'].median(), inplace=True) 
    df['MMRAcquisitionAuctionCleanPrice'].fillna(df['MMRAcquisitionAuctionCleanPrice'].median(), inplace=True) 
    df['MMRAcquisitionRetailAveragePrice'].fillna(df['MMRAcquisitionRetailAveragePrice'].median(), inplace=True) 
    df['MMRAcquisitonRetailCleanPrice'].fillna(df['MMRAcquisitonRetailCleanPrice'].median(), inplace=True) 
    df['MMRCurrentAuctionAveragePrice'].fillna(df['MMRCurrentAuctionAveragePrice'].median(), inplace=True) 
    df['MMRCurrentAuctionCleanPrice'].fillna(df['MMRCurrentAuctionCleanPrice'].median(), inplace=True) 
    df['MMRCurrentRetailAveragePrice'].fillna(df['MMRCurrentRetailAveragePrice'].median(), inplace=True) 
    df['MMRCurrentRetailCleanPrice'].fillna(df['MMRCurrentRetailCleanPrice'].median(), inplace=True) 
    
    
    # one hot encoding all categorical variables
    df = pd.get_dummies(df)
    
    #Changing the float into int
    df['VehYear'] = df['VehYear'].astype(int)
    df['PurchaseYear'] = df['PurchaseYear'].astype(int)
    df['PurchaseMonth'] = df['PurchaseMonth'].astype(int)
    df['VehOdo'] = df['VehOdo'].astype(int)
    
    return df

def analyse_feature_importance(dm_model, feature_names, n_to_display=20):
    
    # grab feature importances from the model
    importances = dm_model.feature_importances_
    
    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)

    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]

    for i in indices:
        print(feature_names[i], ':', importances[i])


