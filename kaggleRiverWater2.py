import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
import copy
import warnings
warnings.filterwarnings("ignore")

train=pd.read_csv("trainRiverWater.csv")

rf=RandomForestRegressor(
    n_estimators=1000, #1000
    max_depth=7, #7
    n_jobs=-1,
    random_state=42)

def jacknifePlus(X,y,rowsToBeImputed):
    kf=KFold(n_splits=len(y), shuffle=True, random_state=0)   
    residuals=[]
    models=[]
    
    for trainIndex, testIndex in kf.split(X):
        xTrain, xTest = X.iloc[trainIndex], X.iloc[testIndex]
        yTrain, yTest = y.iloc[trainIndex], y.iloc[testIndex]       
        rf.fit(xTrain,yTrain)
        models.append(copy.copy(rf))
        residuals.append(yTest-rf.predict(xTest))
        print('.',end="")
    print()   
    
    predictionDistributions = [[] for _ in range(rowsToBeImputed.shape[0])]
    for index in range(rowsToBeImputed.shape[0]): 
        predictions = np.column_stack([model.predict(rowsToBeImputed.iloc[index].to_numpy().reshape(1,-1)) for model in models])
        predictionDistributions[index] = [np.median(predictions,axis=1)-resid for resid in residuals]
        
    return predictionDistributions

def synthesizeData(rows,predictionDistributions,targetColIndex):
    imputeCopies= train.loc[rows*16]
    print(imputeCopies)
    numberOfRows=len(rows)
    for obs in range(numberOfRows):
        for i in range(16):
            percentile=range(5,96,6)[i]
            value = np.quantile(predictionDistributions[obs], percentile/100)
            imputeCopies.iloc[i*numberOfRows+obs,targetColIndex]=value
    print(imputeCopies)
    return imputeCopies


def impute(data,rows,targetCol):
    
    features= data.iloc[rows].columns[data.iloc[rows].notna().all()].tolist()
    target=[train.columns[targetCol]]
    
    dataSubset=data.dropna(subset=features+target)
    X=dataSubset[features]
    y=dataSubset[target]
    rowsToBeImputed = data.iloc[rows]
    rowsToBeImputed=rowsToBeImputed[features]
    predictionDistributions=jacknifePlus(X,y,rowsToBeImputed)
    return synthesizeData(rows,predictionDistributions,targetColIndex=targetCol)    

###
#56,97

trainSubset=train[['target','O2_1','O2_2','NH4_1','NH4_2','NO2_1','NO2_2','NO3_1','NO3_2','BOD5_1','BOD5_2']]

imp_56_97=impute(trainSubset,[56,97],2)

train=pd.concat([train.iloc[:]]*16,ignore_index=True)
train=train[train['O2_1'].notna()]

data=pd.concat([train,imp_56_97],ignore_index=True)

#modify other columns of less relevance
data['NH4_1'].fillna(data['NH4_1'].mean(),inplace=True)
data['NO2_1'].fillna(data['NO2_1'].mean(),inplace=True)
data['NO3_1'].fillna(data['NO3_1'].mean(),inplace=True)
data['BOD5_1'].fillna(data['BOD5_1'].mean(),inplace=True)
data[['O2_2', 'O2_3', 'O2_4', 'O2_5', 'O2_6', 'O2_7',
       'NH4_2', 'NH4_3', 'NH4_4', 'NH4_5', 'NH4_6', 'NH4_7',
       'NO2_2', 'NO2_3', 'NO2_4', 'NO2_5', 'NO2_6', 'NO2_7','NO3_2',
       'NO3_3', 'NO3_4', 'NO3_5', 'NO3_6', 'NO3_7','BOD5_2',
       'BOD5_3', 'BOD5_4', 'BOD5_5', 'BOD5_6', 'BOD5_7']] =0
        
data.to_csv('riverwater.csv')
