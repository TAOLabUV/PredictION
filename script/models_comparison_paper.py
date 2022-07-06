#!/usr/bin/env python
# coding: utf-8

# In[1]:
# Import Libraries 
import os
os.chdir('C:\PredictION\scripts')
import pandas as pd
from pandas import DataFrame,Series
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns # plotting library
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm 
from lime import lime_tabular
from math import sqrt

import sklearn
from sklearn import tree, metrics, linear_model, svm, metrics, model_selection, preprocessing, feature_selection, ensemble, decomposition
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict, RepeatedKFold, KFold, GridSearchCV, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import r2_score, get_scorer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# load dataset
names = ['quantification_cDNA','coverage_depth','reads','coverage_genome']
dtf = pd.read_csv("dataset_1.csv", header=0, delimiter=";", names=names)
dtf2 = pd.read_csv("dataset_2.csv", header=0, delimiter=";")
dtf, dtf2

# In[2]:
x = "reads"
fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=(12, 6))
fig.suptitle(x, fontsize=20)
### Distribution for dataset 1 when x= "reads"
ax[0].title.set_text('distribution')
variable = dtf[x].fillna(dtf[x].mean())
breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
variable = variable[ (variable > breaks[0]) & (variable <  breaks[10]) ]
sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
des = dtf[x].describe()
ax[0].axvline(des["25%"], ls='--')
ax[0].axvline(des["mean"], ls='--')
ax[0].axvline(des["75%"], ls='--')
ax[0].grid(axis='x')
des = round(des, 2).apply(lambda x: str(x))
box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))
### boxplot for dataset 1 when x= "reads" 
ax[1].title.set_text('outliers (log scale)')
tmp_dtf = pd.DataFrame(dtf[x])
tmp_dtf[x] = np.log(tmp_dtf[x])
tmp_dtf.boxplot(column=x, ax=ax[1])
plt.show()

# In[3]:
x = "coverage_depth"
fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=(12, 6))
fig.suptitle(x, fontsize=20)
### distribution for dataset 1 when x = "coverage_depth" 
ax[0].title.set_text('distribution')
variable = dtf[x].fillna(dtf[x].mean())
breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
variable = variable[ (variable > breaks[0]) & (variable < 
                    breaks[10]) ]
sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
des = dtf[x].describe()
ax[0].axvline(des["25%"], ls='--')
ax[0].axvline(des["mean"], ls='--')
ax[0].axvline(des["75%"], ls='--')
ax[0].grid(axis='x')
des = round(des, 2).apply(lambda x: str(x))
box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))
### boxplot for dataset 1 when x = "coverage_depth" 
ax[1].title.set_text('outliers (log scale)')
tmp_dtf = pd.DataFrame(dtf[x])
tmp_dtf[x] = np.log(tmp_dtf[x])
tmp_dtf.boxplot(column=x, ax=ax[1])
plt.show()

# In[4]:
x = "reads"
fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=(12, 6))
fig.suptitle(x, fontsize=20)
### distribution for dataset 2 when x= "reads"
ax[0].title.set_text('distribution')
variable = dtf2[x].fillna(dtf2[x].mean())
breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
variable = variable[ (variable > breaks[0]) & (variable < 
                    breaks[10]) ]
sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
des = dtf2[x].describe()
ax[0].axvline(des["25%"], ls='--')
ax[0].axvline(des["mean"], ls='--')
ax[0].axvline(des["75%"], ls='--')
ax[0].grid(axis='x')
des = round(des, 2).apply(lambda x: str(x))
box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))
### boxplot for dataset 2 when x= "reads"
ax[1].title.set_text('outliers (log scale)')
tmp_dtf2 = pd.DataFrame(dtf2[x])
tmp_dtf2[x] = np.log(tmp_dtf2[x])
tmp_dtf2.boxplot(column=x, ax=ax[1])
plt.show()

# In[5]:
x = "coverage_depth"
fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=(12, 6))
fig.suptitle(x, fontsize=20)
### distribution for dataset 2 when x = "coverage_depth" 
ax[0].title.set_text('distribution')
variable = dtf2[x].fillna(dtf2[x].mean())
breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
variable = variable[ (variable > breaks[0]) & (variable < 
                    breaks[10]) ]
sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
des = dtf2[x].describe()
ax[0].axvline(des["25%"], ls='--')
ax[0].axvline(des["mean"], ls='--')
ax[0].axvline(des["75%"], ls='--')
ax[0].grid(axis='x')
des = round(des, 2).apply(lambda x: str(x))
box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))
### boxplot for dataset 2 when x = "coverage_depth" 
ax[1].title.set_text('outliers (log scale)')
tmp_dtf2 = pd.DataFrame(dtf2[x])
tmp_dtf2[x] = np.log(tmp_dtf2[x])
tmp_dtf2.boxplot(column=x, ax=ax[1])
plt.show()

# In[6]:
x, y = "reads", "coverage_depth"
### bin plot for dataset 1
figsize = (6, 6)
dtf_noNan = dtf[dtf[x].notnull()]
breaks = np.quantile(dtf_noNan[x], q=np.linspace(0, 1, 11))
groups = dtf_noNan.groupby([pd.cut(dtf_noNan[x], bins=breaks, 
           duplicates='drop')])[y].agg(['mean','median','size'])
fig, ax = plt.subplots(figsize=figsize)
fig.suptitle(x+"   vs   "+y, fontsize=20)
groups[["mean", "median"]].plot(kind="line", ax=ax)
groups["size"].plot(kind="bar", ax=ax, rot=45, secondary_y=True,
                    color="grey", alpha=0.3, grid=True)
ax.set(ylabel=y)
ax.right_ax.set_ylabel("Observations in each bin")
plt.show()
### scatter plot for dataset 1
sns.jointplot(x=x, y=y, data=dtf, dropna=True, kind='reg', 
              height=int((figsize[0]+figsize[1])/2) )
plt.show()

dtf_noNan = dtf[dtf[x].notnull()]
coeff, p = scipy.stats.pearsonr(dtf_noNan[x], dtf_noNan[y])
coeff, p = round(coeff, 3), round(p, 3)
conclusion = "Significant" if p < 0.05 else "Non-Significant"
print("Pearson Correlation:", coeff, conclusion, "(p-value: "+str(p)+")")


# In[7]:
x, y = "reads", "coverage_depth"
### bin plot for dataset 2
figsize = (6, 6)
dtf2_noNan = dtf2[dtf2[x].notnull()]
breaks = np.quantile(dtf2_noNan[x], q=np.linspace(0, 1, 11))
groups = dtf2_noNan.groupby([pd.cut(dtf2_noNan[x], bins=breaks, 
           duplicates='drop')])[y].agg(['mean','median','size'])
fig, ax = plt.subplots(figsize=figsize)
fig.suptitle(x+"   vs   "+y, fontsize=20)
groups[["mean", "median"]].plot(kind="line", ax=ax)
groups["size"].plot(kind="bar", ax=ax, rot=45, secondary_y=True,
                    color="grey", alpha=0.3, grid=True)
ax.set(ylabel=y)
ax.right_ax.set_ylabel("Observations in each bin")
plt.show()
### scatter plot for dataset 2
sns.jointplot(x=x, y=y, data=dtf2, dropna=True, kind='reg', 
              height=int((figsize[0]+figsize[1])/2) )
plt.show()

dtf2_noNan = dtf2[dtf2[x].notnull()]
coeff, p = scipy.stats.pearsonr(dtf2_noNan[x], dtf2_noNan[y])
coeff, p = round(coeff, 3), round(p, 3)
conclusion = "Significant" if p < 0.05 else "Non-Significant"
print("Pearson Correlation:", coeff, conclusion, "(p-value: "+str(p)+")")

# In[8]:
# Univariate analysis for dataset 1
fig = plt.figure(figsize=(16, 8))  
i = 0
for column in dtf:
    sub = fig.add_subplot(2,4 , i + 1)
    sub.set_xlabel(column)
    dtf[column].plot(kind = 'hist')
    i = i + 1

dtf.describe(), dtf2.describe()

# In[9]:
# Univariate analysis for dataset 2
fig = plt.figure(figsize=(16, 8))  
i = 0    
for column in dtf2:
    sub = fig.add_subplot(2,4 , i + 1)
    sub.set_xlabel(column)
    dtf2[column].plot(kind = 'hist')
    i = i + 1

# In[10]:
# Creating basic correlation plots with the data for dataset 1
sns.pairplot(dtf[['quantification_cDNA','coverage_depth','reads','coverage_genome']], diag_kind="kde")

# In[11]:
# Creating basic correlation plots with the data for dataset 2
sns.pairplot(dtf2[['quantification_cDNA','CT','coverage_depth','reads','coverage_genome']], diag_kind="kde")

# In[12]:
# Define seed for reproducibility
seed = 27

# Splitting data for dataset 1
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.25, random_state=seed)
print(dtf_train)
print(dtf_test)

# In[48]:
# Splitting data for dataset 2
dtf2_train, dtf2_test = model_selection.train_test_split(dtf2, test_size=0.25, random_state=seed)
print(dtf2_train)
print(dtf2_test)

# In[49]:
# Pearson Correlation Matrix for dataset 1
corr_matrix = dtf_train.corr(method="pearson")
sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("pearson correlation")

# In[50]:
# Pearson Correlation Matrix for dataset 2
corr_matrix2 = dtf2_train.corr(method="pearson")
sns.heatmap(corr_matrix2, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("pearson correlation")

# In[51]:
## Scalling for dataset 1
scaler = preprocessing.RobustScaler()
X_names = ['quantification_cDNA','coverage_depth','coverage_genome']
X = scaler.fit_transform(dtf_train[X_names])
dtf_scaled= pd.DataFrame(X, columns=dtf_train[X_names].columns, index=dtf_train[X_names].index)
print(dtf_scaled)
dtf_scaled['reads'] = scaler.fit_transform(dtf_train['reads'].values.reshape(-1,1))
print(dtf_scaled)

dtf_test_scaled = pd.DataFrame(scaler.fit_transform(dtf_test[X_names]), columns=dtf_test[X_names].columns, index=dtf_test[X_names].index)
print(dtf_test_scaled)
dtf_test_scaled['reads'] = scaler.fit_transform(dtf_test['reads'].values.reshape(-1,1))
print(dtf_test_scaled)

## Scalling for dataset 2
scalerX2 = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
X2 = scalerX2.fit_transform(dtf2_train.drop("reads", axis=1))
dtf2_scaled= pd.DataFrame(X2, columns=dtf2_train.drop("reads", axis=1).columns, index=dtf2_train.index)
print(dtf2_scaled.head())
# scale Y
scalerY2 = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
dtf2_scaled['reads'] = scalerY2.fit_transform(dtf2_train['reads'].values.reshape(-1,1))
print(dtf2_scaled.head())

# In[52]:

## Selected features for dataset 1
X_names = ['quantification_cDNA','coverage_depth','coverage_genome']
X_train = dtf_train[X_names].values
y_train = dtf_train["reads"].values
X_test = dtf_test[X_names].values
y_test = dtf_test["reads"].values

## Selected features for dataset 2
X_names2 = ['quantification_cDNA','CT','coverage_depth','coverage_genome']
X_train2 = dtf2_train[X_names2].values
y_train2 = dtf2_train["reads"].values
X_test2 = dtf2_test[X_names2].values
y_test2 = dtf2_test["reads"].values

# In[53]:
# Models
from sklearn.pipeline import make_pipeline
Lss_params = [{"alpha": 2.0970464013232393}]
GB_params =  [{"random_state": seed}]
RF_params =  [{"n_estimators": 1000 ,  "random_state": seed}]
SV_params = [{"random_state": seed, "max_iter": 100000}]

models = []
models.append(['Lasso', Lasso, Lss_params])
models.append(['GBR', GradientBoostingRegressor, GB_params])
models.append(['RFR', RandomForestRegressor, RF_params])
models.append(['SVR', LinearSVR, SV_params])
## for dataset 1
for modelname, Model, params_list in models:
    for params in params_list:
        # Fit model to the training set
        model = Model(**params)
        model.fit(X_train, y_train)    
        # Predict y_pred
        y_pred = model.predict(X_test)
        # Regressin performance metrics
        print('R2','%s: %s' %(Model.__name__, metrics.r2_score(y_test, y_pred)))
        print('MAE','%s: %s' %(Model.__name__, metrics.mean_absolute_error(y_test, y_pred)))
        print('MSE','%s: %s' %(Model.__name__, metrics.mean_squared_error(y_test, y_pred)))
        print('RMSE','%s: %s' %(Model.__name__, metrics.mean_squared_error(y_test, y_pred, squared=False)))
        print("Mean Absolute Perc Error (Σ(|y-pred|/y)/n):", '%s: %s' %(Model.__name__, round(np.mean(np.abs((y_test-y_pred)/y_pred)), 2)))

# In[54]:
## for dataset 2
for modelname, Model, params_list in models:
    for params in params_list:
        # Fit model to the training set
        model = Model(**params)
        model.fit(X_train2, y_train2)    
        # Predict y_pred
        y_pred2 = model.predict(X_test2)
        # Regressin performance metrics
        print('R2','%s: %s' %(Model.__name__, metrics.r2_score(y_test2, y_pred2)))
        print('MAE','%s: %s' %(Model.__name__, metrics.mean_absolute_error(y_test2, y_pred2)))
        print('MSE','%s: %s' %(Model.__name__, metrics.mean_squared_error(y_test2, y_pred2)))
        print('RMSE','%s: %s' %(Model.__name__, metrics.mean_squared_error(y_test2, y_pred2, squared=False)))
        print("Mean Absolute Perc Error (Σ(|y-pred|/y)/n):", '%s: %s' %(Model.__name__, round(np.mean(np.abs((y_test2-y_pred2)/y_pred2)), 2)))
   
# In[55]:

## K fold validation for dataset 1
scores = []
cv = model_selection.KFold(n_splits=5, shuffle=True, random_state = seed)
for modelname, Model, params_list in models:
    for params in params_list:
        fig = plt.figure(figsize=(6, 6))
        i = 1
        for train, test in cv.split(X_train, y_train):
            prediction = Model(**params).fit(X_train[train],y_train[train]).predict(X_train[test])
            true = y_train[test]
            score = metrics.r2_score(true, prediction)
            scores.append(score)
            plt.scatter(prediction, true, lw=2, alpha=0.3, label='Fold %d (R2 = %0.2f)' % (i,score))
            i = i+1
        plt.plot([min(y_train),max(y_train)], [min(y_train),max(y_train)], linestyle='--', lw=2, color='black')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('%s: %s' %(Model.__name__,'K-Fold Validation'))
        plt.legend()
        plt.show()

# In[56]:
## K fold validation for dataset 2
scores = []
cv = model_selection.KFold(n_splits=5, shuffle=True, random_state = seed)
for modelname, Model, params_list in models:
    for params in params_list:
        fig = plt.figure(figsize=(6, 6))
        i = 1
        for train, test in cv.split(X_train2, y_train2):
            prediction = Model(**params).fit(X_train2[train],y_train2[train]).predict(X_train2[test])
            true = y_train2[test]
            score = metrics.r2_score(true, prediction)
            scores.append(score)
            plt.scatter(prediction, true, lw=2, alpha=0.3, label='Fold %d (R2 = %0.2f)' % (i,score))
            i = i+1
        plt.plot([min(y_train2),max(y_train2)], [min(y_train2),max(y_train2)], linestyle='--', lw=2, color='black')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('%s: %s' %(Model.__name__,'K-Fold Validation'))
        plt.legend()
        plt.show()

# In[57]:
## K fold validation for both dataset 1 and 2
fig = plt.figure(figsize=(12, 6))
for modelname, Model, params_list in models:
        for params in params_list:
            prediction = Model(**params).fit(X_train,y_train).predict(X_test)
            true = y_test
            score = metrics.r2_score(true, prediction)
            scores.append(score)
            plt.scatter(prediction, true, lw=2, alpha=0.3, label= '%s: %s' %(modelname, round(score, 4)))
plt.plot([min(y_test),max(y_test)], [min(y_test),max(y_test)], linestyle='--', lw=2, color='black')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.legend()
plt.show()

fig = plt.figure(figsize=(12, 6))
for modelname, Model, params_list in models:
        for params in params_list:
            prediction = Model(**params).fit(X_train2,y_train2).predict(X_test2)
            true = y_test2
            score = metrics.r2_score(true, prediction)
            scores.append(score)
            plt.scatter(prediction, true, lw=2, alpha=0.3, label= '%s: %s' %(modelname, round(score, 4)))
plt.plot([min(y_test2),max(y_test2)], [min(y_test2),max(y_test2)], linestyle='--', lw=2, color='black')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.legend()
plt.show()

# In[58]:
import matplotlib.colors as mcolors
## Residuals and error plots for dataset 1
from statsmodels.graphics.api import abline_plot

for modelname, Model, params_list in models:
    for params in params_list:
        # Fit model to the training set
        model = Model(**params)
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        ## residuals ############
        residuals = y_test - y_pred
        max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
        max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
        max_true = y_test[max_idx]
        max_pred = y_pred[max_idx]
        print(Model.__name__, "Max Error:", "{:,.0f}".format(max_error))
        
        ## Plot predicted vs true ##############
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax[0].scatter(y_pred, y_test, color="black")
        abline_plot(intercept=0, slope=1, color="red", ax=ax[0])
        ax[0].vlines(x=max_pred, ymin=max_true, ymax=max_true-max_error, color='red', linestyle='--', alpha=0.7, label="max error")
        ax[0].grid(True)
        ax[0].set(xlabel="Predicted", ylabel="True", title= "Predicted vs True")
        ax[0].legend()
        
        ## Plot predicted vs residuals
        ax[1].scatter(y_pred, residuals, color="red")
        ax[1].vlines(x=max_pred, ymin=0, ymax=max_error, color='black', linestyle='--', alpha=0.7, label="max error")
        ax[1].grid(True)
        ax[1].set(xlabel="Predicted", ylabel="Residuals", title= "Predicted vs Residuals")
        ax[1].hlines(y=0, xmin=np.min(y_pred), xmax=np.max(y_pred))
        ax[1].legend()
        fig.suptitle(Model.__name__, fontsize=13)
        plt.subplots_adjust(wspace=0.2)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.distplot(residuals, color="red", hist=True, kde=True, kde_kws={"shade":True}, ax=ax)
        ax.grid(axis='x')
        ax.set(title= '%s: %s' %(Model.__name__, "Residuals distribution"))
        plt.show()

# In[59]:
## Residuals and error plots for dataset 2
from statsmodels.graphics.api import abline_plot

for modelname, Model, params_list in models:
    for params in params_list:
        # Fit model to the training set
        model = Model(**params)
        model.fit(X_train2, y_train2) 
        y_pred = model.predict(X_test2)
        ## residuals ############
        residuals = y_test2 - y_pred
        max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
        max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
        max_true = y_test2[max_idx]
        max_pred = y_pred[max_idx]
        print(Model.__name__, "Max Error:", "{:,.0f}".format(max_error))
        
        ## Plot predicted vs true ##############
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax[0].scatter(y_pred, y_test2, color="black")
        abline_plot(intercept=0, slope=1, color="red", ax=ax[0])
        ax[0].vlines(x=max_pred, ymin=max_true, ymax=max_true-max_error, color='red', linestyle='--', alpha=0.7, label="max error")
        ax[0].grid(True)
        ax[0].set(xlabel="Predicted", ylabel="True", title= "Predicted vs True")
        ax[0].legend()
        
        ## Plot predicted vs residuals
        ax[1].scatter(y_pred, residuals, color="red")
        ax[1].vlines(x=max_pred, ymin=0, ymax=max_error, color='black', linestyle='--', alpha=0.7, label="max error")
        ax[1].grid(True)
        ax[1].set(xlabel="Predicted", ylabel="Residuals", title= "Predicted vs Residuals")
        ax[1].hlines(y=0, xmin=np.min(y_pred), xmax=np.max(y_pred))
        ax[1].legend()
        fig.suptitle(Model.__name__, fontsize=13)
        plt.subplots_adjust(wspace=0.2)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.distplot(residuals, color="red", hist=True, kde=True, kde_kws={"shade":True}, ax=ax)
        ax.grid(axis='x')
        ax.set(title= '%s: %s' %(Model.__name__, "Residuals distribution"))
        plt.show()

# In[60]:
## Explainibility for dataset 1
for modelname, Model, params_list in models:
    for params in params_list:
        # Fit model to the training set
        model = Model(**params)
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        print(Model.__name__,"True:", "{:,.0f}".format(y_test[1]), "--> Pred:", "{:,.0f}".format(y_pred[1]))
        
        explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names="reads", mode="regression")
        explained = explainer.explain_instance(X_test[1], model.predict, num_features=3)
        explained.as_pyplot_figure()
        plt.show()

# In[61]:
## Explainibility for dataset 2
for modelname, Model, params_list in models:
    for params in params_list:
        # Fit model to the training set
        model = Model(**params)
        model.fit(X_train2, y_train2) 
        y_pred = model.predict(X_test2)
        print(Model.__name__,"True:", "{:,.0f}".format(y_test2[1]), "--> Pred:", "{:,.0f}".format(y_pred[1]))
        
        explainer = lime_tabular.LimeTabularExplainer(training_data=X_train2, feature_names=X_names2, class_names="reads", mode="regression")
        explained = explainer.explain_instance(X_test2[1], model.predict, num_features=4)
        explained.as_pyplot_figure()
        plt.show()

