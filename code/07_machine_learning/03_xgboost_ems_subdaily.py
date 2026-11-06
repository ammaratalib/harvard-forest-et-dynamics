# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 18:53:43 2024

@author: ammar
"""

# -*- coding: utf-8 -*-


"""
Spyder Editor

This is a temporary script file.
"""



#############################################################################################################

#import warnings
#if not sys.warnoptions:
 #   warnings.simplefilter("ignore")

def stat (obs,pre):
    nse=NSE(pre,obs)
    pear=pearsonr(obs,pre)
    #wil=will(pre,obs)   
    MAE=mean_absolute_error(obs,pre)
    RMSE=sqrt(mean_squared_error(obs,pre))
#    reg_=reg.score(x1, y1)
    pbias=(np.sum(pre-obs)/np.sum(obs))*100
    return nse,pear,MAE,reg_,pbias
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_predict
import os
from xgboost import XGBRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# evaluate xgboost random forest ensemble for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRFRegressor
## start with HEM data 
from numpy import asarray
from pandas import read_csv
# evaluate an xgboost regression model on the housing dataset
from numpy import absolute
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from matplotlib.legend import Legend
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
# 2009-2023 is good.
# transformation is not an issue. error persist after
##################################################################################################
##################################################################################################
import numpy as np
import pandas as pd
from time import time
import pprint
import joblib
from functools import partial

# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")

# Classifier/Regressor
from xgboost import XGBRegressor, DMatrix

# Model selection
from sklearn.model_selection import KFold, StratifiedKFold

# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer

# Data processing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

###############################################################################################3



import numpy as np
import pandas as pd
from time import time
import pprint
import joblib
from functools import partial

# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")

# Classifier/Regressor
from xgboost import XGBRegressor, DMatrix

# Model selection
from sklearn.model_selection import KFold, StratifiedKFold

# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer

# Data processing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def Grow_sea(df):
    
    if (df['DoY'] >=df['sos']) and (df['DoY'] <=df['eos']):
        return 'GS'
    else:
        return 'NGS'
    
###############################################################################################3
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")
df_ML_ems=pd.read_csv("EMS_xg_Rd_fill.csv") 
df=df_ML_ems
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df["Month"]=df.TIMESTAMP.dt.month

##############################################################################################
# dont use it. this part is to convert hourly ton half hourly 
#df['datetime'] = pd.to_datetime(df['TIMESTAMP'])

#df = df.drop('TIMESTAMP', axis=1)
#df = df.drop('TIMESTAMP.1', axis=1)

#df.set_index('datetime', inplace=True)

#df=df.resample('30T').interpolate()
#df=df.reset_index()


#df.to_csv('agg.csv', index=False, header=True)

###########################################################################################
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)

mask = (df['Grow_sea'] == 'NGS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
   
back=df

# starts at 10/1/1991


#df['Month_sin'] = np.sin((df.Month-1)*(2.*np.pi/12))
df['Month_cos'] = np.cos((df.Month-1)*(2.*np.pi/12))
#df['Hour_sin'] = np.sin(df.Hour*(2.*np.pi/24))
df['Hour_cos'] = np.cos(df.Hour*(2.*np.pi/24))
df["lambda"]=(3149000-2370*(df['Tair_fill']+273.16))*1e-6
df["WD_gaps_1"]=np.sin(df.WD_gaps*(2.*np.pi/24))
df["WD_gaps_2"]=np.cos(df.WD_gaps*(2.*np.pi/24))
back=df
df=back

mask = (df['TIMESTAMP'] >='1991-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')

df = df.loc[mask]
df.index = np.arange(0, len(df))

back_al=df
df_no_gap=df[df['LE_Rd'].notnull()]

df_no_gap.index = np.arange(0, len(df_no_gap))

df=df_no_gap
# save year from df_no_gaps

df_x_y=pd.concat((df['DoY'], df['Year'],df['NETRAD_fill'],df['VPD_fill'],df['Tair_fill'],
df["PAR_fill"],df["WD_gaps"],df["WS_gaps"],df["PA_Rd"],df["H_Rd"],df["Rg_fill"],df["Ustar_gaps"],
df["GPP_mat"],df['gee.e.6mol.m2.s'],df["sos"],df["eos"],
df['Month_cos'],df['Hour_cos'],df["tsoil_upper_mean_gaps"],df['LE_Rd']),axis=1)
#corr=df_x_y.corr()

# use NEE with gaps remove Co2 and H20 when predict LE

#original 282720

#NETRAD: 10.078 % missing # split at 177960
#NEE: 56.5% missing split 86022
#LE: 57% missing split 85243
#VPD: 9.3% missing split 179240
#Tair: 0.76% missing split 196393
#PAR: 0.76% is missing split 196393
#WD: 6.5% is missing split 185003
#WS 7.05% data is missing split 183951
#RH 6.98% data is missing split 184091
#PA 6.69% data is missing split 184646
#H 24.7% data is missing split 148495
#Rg 0.235% data is missing split 197440
# Co2_1_1_1 33.87# data is missing split 130862
# H20_1_1_1 24.2# data is missing split 149941
# TAU 25.18# data is missing split 148067
#Ustar 25% data missing  148443
#TS_PI_1_1_A 54% is missing  split 107582
#tsoil_upper_mean 25% is missing  split 150191

# growing season:  77584
# non growing season 113152
###################################################################################################

## initial time series training
#data = df_no_gap.values.astype('float64')

data=df_x_y.values.astype('float64')
values=np.round(data,3)

n_train_hours = 113152

train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

###################################################################################################
train_back=train
test_back=test

train_X, train_y=train[:, :-1], train[:, -1]
test_X, test_y=test[:, :-1], test[:, -1]


train_y=train_y.reshape(train_y.shape[0],1)
test_y=test_y.reshape(test_y.shape[0],1)

scalerX = StandardScaler().fit(train_X)
scalery = StandardScaler().fit(train_y)
train_X = scalerX.transform(train_X)
train_y = scalery.transform(train_y)
test_X = scalerX.transform(test_X)
test_y = scalery.transform(test_y)

##################################################################################################


# go ahead
b_X_train=train_X
b_X_test=test_X

b_y_train=train_y
b_y_test=test_y

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

####################################################################################################


param_grid = {
    'n_estimators': [250,350,500],
    'max_depth': [6,8,12],
    'learning_rate': [0.02,0.05,0.09, 0.15],
    'min_child_weight': [1,2,3],
    'subsample': [0.3,0.5, 0.8, 1]
    }

# old method 
##################################################################################################3
#grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=3)


#est=grid_search.fit(train_X, train_y)


#best_params = grid_search.best_params_

#0.05, 1,0.7
#####################################################################################################
#https://www.kaggle.com/code/lucamassaron/tutorial-bayesian-optimization-with-xgboost

best_params= [('colsample_bytree', 0.5778677484569327),
             ('learning_rate', 0.6922755334330679),
             ('max_depth', 10),
             ('min_child_weight', 2.622051770179463),
             ('n_estimators', 880),
             ('reg_alpha', 98.05562626781331),
             ('reg_lambda', 12.806865632291673),
             ('subsample', 0.30114587249385294)]

####################################################################################################



search_spaces = {'learning_rate': Real(0.05, 1.0, 'uniform'),
                 'max_depth': Integer(3, 15),
                 'subsample': Real(0.3, 1.0, 'uniform'),
                 'colsample_bytree': Real(0.1, 1.0, 'uniform'), # subsample ratio of columns by tree
                 'reg_lambda': Real(1e-9, 100., 'uniform'), # L2 regularization
                 'reg_alpha': Real(1e-9, 100., 'uniform'), # L1 regularization
                 'min_child_weight': Real (0.1,4, 'uniform'),
                 'n_estimators': Integer(150, 1000)}


#best_params = report_perf(opt, train_X, train_y,'XGBoost_regression', 
 #                         callbacks=[overdone_control, time_limit_control])


# after calibration 

##################################################################################################
#xgb_model = XGBRegressor(**best_params)

#xgb_model = XGBRegressor(random_state=0, booster='gbtree', objective='reg:squarederror', tree_method='gpu_hist', **best_params)



xgb_model = XGBRegressor(random_state=0, booster='gbtree', objective='reg:squarederror', tree_method='gpu_hist', colsample_bytree=0.577, learning_rate=0.69227, max_depth=10, min_child_weight=2.62205,
n_estimators=880,reg_alpha= 98.0556,reg_lambda=12.807,subsample=0.3011)




est=xgb_model.fit(train_X, train_y)

predictions = xgb_model.predict(test_X)

mae = mean_absolute_error(test_y, predictions)

mae #0.148, 0.17, 0.19

rmse = np.sqrt(mean_squared_error(test_y, predictions))
rmse #0.198, 0.27, 0.37


#####################################################################################################

data = df_x_y
values=data

values.index = np.arange(0, len(values))
train  = values.loc[: 113151]  # one less than the previous 
train.index = np.arange(0, len(train))

test=values.loc[113152:]
test.index = np.arange(0, len(test)) #use the next one 

train_back=train
test_back=test
###########################################################################################



feature_importance = est.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(test_back.columns)[sorted_idx])
plt.title('Feature Importance')

a=test_back.columns
b=feature_importance

################################################################################################
#https://mljar.com/blog/feature-importance-xgboost/
## permutation method
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(xgb_model, train_X, train_y)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(test_back.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
############################################################################################
yhat_train = est.predict(train_X)

yhat_train = yhat_train.reshape(yhat_train.shape[0],1)
inv_yhat_train = scalery.inverse_transform(yhat_train)


inv_y_train = scalery.inverse_transform(train_y)

train_back["obs"]=inv_y_train
train_back["xg_model"]=inv_yhat_train


#yhat_test=est.predict(sm.add_constant(test_X))
yhat_test=predictions
yhat_test= yhat_test.reshape(yhat_test.shape[0],1)
inv_yhat_test = scalery.inverse_transform(yhat_test)
inv_y_test = scalery.inverse_transform(test_y)
test_back["obs"]=inv_y_test
test_back["xg_model"]=inv_yhat_test

plt.plot(train_back["obs"])
plt.plot(train_back["xg_model"])

plt.plot(test_back["obs"])
plt.plot(test_back["xg_model"])


os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")
train_back.to_csv('ems_ngs_hr_train.csv', index=False, header=True)
test_back.to_csv('ems_ngs_hr_test.csv', index=False, header=True)
df_no_gap.to_csv('df_no_gap.csv', index=False, header=True)







#df=train_back
df=test_back

obs=df["obs"]
pre=df["xg_model"]


x1=obs
y1=pre
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)



reg = LinearRegression(fit_intercept=True).fit(x1, y1)
#a=reg.coef_
#reg.coef_
reg.score(x1, y1)

stat(obs,pre)


#plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(df["obs"],df["xg_model"],'x', color="green",markersize=5,marker='^',mfc='none')
#ax.plot(x1,0.9644*x1+9.0488,'--', linewidth=2, color='black')#overall trend line 
line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
ax.set_title("XG Prediction Model, Testing Hourly: US-Ha1",fontsize=8)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': '9'})
#ax.set_xlabel('Observed NETRAD (Wm$^{-2}$)')  # we already handled the x-label with ax1
#ax.set_ylabel('Predicted NETRAD (Wm$^{-2}$)')  # we already handled the x-label with ax1
ax.set_xlabel('Observed')  # we already handled the x-label with ax1
ax.set_ylabel('Predicted')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
#ax.set_ylim([-2.5,2.5])
#ax.set_xlim([-2.5,2.5])

#R2	   NSE	   will	   pear  	MAE  	RMSE	pbias
#0.91	0.90	0.97	0.95	0.113	0.160	-1.30

#nse,pear,MAE,reg_,pbias

ax.text(0.95, 0.35, 'NSE = 0.56',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=8)

ax.text(0.95, 0.29, 'Pear = 0.77',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=8)

ax.text(0.95, 0.24, 'MAE = `13',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=8)

ax.text(0.95, 0.19, '$\mathregular{R^{2}}$ = 0.59',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=8)


ax.text(0.95, 0.14, 'pbias = 3.8%',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=8)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

###############################################################################################3

# prediction on gaps

df=back_al

df_x_y=pd.concat((df['DoY'], df['Year'],df['NETRAD_fill'],df['VPD_fill'],df['Tair_fill'],df["PAR_fill"],
df["WD_gaps"],df["WS_gaps"],df["PA_Rd"],df["H_Rd"],df["Rg_fill"],df["TAU_gaps"],df["Ustar_gaps"],
df["nee.e.6mol.m2.s"],df['Resp.e.e.6mol.m2.s'],df['gee.e.6mol.m2.s'],df["sos"],df["eos"],
df['Month_cos'],df['Hour_cos'],df["tsoil_upper_mean_gaps"],df['LE_Rd']),axis=1)

df_gap=df_x_y

data = df_gap.values.astype('float64')

values=np.round(data,3)


gap_X, gap_y=values[:, :-1], values[:, -1]

gap_y=gap_y.reshape(gap_y.shape[0],1)

gap_X = scalerX.transform(gap_X)
gap_y = scalery.transform(gap_y)

predictions = xgb_model.predict(gap_X)

yhat_gap=predictions
yhat_gap= yhat_gap.reshape(yhat_gap.shape[0],1)
inv_yhat_gap = scalery.inverse_transform(yhat_gap)


inv_y_gap = scalery.inverse_transform(gap_y)
df_gap["obs"]=inv_y_gap
df_gap["xg_model"]=inv_yhat_gap



###########################################################################################
df_gap['LE_ID'] = df_gap['LE_Rd'].isnull()
def xg_fill(df):
    
    if (df['LE_ID']==True):
        return df["xg_model"]
  
    else:

        return df["LE_Rd"]


df_gap['LE_fill'] = df_gap.apply(xg_fill, axis = 1)

df_gap.to_csv('gaps_ML_fill.csv', index=False, header=True)
#####################################################################################################

df=fill_ems # make sure to run TIMESTAMP
mask = (df['TIMESTAMP'] >='1991-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))

reddy=df


df=df_gap
df["TIMESTAMP"]=back_al["TIMESTAMP"]
mask = (df['TIMESTAMP'] >='1991-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))
back=df
df["TIMESTAMP"]=pd.to_datetime(df.TIMESTAMP)
df["LE_Rd"]=reddy["LE"]


# bring reddy proc gap filled data

#df_gap["TIMESTAMP"]=back_al["TIMESTAMP"]

df.to_csv('gaps_ML_fill_.csv', index=False, header=True)

###################################################################################################

# comapre xgboost gap filled 


####################################################################################################

def Grow_sea(df):
    
    if (df['DoY'] >=df['sos']) and (df['DoY'] <=df['eos']):
        return 'GS'
    else:
        return 'NGS'
    

def xg_fill(df):
    
    if (df['LE_ID2']==True):
        return df["LE_fill_bound"]
  
    else:

        return df["LE_Rd"]

    

#check results 
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")
df=pd.read_csv("EMS_xg_Rd_fill.csv")
back=df

df=pd.concat((df['TIMESTAMP'], df['DoY'],df['Year'],df['sos'],df["eos"],
df['LE_gaps'],df["LE_Rd"],df["LE_xg_Rd"],df["LE_xg_Rd2"],df["NETRAD_fill"],df["PAR_fill"],df["WS_gaps"],df["RH_fill"],df['Tair_fill'],
df["PA_gaps"],df["VPD_fill"]),axis=1)


#df=pd.concat((df['TIMESTAMP'], df['DoY'],df['Year'],df['sos'],df["eos"],df["LE_ID"],
#df['LE_gaps'],df["LE_Rd"],df["LE_fill"],df["LE_fill_bound"],df["NETRAD_fill"],df["PAR_fill"],df["WS_gaps"],df["RH_fill"],df['Tair_fill'],
#df["PA_gaps"],df["VPD_fill"]),axis=1)

df["TIMESTAMP"] = pd.DatetimeIndex(df.TIMESTAMP)
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)

mask = (df['TIMESTAMP'] >='1992-01-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))
back=df

#####################################################################################################
# mask for growing and non growing season

seas=df

mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
gw=df

df=seas
mask = (df['Grow_sea'] == 'NGS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
ngw=df



####################################################################################################
#df=ngw


df['lambda']=(3149000-2370*(df['Tair_fill']+273.16))*1e-6
df['ET_gaps']=(df['LE_gaps']/df['lambda'])*(1/1e6)*3600
df['ET_Rd']=(df['LE_Rd']/df['lambda'])*(1/1e6)*3600
#df['ET_xg']=(df['LE_fill']/df['lambda'])*(1/1e6)*3600
#df['ET_xg_bound']=(df['LE_fill_bound']/df['lambda'])*(1/1e6)*3600
df['ET_xg_Rd']=(df['LE_xg_Rd']/df['lambda'])*(1/1e6)*3600
df['ET_xg_Rd2']=(df['LE_xg_Rd2']/df['lambda'])*(1/1e6)*3600
compare=df


df=compare
mask = (df['TIMESTAMP'] >='1992-01-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))
back=df
df["TIMESTAMP"]=pd.to_datetime(df.TIMESTAMP)
back=df
df=back

df=df.groupby([df['TIMESTAMP'].dt.date]).sum() # change sum for ET
df=df.reset_index()
daily_ET=df


df["Year"]= pd.to_datetime(df['TIMESTAMP']).dt.year

annual= df.groupby(['Year']).sum()
df=annual
df=df.reset_index()

###############################################################################################
df["diff"]=df['ET_xg']-df["ET_Rd"]


fig, ax = plt.subplots()
ax.plot(df["Year"],df["diff"])
for x, y in zip(df["Year"],df["diff"]):
    ax.text(x, y, f"{x}", fontsize=8)

plt.plot(df["Year"], df["diff"], marker='.', linestyle='--')
plt.legend()
plt.ylabel("Annual diff (xgRd-Rd)")



####################################################################################################
# use hourly data
##count
k=df.LE_gaps.isnull().groupby([df['Year']]).sum().astype(int).reset_index(name='count')

fig, ax = plt.subplots()
ax.plot(df["PAR_fill"],df["diff"])

for x, y in zip(df["PAR_fill"],df["diff"]):
    ax.text(x, y, f"{x}", fontsize=8)

#plt.plot(k["Year"], k["count"], marker='.', linestyle='--', label="gaps")
#plt.legend()
#plt.ylabel("number of gaps")


plt.scatter(df["PAR_fill"],df["diff"], marker='.', linestyle='--', label="PAR vs diff")
plt.legend()
plt.ylabel("diff (mm)")
#plt.xlabel("num of gaps")



#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(df['Year'], df['ET_gaps'],color="Red",label="ET (Gaps)",linestyle='--', marker='D',markersize=4)
plt.plot(df['Year'], df['ET_Rd'],color="blue",label="ET (REddyProc)",linestyle='--', marker='D',markersize=4)
plt.plot(df['Year'], df['ET_xg_Rd'],color="pink",label="ET (XGboost, Rd)",linestyle='--', marker='o',markersize=4)
plt.plot(df['Year'], df['ET_xg_Rd2'],color="yellow",label="ET (XGboost, Rd2)",linestyle='--', marker='o',markersize=4)


ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '6')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(8) 
ax.set_ylabel('Annual ET (mm)')  # we already handled the x-label with ax1
ax.set_title("Harvard Forest EMS:US-Ha1, Deciduous Broadleaf Forest",fontsize=10)
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#################################################################################################











###################################################################################################
plt.plot(df_gap["obs"])
plt.plot(df_gap["xg_model"])



#df=train_back
df=df_gap

obs=df["obs"]
pre=df["xg_model"]


x1=obs
y1=pre
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)



reg = LinearRegression(fit_intercept=True).fit(x1, y1)
a=reg.coef_
#reg.coef_
reg.score(x1, y1)

stat(obs,pre)







def stat (obs,pre):
    nse=NSE(pre,obs)
    pear=pearsonr(obs,pre)
    #wil=will(pre,obs)   
    MAE=mean_absolute_error(obs,pre)
    #RMSE=sqrt(mean_squared_error(obs,pre))
    reg_=reg.score(x1, y1)
    pbias=(np.sum(pre-obs)/np.sum(obs))*100
    return nse,pear,MAE,reg_,pbias


#0.2, 5, 0.8 





param_grid = {
    'learning_rate': [0.05, 0.1, 0.2,0.3],

    'max_depth': [3, 4,5,6],

    'subsample': [0.7,0.8, 0.9, 1.0],
}

param_grid = {
    'n_estimators': [200,400],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.05, 0.1, 0.20],
    'min_child_weight': [1, 10, 20,50],
    'subsample': [0.6, 0.8]
    }

# find x and y columns.

#############################################################################################################
#os.chdir(r"C:\ammara_MD\a_harvard\Project\data\machine_learning")

os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")
df=pd.read_csv("daily_EMS_ML.csv")
mask = (df['TIMESTAMP'] >='1991-06-10') & (df['TIMESTAMP'] <= '2023-12-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]=pd.to_datetime(df.TIMESTAMP)
back_ml=df


df=pd.read_csv("gs_FG_all.csv") # without precip days
df=pd.concat((df["TIMESTAMP"],df["gs_FG"]),axis=1)
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df=df.groupby([df['TIMESTAMP'].dt.date]).mean()
df=df.rename_axis('TIMESTAMP').reset_index()
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df["Year"]=df.TIMESTAMP.dt.year
df["month"]=df.TIMESTAMP.dt.month
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
mask = (df['TIMESTAMP'] >='1992-06-10 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]=pd.to_datetime(df.TIMESTAMP)
gs=df
df=gs
df= df.groupby(['Year','month']).mean()
df=df.reset_index()
gs_m=df


data=back_ml
data['water_year'] = data.TIMESTAMP.dt.year.where(data.TIMESTAMP.dt.month < 10, data.TIMESTAMP.dt.year + 1)
data["TIMESTAMP"]=pd.to_datetime(data["TIMESTAMP"])
data["Year"]=data.TIMESTAMP.dt.year
mask = (data['TIMESTAMP'] >='1992-06-01 00:00:00') & (data['TIMESTAMP'] <= '2023-12-31 23:45:00')
data = data.loc[mask]
data.index = np.arange(0, len(data))
data["TIMESTAMP"]=pd.to_datetime(data.TIMESTAMP)
data['month']=data['TIMESTAMP'].dt.month
com= data.merge(gs_m, on=["Year","month"], how="outer")
df=com
df=df.reset_index()
back_var=df

phen_month=df # bring this from phenology data

com= data.merge(gs_m, on=["Year","month"], how="outer")
df=com
df=df.reset_index()
back_var=df




df.isnull().values.any()

df.columns = ['TIMESTAMP', 'DoY', 'Year', 'Hour', 'FC', 'CO2', 'TAU',
       'USTAR', 'SC', 'TS', 'Ustar', 'tsoil_upper_mean', 'NEE',
       'H', 'RH', 'WS', 'PA', 'Tair', 'lambda', 'VPD', 'GPP',
       'Rg', 'PAR', 'NETRAD', 'LE', 'ET', 'water_year']          




df_x=pd.concat((df['DoY'],df["water_year"],df['FC'],df['CO2'], df['TAU'], df['SC'],
df['TS'],df["Ustar"],df["tsoil_upper_mean"],df['NEE'],df['H'],df['RH'],df['WS'],df['PA'],
df['Tair'],df['lambda'],df['VPD'],df['GPP'],df['Rg'],df['PAR'],df['NETRAD'],df['LE'],df['ET']),axis=1)


corr=df_x.corr()



plt.figure(figsize=(16, 6))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(df_x.corr(), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':16}, pad=12);


#df[df.index.year != 2019]



mask = (df['TIMESTAMP'] >='1992-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))
back=df
df["TIMESTAMP"]=pd.to_datetime(df.TIMESTAMP)
back=df
df=back

df=data_prep

# XGboost and RF
##########################################################################################################
# test regression dataset
from sklearn.datasets import make_regression
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=7)
# summarize the dataset
print(X.shape, y.shape)

# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=7)
# define the model
model = XGBRFRegressor(n_estimators=100, subsample=0.9, colsample_bynode=0.2)
# define the model evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# gradient xgboost random forest for making predictions for regression
from numpy import asarray
from sklearn.datasets import make_regression
from xgboost import XGBRFRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=7)
# define the model
model = XGBRFRegressor(n_estimators=100, subsample=0.9, colsample_bynode=0.2)
# fit the model on the whole dataset
model.fit(X, y)
# define a single row of data
row = [0.20543991,-0.97049844,-0.81403429,-0.23842689,-0.60704084,-0.48541492,0.53113006,2.01834338,-0.90745243,-1.85859731,-1.02334791,-0.6877744,0.60984819,-0.70630121,-1.29161497,1.32385441,1.42150747,1.26567231,2.56569098,-0.11154792]
row = asarray([row])
# make a prediction
yhat = model.predict(row)
# summarize the prediction
print('Prediction: %d' % yhat[0])

#############################################################################################################

#only XGboost
###########################################################################################################
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
# split dataset into input and output columns
X, y = data[:, :-1], data[:, -1]
# define model
model = XGBRegressor()
# fit model
model.fit(X, y)



row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
new_data = asarray([row])
# make a prediction
yhat = model.predict(new_data)
# summarize prediction
print('Predicted: %.3f' % yhat)






















######################################################################################################3####

### nature paper

########################################################################################
#https://github.com/hvekuri/co2_gapfilling/blob/main/CO2_gapfill.py
#henry code 




#XG boost
#https://www.datacamp.com/tutorial/xgboost-in-python


#https://www.esrl.noaa.gov/psd/forecasts/reforecast2/download.html
# for forecasting netcdf 3 size is big. use netcdf 4

# prediction model  26331, 16

import pandas as pd 
import numpy as np 
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_predict
import os

## start with HEM data 

os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")
df=pd.read_csv("back_up_EMS.csv")

df=back_m

mask = (df['TIMESTAMP'] >='1992-06-08 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))


back=df

df["TIMESTAMP"]=pd.to_datetime(df.TIMESTAMP)

back=df

df=back

data_prep=pd.concat((df["TIMESTAMP"],df['doy'],df["year"], df['Hour'],df['WS'],df['WD'],
df['LE_1_1_1'],df['FC_1_1_1'],df['CO2_1_1_1'], df['TAU'],df['USTAR'], df['SC'], df['H'], df['NETRAD'],
 df['PPFD'],df['TA_1_1_1'], df['TS_PI_1_1_A'],df['PA'], df["NEE"],df["Tair"],df["lambda"], 
 df["ET_mm"],df["RH"],df["VPD"],df["Ustar"], df["doy"],df["Rg"],df["tsoil_upper_mean"]),axis=1)


df=data_prep

df.set_index('TIMESTAMP', inplace=True)

def add_time_vars(data):
    """
    Adds time variables to data
    """
    data['Month'] = data.index.month
    data['Hour'] = data.index.hour+data.index.minute/60
    data['Month_sin'] = np.sin((data.Month-1)*(2.*np.pi/12))
    data['Month_cos'] = np.cos((data.Month-1)*(2.*np.pi/12))
    data['Hour_sin'] = np.sin(data.Hour*(2.*np.pi/24))
    data['Hour_cos'] = np.cos(data.Hour*(2.*np.pi/24))
    data['Time'] = np.arange(0, len(data), 1)
    return data



df=add_time_vars(df)
df=df.reset_index()


x_cols=['doy', 'year', 'Hour', 'WS', 'WD', 'FC_1_1_1', 'CO2_1_1_1',
       'TAU', 'USTAR', 'SC', 'H', 'NETRAD', 'PPFD', 'TA_1_1_1', 'TS_PI_1_1_A',
       'PA', 'NEE', 'Tair', 'lambda', 'RH', 'VPD', 'Ustar',
       'Rg', 'tsoil_upper_mean', 'Month', 'Month_sin', 'Month_cos', 'Hour_sin',
       'Hour_cos', 'Time']

y_col=['LE_1_1_1']



# find x and y columns.

data=df
def org_data(data, x_cols, y_col):
    """
    Shuffles data, returns X, y and their indices
    """
    #data = data[data[y_col].notnull()]
    X = np.asarray(data[x_cols])
    y = np.asarray(data[y_col])
    X, y, id = shuffle(X, y, data.index)

    return X, y, id


test=org_data(df, x_cols, y_col)




def optimize_hyperparameters(data, x_cols, y_col):
    """
    Optimizes hyperparameters using 5-fold cross validation and grid search
    Returns best parameters
    """
    X, y, _ = org_data(data, x_cols, y_col)

    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror')

    params = {
        "subsample" : [0.5, 0.75, 1],
        "max_depth" :  [3, 5, 10, 15],
        "min_child_weight" : [2, 5, 10],
        "colsample_bytree" : [0.4, 0.6, 0.8, 1],
    }

    xgb_grid = GridSearchCV(xgb_reg,
                            params,
                            cv=5,
                            n_jobs=-1,
                            verbose=False)

    xgb_grid.fit(X, y)

    return xgb_grid.best_params_





optim=optimize_hyperparameters(data, x_cols, y_col)





def cv_preds(data, x_cols, y_col, hyperparams, folds):
    """
    Gapfills y_col using x_cols as predictors
    """
    colsample_bytree=hyperparams.get('colsample_bytree')
    max_depth=hyperparams.get('max_depth')
    min_child_weight=hyperparams.get('min_child_weight')
    subsample=hyperparams.get('subsample')

    # Organize training data
    X, y, id = org_data(data, x_cols, y_col)

    # Fit model
    model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=colsample_bytree, max_depth=max_depth, min_child_weight=min_child_weight, subsample=subsample)

    # CV predict all y where measured data
    y_pred = cross_val_predict(model, X, y, cv=folds)
    data.loc[id, 'modelled_'+y_col] = y_pred

    # Use all available measured data to predict real gaps
    model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=colsample_bytree, max_depth=max_depth, min_child_weight=min_child_weight, subsample=subsample).fit(X, y)
    y_pred_gaps = model.predict(np.asarray(data.loc[data[y_col].isnull(), x_cols]))
    gap_ids = data[data[y_col].isnull()].index
    data.loc[gap_ids, 'modelled_'+y_col] = y_pred_gaps

    return data











































































##############################################################################################################
# random forest stuff


import sys

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from math import*
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
#from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble._forest import _generate_unsampled_indices
#from rfpimp import *
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from math import sqrt
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import roc_auc_score


import scipy
#RF_Model = RandomForestRegressor(n_estimators=100,
 #                                max_features="auto", oob_score=True)
#from hydroeval import *


def permutation_importances(rf, X_train, y_train, metric):
    baseline = metric(rf, X_train, y_train)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(rf, X_train, y_train)
        X_train[col] = save
        imp.append(baseline - m)
    imp = np.array(imp)
    I = pd.DataFrame(
        data={'Feature':X_train.columns,
                  'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I












def oob_regression_r2_score(rf, X_train, y_train):
    """
    Compute out-of-bag (OOB) R^2 for a scikit-learn random forest
    regressor. We learned the guts of scikit's RF from the BSD licensed
    code:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L702
    """
    X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y = y_train.values if isinstance(y_train, pd.Series) else y_train

    n_samples = len(X)
    predictions = np.zeros(n_samples)
    n_predictions = np.zeros(n_samples)
    for tree in rf.estimators_:
        unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples)
        tree_preds = tree.predict(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds
        n_predictions[unsampled_indices] += 1

    if (n_predictions == 0).any():
        warnings.warn("Too few trees; some variables do not have OOB scores.")
        n_predictions[n_predictions == 0] = 1
    predictions /= n_predictions

    oob_score = r2_score(y, predictions)
    return oob_score


#########################################################################



Y=pd.DataFrame(df["ET"])

Y=Y.iloc[:,0]



#X_train = X.loc[df.index <= 18481]  # one row less than row ID

X_train = X[X.index <= 18481].copy()

# just to make sure you are correct
X_train.tail(10)

#y_train = Y.loc[df.index <= 18439]              

y_train = Y[Y.index <= 18481].copy()            

#y_train=y_train["ET"]

#X_test = X.loc[df.index > 18439]    
#y_test = Y.loc[df.index > 18439]   

X_test = X[X.index > 18481].copy()
y_test = Y[Y.index > 18481].copy()             



print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


rf = RandomForestRegressor(criterion='mse',n_estimators=150,min_samples_leaf=5,min_samples_split=6,n_jobs=-1,oob_score=True,max_depth=17,max_features='auto',bootstrap=True)   # model with 16 parameters

#rf = RandomForestRegressor(criterion='mse',n_estimators=100,min_samples_leaf=5,min_samples_split=15,n_jobs=-1,oob_score=True,max_depth=17,max_features='auto',bootstrap=True)   # 

## 8 parameters
rf.fit(X_train, y_train)

# pick between either default or oermuted score
# default score
names =X_train.columns.tolist()
print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
            reverse=True))

#k=sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
 #           reverse=True)

im=pd.DataFrame(rf.feature_importances_, names)
im.columns=["imp"]
im.reset_index(inplace=True)
im=im.sort_values(by=['imp'],ascending=False)

im.to_csv('imp.csv', index=False, header=True)

 #permuted score
#imp=(permutation_importances(rf, X_train, y_train,oob_regression_r2_score))
###############################################################################
labels = y_train#[:, None]
features = X_train[:]
#rf=RF_Model.fit(features, labels) 

#########################################################################

X_test_predict=pd.DataFrame(
    rf.predict(X_test[:])).rename(
    columns={0:'predicted_ET'}).set_index('predicted_ET')

X_train_predict=pd.DataFrame(
    rf.predict(X_train[:])).rename(
    columns={0:'predicted_ET'}).set_index('predicted_ET')
RF_predict = X_train_predict.append(X_test_predict)
RF_predict .reset_index(level=0, inplace=True)


X_test_predict.reset_index(level=0, inplace=True)
r2_score(y_test,X_test_predict)

X_train_predict.reset_index(level=0, inplace=True)
r2_score(y_train,X_train_predict)

obs=X_test_predict
pre=y_test

r2_score(pre,obs)

NSE(pre,obs)

will(pre,obs)

pearsonr(obs,pre)

mean_absolute_error(obs,pre)

ubRMSE(obs,pre)

RMSE=sqrt(mean_squared_error(obs,pre))
RMSE/np.std(obs)

pbias=(np.sum(pre-obs)/np.sum(obs))*100
pbias

###

#################################################################################################
y_train_predicted = rf.predict(X_train)
y_test_predicted_pruned_trees = rf.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_predicted)
mse_test = mean_squared_error(y_test, y_test_predicted_pruned_trees)
print("RF with pruned trees, Train MSE: {} Test MSE: {}".format(mse_train, mse_test))

print("RF with pruned trees, Train RMSE: {} Test RMSE: {}".format(sqrt(mse_train), sqrt(mse_test)))



plt.plot(X_train_predict,color='r')
plt.plot(y_train,color='b')


plt.plot(X_test_predict,color='r')
plt.plot(y_test,color='b')


1 - sum((X_test_predict-y_test)**2)/sum((y_test-np.mean(y_test))**2)


np.sum((X_test_predict-y_test)**2)

#######################################################################
y_train.index=np.arange(0,len(y_train))
sse=mean_squared_error(y_train,X_train_predict)
var = np.var(X_train_predict)
bias = sse -var-0.01
pearson_def(y_train.values,X_train_predict.values)


y_test.index=np.arange(0,len(y_test))
sse=mean_squared_error(y_test,X_test_predict)
var = np.var(X_test_predict)
bias = sse -var-0.01
pearson_def(y_test.values,X_test_predict.values)
##########################################################################


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from skgarden import RandomForestQuantileRegressor



import sklearn.model_selection as xval
from sklearn.datasets.mldata import fetch_mldata
import forestci as fci
plt.scatter(y_test,X_test_predict)
#plt.scatter(y_train,X_train_predict)
plt.plot([0, 12], [0, 12], 'k--')


plt.plot([0, 12], [0, 12], 'k--')

plt.xlabel('Reported MPG')
plt.ylabel('Predicted MPG')
plt.show()

mpg_V_IJ_unbiased= fci.random_forest_error(rf,X_train, X_test)

#plt.errorbar(y_test,X_test_predict, yerr=np.sqrt(mpg_V_IJ_unbiased), fmt='o')

s=np.array(y_test)
t=np.array(X_test_predict)
yerr=np.sqrt(mpg_V_IJ_unbiased)
plt.errorbar(y_test,X_test_predict, yerr=np.sqrt(mpg_V_IJ_unbiased), fmt='o')

plt.errorbar(s,t, yerr, fmt='o')

plt.plot([0, 12], [0, 12], 'k--')
plt.xlabel('Observed Daily ET (mm)')
plt.ylabel('Predicted Daily ET (mm)')
plt.show()




##########################################################################

### work on it for extremes 
#### pick extreme conditions

'ID', 'TIMESTAMP', 'DOY', 'month', 'week', 'ET'

df['ID'],


X=pd.concat((df['prcp_7'],df['prcp_15'],df['prcp_30'],df['vp'],df['tmax_F'],\
             df['LW'],df['SW'],df['wind'],df['SolarZenith_1'],df['ALBEDO_1'],\
             df['soil'],df['irr_nonirr'],df['crop_cover'],df['crop_coeff'],df['cumGDD'],
             df['EVI']
              ),axis=1)

