# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:56:10 2024

@author: ammar
"""

#canody
import pymannkendall as mk
import datetime
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
import numpy as np
from math import sqrt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score
import pandas as pd
import os
from sklearn.metrics import r2_score
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import polyfit
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend
import matplotlib.dates as mdates
import statsmodels.api as sm
##################################################################################

import pandas as pd
import os
from sklearn.metrics import r2_score
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import polyfit
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend
from scipy.stats.stats import pearsonr
import matplotlib.ticker as ticker
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from datetime import datetime as dt

##########################################################################################################

df=daily_prec
# mask precip data data from 1992-2023 
mask = (df['year'] >= 2004) & (df['year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]=pd.to_datetime(df["date"])
prec_mask=df


# partitioning soil evaporation and transpiration

os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")
df=pd.read_csv("hem_gs_all.csv") # half hourly data
df["TIMESTAMP"]=df["datetime"]
#df["DoY"]=df["doy"]

df=pd.concat((df["TIMESTAMP"],df["year"],df['DoY'],df['gs_iPM'], df['gs_FG'],df["VPD"],df["LE"],df["Tair"],df["lambda"],df["PA"],df["sos"],df["eos"]),axis=1)

df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
canop=df


#######################################################################################################
# merge both prec and canop data
#merg=pd.merge(canop, prec_mask, on='TIMESTAMP')
merg=pd.merge_asof(canop, prec_mask, on='TIMESTAMP')
df=merg


df = df[df['prec'] ==0] # drop values where precip is greater than zero. 

#df=pd.concat((df["TIMESTAMP"],df['DoY'],df['gs_iPM'], df['gs_FG'],df["GPP_g"],df["VPD"],df["LE"],df["Tair"],df["lambda"],df["PA"]),axis=1)

df=pd.concat((df["TIMESTAMP"],df['DoY'],df['gs_iPM'], df['gs_FG'],df["VPD"],df["LE"],df["Tair"],df["lambda"],df["PA"],df["sos"],df["eos"]),axis=1)


df['TIMESTAMP']= pd.DatetimeIndex(df.TIMESTAMP)

df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
df["year"]=df.TIMESTAMP.dt.year 

############################################################################################################
#df=df.dropna()

canop=df 


############################################################################################################
   
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
back=df

np.nanpercentile(df["gs_FG"], [5,95])

# gs_iPM: 0.1-1.05
# gs_FG: 0.14-1.5

#####################################################################################################
#gs_iPM method


################################################################################################


#df=back
df=merg
mask = (df['gs_FG'] >= 0.116629) & (df['gs_FG'] <= 0.822035)
df = df.loc[mask]
df.index = np.arange(0, len(df))
gs_FG=df

###########################################################################################################

####################################################################################

#########################################################################################################

#####################################################################################################
#gs_FG method


################################################################################################

df=gs_FG
df.to_csv('gs_FG_all_hem.csv', index=False, header=True)

###########################################################################################################
df=pd.read_csv("gs_FG_all_hem.csv") 

#########################################################################################################
mask = (df['Grow_sea'] == 'NGS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
#####################################################################################################
df_h=df


#########################################################################################################


grow_gs_FG=df_h

df=grow_gs_FG

df= df.groupby(['water_year']).mean()
df=df.reset_index()

mask = (df['water_year'] >= 2006) & (df['water_year'] <= 2022)
df = df.loc[mask]
df.index = np.arange(0, len(df))



df_h_y=df

x1=df_h_y['water_year']
y1=df_h_y['gs_FG'].astype(float)
pearsonr(x1, y1)
mk.original_test(y1)



#back=grow_gs_FG
#df=grow_gs_FG

############################################################################################################
## canopy comparison HEM_EMS


############################################################################################################

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df_h_y['water_year'], df_h_y['gs_FG'],color="green", label="US:Ha2,Tau=0.2,p=0.27",linestyle='--', marker='o')
#plt.fill_between(x1, y1-y1.std(),y1+y1.std(), color='lightgrey')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('$G_{s}$ ($mol$ $m^{-2}$ $s^{-1}$)')  # we already handled the x-label with ax1
#'$ET_{a}$ (mm $day^{-1}}$)'
#ax.set_ylabel('Std Annual conductance')  # we already handled the x-label with ax1
#z = np.polyfit(x, y, 1)
#p = np.poly1d(z)
#plt.plot(x, p(x),color='black')

z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='green')
ax.set_title("US-Ha2:Annual HEM Canopy Conductance ($G_{s}$): Mann Kendall test: p=0.15",fontsize=7)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

##############################################################################################################




plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['water_year'], df['gs_FG'],color="green", label="corr=0.43",linestyle='--', marker='o')
plt.fill_between(x1, y1-y1.std(),y1+y1.std(), color='lightgrey')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('$G_{s}$ ($mol$ $m^{-2}$ $s^{-1}$)')  # we already handled the x-label with ax1
#'$ET_{a}$ (mm $day^{-1}}$)'
#ax.set_ylabel('Std Annual conductance')  # we already handled the x-label with ax1
#z = np.polyfit(x, y, 1)
#p = np.poly1d(z)
#plt.plot(x, p(x),color='black')

z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='green')
ax.set_title("US-Ha2:Annual HEM Canopy Conductance ($G_{s}$): Mann Kendall test: p=0.15",fontsize=7)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()




















plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['water_year'], df['gs_FG'],color="green", label="corr=0.36",linestyle='--', marker='o')
#plt.plot(df['water_year'], df['gs_FG'],color="green", label="gs_FG (corr= -0.66)",linestyle='--', marker='o')
plt.fill_between(x1, y1-y1.std(),y1+y1.std(), color='lightgrey')
#plt.errorbar(x1, y1, 0.08, linestyle='None', marker='o', color='green')
#plt.errorbar(x1, y1, e2, linestyle='None', marker='^',label="gs_FG (corr= -0.58)", color='green')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('$G_{s}$ ($mol$ $m^{-2}$ $s^{-1}$)')  # we already handled the x-label with ax1
z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='green')
ax.set_title("US-Ha2:Growing season HEM Canopy Conductance ($G_{s}$): Mann Kendall test: p=0.2",fontsize=7)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()
###########################################################################################################3

# box plot 

df=grow_gs_iPM
# mask precip data data from 1992-2023 
mask = (df['water_year'] >= 1992) & (df['water_year'] <= 2008)
df = df.loc[mask]
df.index = np.arange(0, len(df))
#df["TIMESTAMP"]=pd.to_datetime(df["date"])
before_iPM=df["gs_iPM"]

df=grow_gs_iPM
# mask precip data data from 1992-2023 
mask = (df['water_year'] >= 2009) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
#df["TIMESTAMP"]=pd.to_datetime(df["date"])
after_iPM=df["gs_iPM"]

df=grow_gs_FG
# mask precip data data from 1992-2023 
mask = (df['water_year'] >= 1992) & (df['water_year'] <= 2008)
df = df.loc[mask]
df.index = np.arange(0, len(df))
#df["TIMESTAMP"]=pd.to_datetime(df["date"])
before_FG=df["gs_FG"]

df=grow_gs_FG
# mask precip data data from 1992-2023 
mask = (df['water_year'] >= 2009) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
#df["TIMESTAMP"]=pd.to_datetime(df["date"])
after_FG=df["gs_FG"]

my_dict1={'gs_iPM':before_iPM, 'gs_FG':before_FG}

my_dict2={'gs_iPM':after_iPM, 'gs_FG':after_FG}

#ax.set_xticklabels(['gs_iPM (1992-2008)','gs_FG (1992-2008)','gs_iPM (2009-2023)','gs_FG (2009-2023)'])
c1="C0"
c2="C2"

plt.rcParams['figure.figsize'] = (3.5,2.5)
fig, ax = plt.subplots()

bp1=ax.boxplot(my_dict1.values(),sym='.',positions =[1,8],notch=False, widths=3, 
patch_artist=True,whiskerprops = dict(linestyle='--'
,color='royalblue', linewidth=0.5),boxprops=dict(facecolor='navy',edgecolor='slateblue'))

bp2=ax.boxplot(my_dict2.values(),sym='.',positions = [4,11],notch=False, widths=3, 
 patch_artist=True, boxprops=dict(facecolor='pink',edgecolor='salmon'),whiskerprops = dict(linestyle='--'
               ,color='indianred' , linewidth=0.5))

plt.setp(bp1['medians'], color='slateblue')
#plt.setp(bp['medians'][0], color='slateblue')
plt.setp(bp2['medians'][0], color='salmon')

plt.xticks([1, 4, 8,11], ['gs_iPM','gs_iPM','gs_FG','gs_FG'])

ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['1992-2008', '2009-2023'], loc='upper left',prop={'size': 5})

ax.set_ylabel('Annual Canopy conductance')  # we already handled the x-label with ax1
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
#ax.set_ylim(-0.5,1.5)
#ax.set_title("Training Sites, Potatoes: (2018-2020), n=309",fontsize=6)
ax.set_title("whole year",fontsize=7)
ax.yaxis.label.set_size(8)
ax.xaxis.label.set_size(10) #there is no label 
#ax.locator_params(axis='y', nbins=4)
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

#############################################################################################################
## partitioning df_gs_iPM
# ameriflux unit, GEP=umolco2 m-2 s-1, VPD=kha
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")
#df_gs_iPM=pd.read_csv('gs_iPM_all.csv')
df=pd.read_csv('gs_FG_all_hem.csv')

#####################################################################################################
# use this part only for growing season. skip it if you want whole year. 

mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))

#####################################################################
#grow=df
std=df.groupby(df.year).std()
std=df.reset_index()

#df=grow
#df= df.groupby('water_year').mean()
#df=df.reset_index()

#df["std_gs_FG"]=std["gs_FG"]
#grow_gs_FG=df

df=df_gs_FG
# calculate psychrometric constant 
cp= 1.013*10**3 # J kg-1 K-1 # specific heat of air
den_a=1.293 # air density in kg m-3
df["psy"]=(cp*df["PA"])/(0.622*df["lambda"])

df["vpd_leaf"]=(df["psy"]*df["LE"])/(den_a*cp*(df["gs_iPM"]*0.024))#multiply gs with 0.024 for unit conversion to ms-1

##########################################################################################################

## partitioning
# ameriflux unit, GEP=umolco2 m-2 s-1, VPD=kha
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")
df_gs_FG=pd.read_csv('gs_FG_all.csv')
df=df_gs_FG
# calculate psychrometric constant 
cp= 1.013*10**3 # J kg-1 K-1 # specific heat of air
den_a=1.293 # air density in kg m-3
df["psy"]=(cp*df["PA"])/(0.622*df["lambda"])

df["vpd_leaf"]=(df["psy"]*df["LE"])/(den_a*cp*(df["gs_FG"]*0.024))#multiply gs with 0.024 for unit conversion to ms-1

##########################################################################################################
# change value to m


#gs_gs_FG method
##########################################################################################################
m=0.6
df["X"]=(df["GPP_g"])/(df["vpd_leaf"]**m)
df["Y"]=df["gs_FG"]

plt.plot(df["X"])


np.percentile(df["X"], [5,95])


#####################################################################################################

back=df
df=back
X=df["X"]
Y=df["Y"]

mask = (df['X'] >= 1.35) & (df['X'] <= 79.10)
df = df.loc[mask]
df.index = np.arange(0, len(df))
back1=df
df=back1
df["TIMESTAMP"]=pd.DatetimeIndex(df.TIMESTAMP)
df_daily=df.reset_index().groupby(pd.Grouper(key="TIMESTAMP", freq='D')).mean()
df=df_daily
df=df.dropna()
df=df.reset_index()
# Assuming you have a DataFrame named df and want to create a new column 'new_id' every n rows
n = 16  
df['unique_id'] = (df.index // n) + 1
backc=df

df=backc


df["feature"]=df["X"]
df["target"]=df["Y"]



regression_results = pd.DataFrame(columns=['unique_id', 'intercept', 'slope'])

# Group the data by 'unique_id'
grouped_data = df.groupby('unique_id')

# Loop through each group and perform linear regression
for group_name, group_data in grouped_data:
    X = sm.add_constant(group_data['feature'])  # Add a constant term to the feature
    y = group_data['target']

    model = sm.OLS(y, X).fit()  # Fit the linear regression model

    # Append the results to the regression_results DataFrame
    regression_results = regression_results.append({
        'unique_id': group_name,
        'intercept': model.params['const'],
        'slope': model.params['feature']
    }, ignore_index=True)

# Display the regression results
print(regression_results)


coeff=regression_results

coeff["intercept"]=coeff["intercept"].mask(coeff["intercept"] < 0, 0)

merg=pd.merge(df, coeff, on='unique_id')
df=merg

m=0.6


df["X"]=(df["GPP_g"])/(df["vpd_leaf"]**m)
df["Y"]=df["gs_FG"]

df["G1"]=df["slope"]
df["Go"]=df["intercept"]

df["y_obs"]=df["gs_FG"]
df["y_pred"]=df["Go"]+(df["G1"]*((df["GPP_g"])/(df["vpd_leaf"]**m)))
df["GS_p"]=df["y_pred"]

df["rs"]=1/df["GS_p"]
df["rsoil"]=1/df["Go"]
df["Gveg"]=(df["G1"]*((df["GPP_g"])/(df["vpd_leaf"]**0.6)))
df["rveg"]=1/df["Gveg"]
df['lambda']=(3149000-2370*(df['Tair']+273.16))*1e-6
df['ET']=(df['LE']/df['lambda'])*(1/1e6)*3600

df["Evapo"]=(df['ET']*df["rs"])/df["rsoil"]

df["trans"]=(df['ET']*df["rs"])/df["rveg"]
df.to_csv('part_gs_FG.csv', index=False, header=True)
##########################################################################################################
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")

df=pd.read_csv("part_gs_FG.csv")
df["evap_ratio"]=df["Evapo"]/df["ET"]
df["trans_ratio"]=df["trans"]/df["ET"]



df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
grow=df



mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))

part_gs_FG=df

df= df.groupby(['water_year']).mean()
df=df.reset_index()
annual=df

plt.plot(df["Evapo"]/df["ET"])
plt.plot(df["trans"]/df["ET"])
###########################################################################################################
back=df
df=back

df = df.iloc[:, 1:]
df=df.rolling(window=7).mean()
df['water_year']=back['water_year']

mask = (df['water_year'] >= 1999) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
Et_yearly=df


x=df["water_year"].values.flatten()
#y=df["Go"].values.flatten()  # unit conversion 
y=df["evap_ratio"].values.flatten()  #
y1=df["trans_ratio"].values.flatten()  #


pearsonr(x, y)
pearsonr(x, y1)
mk.original_test(y1)


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['water_year'], df["evap_ratio"],color="brown", label="E/ET: corr=-0.4",linestyle='--', marker='o')
plt.fill_between(x, y-y.std(),y+y.std(), color='lightgrey')
plt.plot(df['water_year'], df["trans_ratio"],color="green", label="T/ET: corr=0.4",linestyle='--', marker='o')
plt.fill_between(x, y1-y1.std(),y1+y1.std(), color='lightgrey')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='center left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
#ax.set_ylabel('$G_{o}$ ($mol$ $m^{-2}$ $s^{-1}$)')  # we already handled the x-label with ax1
ax.set_ylabel('Annual ratio')  # we already handled the x-label with ax1
ax.set_ylim(0.2,0.8)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x),color='brown', label='E/ET')

z1 = np.polyfit(x, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x, p1(x),color='green', label='T/ET')

ax.set_title("EMS:Annual Ratio:Mann Kendall p=0.07 ",fontsize=7)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#########################################################################################################

## growing season

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['water_year'], df["evap_ratio"],color="brown", label="E/ET: corr=-0.4",linestyle='--', marker='o')
plt.fill_between(x, y-y.std(),y+y.std(), color='lightgrey')
plt.plot(df['water_year'], df["trans_ratio"],color="green", label="T/ET: corr=0.4",linestyle='--', marker='o')
plt.fill_between(x, y1-y1.std(),y1+y1.std(), color='lightgrey')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='center left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
#ax.set_ylabel('$G_{o}$ ($mol$ $m^{-2}$ $s^{-1}$)')  # we already handled the x-label with ax1
ax.set_ylabel('Annual ratio')  # we already handled the x-label with ax1
ax.set_ylim(0.2,0.8)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x),color='brown', label='E/ET')

z1 = np.polyfit(x, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x, p1(x),color='green', label='T/ET')

ax.set_title("EMS:Annual Ratio:Mann Kendall p=0.07 ",fontsize=7)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()













###########################################################################################################


x1=x.values.flatten()
y1=y.values.flatten()
pearsonr(x1, y1)

x=pd.DataFrame(df["y_obs"])
y=pd.DataFrame(df["y_pred"])

reg = LinearRegression(fit_intercept=True).fit(x, y)
reg.score(x,y) 

r2_score(x, y)

ubRMSE(df["y_obs"],df["y_pred"])
plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.scatter(x1, y1, label="gs_FG (corr= 0.79)",linestyle='--', marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('predicted gs')  # we already handled the x-label with ax1
ax.set_xlabel('observed gs')  # we already handled the x-label with ax1
#ax.set_ylabel('Std Annual conductance')  # we already handled the x-label with ax1
z = np.polyfit(x1, y1, 1)
p = np.poly1d(z)
plt.plot(x1, p(x1),color='black')

ax.set_title("canopy conductance",fontsize=15)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#########################################################################################################

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()

plt.plot(df["TIMESTAMP"], df["y_obs"],color="black", label="obs",linestyle='--')
plt.plot(df["TIMESTAMP"], df["y_pred"],color="green", label="y_pred_m=0.6",linestyle='--')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.legend(loc='lower left',prop={'size': 'small'})
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('canopy conductance Gs (mol m-2 s-1)')  # we already handled the x-label with ax1
ax.set_title("R2=0.62, ubRMSE=0.028",fontsize=15)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
#fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

##############################################################################################################
# convert canopy conductance to LE

df=pd.read_csv('part_gs_FG.csv')

part_gs_FG=df
# remove last two rows
#df = df.iloc[:-2]

df=part_gs_FG

df["Gveg"]=(df["G1"]*((df["GPP_g"])/(df["vpd_leaf"]**0.6)))



#find

df["diff"]=df["y_pred"]-df["y_obs"]

plt.plot(df["year"],df["diff"])


plt.plot(df["Go"])

plt.plot(df["Gveg"])

plt.plot(df["gs_iPM"])

df['LE_pred']=(df["vpd_leaf"]*(den_a*cp*(df["y_pred"]*0.024)))/(df["psy"])


#multiply gs with 0.024 for unit conversion to ms-1


plt.plot(df["LE"])

plt.plot(df["LE_pred"])


back_check=df

df=back_check

df["month"]=pd.to_datetime(df.TIMESTAMP).dt.month 

#########################################################################################################

### 

df=back_check  # all data including growing season

df["evap_perc"]=(df["Go"]*100)/df["y_pred"]


def Grow_sea(df):
    
    if (df['DoY'] >=df['GS_start']) and (df['DoY'] <=df['GS_end']):
        return 'GS'
    
    
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
back=df

df=back
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
grow=df

df=grow

df["check"]=df["Gveg"]+df["Go"]


plt.plot(df["y_pred"])
plt.plot(df["check"])

plt.scatter(df["LE"],df["LE_pred"])


#############################################################################################
df= df.groupby(['water_year']).mean()
df=df.reset_index()
annual=df
#############################################################################################

## split timeseries

df=annual

x1=df["year"].values.flatten()
y1=(df["Go"].values.flatten()/0.024)*1000   # unit conversion 

pearsonr(x1, y1)

mk.original_test(y1)


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['water_year'], df['Go']/0.024,color="green", label="corr=-0.36",linestyle='--', marker='o')
#plt.plot(df['water_year'], df['gs_FG'],color="green", label="gs_FG (corr= -0.66)",linestyle='--', marker='o')
plt.fill_between(x1, y1-0.35,y1+0.35, color='lightgrey')
#plt.errorbar(x1, y1, 0.08, linestyle='None', marker='o', color='green')
#plt.errorbar(x1, y1, e2, linestyle='None', marker='^',label="gs_FG (corr= -0.58)", color='green')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
#ax.set_ylabel('$G_{o}$ ($mol$ $m^{-2}$ $s^{-1}$)')  # we already handled the x-label with ax1
ax.set_ylabel('$G_{o}$ ($mol$ $m^{-2}$ $s^{-1}$)')  # we already handled the x-label with ax1

#'$ET_{a}$ (mm $day^{-1}}$)'
#ax.set_ylabel('Std Annual conductance')  # we already handled the x-label with ax1
#z = np.polyfit(x, y, 1)
#p = np.poly1d(z)
#plt.plot(x, p(x),color='black')

z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='green')
ax.set_title("Growing Season Canopy Conductance ($G_{s}$): Mann Kendall test: p=0.0002",fontsize=7)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

######################################################################################33

x1=df["water_year"].values.flatten()
y1=df["Gveg"].values.flatten()/0.024
pearsonr(x1, y1)
mk.original_test(y1)


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(x1, y1, label="gveg (corr= 0.023)",linestyle='--', marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('gveg_transpiration')  # we already handled the x-label with ax1
ax.set_xlabel('year')  # we already handled the x-label with ax1
#ax.set_ylabel('Std Annual conductance')  # we already handled the x-label with ax1
z = np.polyfit(x1, y1, 1)
p = np.poly1d(z)
plt.plot(x1, p(x1),color='black')

#ax.set_title("canopy conductance",fontsize=15)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

############################################################################################################


plt.scatter(df["LE"],df["LE_pred"])

x1=df["LE"].values.flatten()
y1=df["LE_pred"].values.flatten()
pearsonr(x1, y1)


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.scatter(x1, y1, label="LE (corr= 0.76)",linestyle='--', marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('predicted LE')  # we already handled the x-label with ax1
ax.set_xlabel('observed LE')  # we already handled the x-label with ax1
#ax.set_ylabel('Std Annual conductance')  # we already handled the x-label with ax1
z = np.polyfit(x1, y1, 1)
p = np.poly1d(z)
plt.plot(x1, p(x1),color='black')

#ax.set_title("canopy conductance",fontsize=15)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()
##########################################################################################

df=EMS_sum_annual_ET

x1=df["Year"].values.flatten()
y1=df["ET_f"].values.flatten()
pearsonr(x1, y1)

mk.original_test(y1)





df=df.drop([18])

#gs_iPM method
##########################################################################################################
m=1
df["X"]=(df["GPP_g"])/(df["vpd_leaf"]**m)
df["Y"]=df["gs_iPM"]

plt.plot(df["X"])


np.percentile(df["X"], [5,95])


#####################################################################################################

back=df


df=back
X=df["X"]
Y=df["Y"]

mask = (df['X'] >= 0.968) & (df['X'] <= 50.2627)
df = df.loc[mask]
df.index = np.arange(0, len(df))
back1=df
df=back1
df["TIMESTAMP"]=pd.DatetimeIndex(df.TIMESTAMP)
df_daily=df.reset_index().groupby(pd.Grouper(key="TIMESTAMP", freq='D')).mean()
df=df_daily
df=df.dropna()
df=df.reset_index()
# Assuming you have a DataFrame named df and want to create a new column 'new_id' every n rows
n = 16  
df['unique_id'] = (df.index // n) + 1
backc=df

df=backc


df["feature"]=df["X"]
df["target"]=df["Y"]



regression_results = pd.DataFrame(columns=['unique_id', 'intercept', 'slope'])

# Group the data by 'unique_id'
grouped_data = df.groupby('unique_id')

# Loop through each group and perform linear regression
for group_name, group_data in grouped_data:
    X = sm.add_constant(group_data['feature'])  # Add a constant term to the feature
    y = group_data['target']

    model = sm.OLS(y, X).fit()  # Fit the linear regression model

    # Append the results to the regression_results DataFrame
    regression_results = regression_results.append({
        'unique_id': group_name,
        'intercept': model.params['const'],
        'slope': model.params['feature']
    }, ignore_index=True)

# Display the regression results
print(regression_results)


coeff=regression_results

coeff["intercept"]=coeff["intercept"].mask(coeff["intercept"] < 0, 0)

merg=pd.merge(df, coeff, on='unique_id')
df=merg


m=0.6


df["X"]=(df["GPP_g"])/(df["vpd_leaf"]**m)
df["Y"]=df["gs_iPM"]

df["G1"]=df["slope"]
df["Go"]=df["intercept"]

df["y_obs"]=df["gs_iPM"]
df["y_pred"]=df["Go"]+(df["G1"]*((df["GPP_g"])/(df["vpd_leaf"]**m)))

ipm_model=df


x=pd.DataFrame(df["y_obs"])
y=pd.DataFrame(df["y_pred"])

reg = LinearRegression(fit_intercept=True).fit(x, y)
reg.score(x,y) 

r2_score(x, y)

ubRMSE(df["y_obs"],df["y_pred"])


part_gs_iPM=df

df.to_csv('part_gs_iPM.csv', index=False, header=True)
###########################################################################################################


x1=x.values.flatten()
y1=y.values.flatten()
pearsonr(x1, y1)


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.scatter(x1, y1, label="gs_iPM (corr= 0.7)",linestyle='--', marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('predicted gs')  # we already handled the x-label with ax1
ax.set_xlabel('observed gs')  # we already handled the x-label with ax1
#ax.set_ylabel('Std Annual conductance')  # we already handled the x-label with ax1
z = np.polyfit(x1, y1, 1)
p = np.poly1d(z)
plt.plot(x1, p(x1),color='black')

ax.set_title("canopy conductance",fontsize=15)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#####################################################################################################

dailydf=df

df=dailydf

df["month"]=df.TIMESTAMP.dt.month 


df= df.groupby(['water_year','month']).mean()
df=df.reset_index()

annual=df

###################################################################################################
plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()

plt.plot(df["year"], df["y_obs"],color="black", label="obs",linestyle='--')
plt.plot(df["year"], df["y_pred"],color="green", label="y_pred_m=0.6",linestyle='--')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.legend(loc='lower left',prop={'size': 'small'})
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('canopy conductance Gs (mol m-2 s-1)')  # we already handled the x-label with ax1
ax.set_title("R2=0.5, ubRMSE=0.016",fontsize=15)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
#fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

###########################################################################################################






