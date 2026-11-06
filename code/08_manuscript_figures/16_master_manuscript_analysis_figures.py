# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:42:53 2024

@author: ammar
"""

def stat (obs,pre):
    nse=NSE(pre,obs)
   # pear=pearsonr(obs,pre)
    wil=will(pre,obs)   
    MAE=mean_absolute_error(obs,pre)
    RMSE=sqrt(mean_squared_error(obs,pre))
   # reg_=reg.score(x1, y1)
    pbias=(np.sum(pre-obs)/np.sum(obs))*100
    return nse,pear,MAE,reg_,pbias

from scipy.stats.stats import pearsonr
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
from matplotlib.ticker import FormatStrFormatter
import pymannkendall as mk
import math
import statsmodels.formula.api as smf
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

import numpy as np
import pyhomogeneity as hg
from datetime import datetime as dt

def Grow_sea(df):
    
    if (df['DoY'] >=df['sos']) and (df['DoY'] <=df['eos']):
        return 'GS'
    else:
        return 'NGS'
######################################################################################################

# variable importance 
#os.chdir(r"C:\ammara_MD\a_harvard\Project\data\climate\fisher_station_since_2001")

os.chdir(r"C:\ammara_MD\a_harvard\Project\data\climate\climate data_since_1964")

df=pd.read_csv("hf300-05-daily-m.csv")  
df['TIMESTAMP'] = pd.to_datetime(df["date"], format='%Y-%m-%d') # 1964-2021
df.head(10)
df.tail(10)
df=df.drop(['f.airt', 'f.airtmax', 'f.airtmin','f.prec'], axis=1)
df.isnull().values.any()
df['month'] = df['TIMESTAMP'].dt.month
df['year'] = df['TIMESTAMP'].dt.year
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
mask = (df['water_year'] >= 1965) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
daily_prec=df

df= df.groupby(['water_year']).sum()
df=df.reset_index()
annual_prec=df


x1=df['water_year']
y1=df['prec'].astype(float)
#(x1, y1)
mk.original_test(y1)

#Mann_Kendall_Test(trend='increasing', h=True, p=4.245087600041053e-05, z=4.093718988874227, 
#Tau=0.36645236703682055, s=627.0, var_s=23383.666666666668, slope=5.9899999999999975, intercept=914.1900000000002)

#Sen's slope * length of period)/mean *100
((5.99 * 59)/1111.785) *100


#31.8 %
#slope* data point/average
df=annual_prec
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
x1=df['water_year']
y1=df['prec'].astype(float)
#(x1, y1)
mk.original_test(y1)



#Mann_Kendall_Test(trend='increasing', h=True, p=0.0449013762794892, z=2.0055772020384914, 
#Tau=0.25591397849462366, s=119.0, var_s=3461.6666666666665, slope=9.746153846153849, intercept=1011.2076923076924)

((9.746 * 31)/1189.683871) *100
#25.4 %


df=daily_prec
std=df.groupby(df["water_year"])['prec'].std()

std=df.groupby(df["water_year"])['prec'].std()
var=df.groupby(df["water_year"])['prec'].var()


df=annual_prec

#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5,5)
fig, ax = plt.subplots()
#p=6.9*10$^{-5}$
#plt.plot(df['water_year'], df['prec'],color="blue",label="Precipitation 1964-2023),  p=0.01, Tau=0.33, slope=0.83", marker='D',markersize=4)
plt.plot(df['water_year'], df['prec'],color="blue",label="1965-2023,  p=4.2*10$^{-5}$, Tau=0.37, slope=6.0", marker='D',markersize=4,linestyle='--')

plt.plot(df['water_year'].tail(32), df['prec'].tail(32),color="teal",label="1993-2023,  p=4.0*10$^{-2}$, Tau=0.26, slope=9.8", marker='.',markersize=7,linestyle='--')

ax.tick_params(axis='y', labelcolor='black')
ax.legend(loc='upper left', fontsize = '8',labelcolor='linecolor')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Total Annual Precipitation (mm)')  # we already handled the x-label with ax1
ax.set_ylim(570,2000)

z1 = np.polyfit(df['water_year'], df['prec'], 1)
p1 = np.poly1d(z1)
plt.plot(df['water_year'], p1(df['water_year']),color='blue')

z2 = np.polyfit(df['water_year'].tail(32), df['prec'].tail(32), 1)
p2 = np.poly1d(z2)
plt.plot(df['water_year'].tail(32), p2(df['water_year'].tail(32)),color='teal',linestyle='--')


#ax.set_title("Harvard Forest",fontsize=10)
ax.set_title("Prospect Hill Tract (Harvard Forest), Lat:42.53, Long:-72.12",fontsize=10)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
#ax.yaxis.set_major_formatter(FormatStrFormatter('%f'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

##########################################################################################################3

## find slope for two different periods 

df=annual_prec

df['year'] = 'Period1'

df2=  df[(df['water_year'] >= 1993) & (df['water_year'] <= 2023)]

df2['year'] = 'Period2'

df3 = pd.concat([df, df2], ignore_index=True)


# Fit the linear model with interaction
model = smf.ols('prec ~ water_year * year', data=df3).fit()

# Print the summary
print(model.summary())


model = smf.ols('airt ~ water_year * year', data=df3).fit()

# Print the summary
print(model.summary())












########################################################################################################

## air temperature
df=daily_prec

df= df.groupby(['water_year']).mean()
df=df.reset_index()
annual_mean=df

7.01-9.9
2003-1991

x1=df['water_year']
y1=df['airt'].astype(float)
mk.original_test(y1)

# %change = (Sen's slope * length of period)*100 / mean
(1.324* 44)*100/(483.78)
#12.04 %


df=annual_mean
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
x1=df['water_year']
y1=df['airt'].astype(float)
#(x1, y1)
mk.original_test(y1)

((0.0477 * 31)/8.237448) *100
#17.95 %

df=annual_mean
#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5,5)
fig, ax = plt.subplots()
plt.bar(df['water_year'], df['airt'],color="brown",label="$T_{avg}$ (1965-2023),  p=1.2*10$^{-4}$, Tau=0.34, slope=0.02")

plt.bar(df['water_year'].tail(32), df['airt'].tail(32),color="coral",label="$T_{avg}$ (1993-2023),  p=3.9*10$^{-3}$, Tau=0.37, slope=0.05")
plt.plot(df['water_year'], df['airtmax'],color="red",marker='.',markersize=4,linestyle='--',label="$T_{max}$ (1965-2023),  p=1.1*10$^{-8}$, Tau=0.51, slope=0.04")

ax.tick_params(axis='y', labelcolor='black')
ax.legend(loc='upper left', fontsize = '7',labelcolor='linecolor',labelspacing =0.0008)
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Annual Air Temperature (${C}^o$)')  # we already handled the x-label with ax1
ax.set_ylim(0,18)

z1 = np.polyfit(df['water_year'], df['airt'], 1)
p1 = np.poly1d(z1)
plt.plot(df['water_year'], p1(df['water_year']),color='brown')

z2 = np.polyfit(df['water_year'].tail(32), df['airt'].tail(32), 1)
p2 = np.poly1d(z2)
plt.plot(df['water_year'].tail(32), p2(df['water_year'].tail(32)),color='coral',linestyle='--')

z3 = np.polyfit(df['water_year'], df['airtmax'], 1)
p3 = np.poly1d(z3)
plt.plot(df['water_year'], p3(df['water_year']),color='red')

#ax.set_title("Harvard Forest",fontsize=10)
ax.set_title("Prospect Hill Tract (Harvard Forest), Lat:42.53, Long:-72.12",fontsize=10)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
#ax.yaxis.set_major_formatter(FormatStrFormatter('%f'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=8))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


#######################################################################################################
#compare Rnet for both sites

import pandas as pd

# --- load ---
dfe = pd.read_csv("EMS_daily.csv")  # has NETRAD_fill, ET, Year, DoY, sos, eos
dfh = pd.read_csv("HEM_daily.csv")  # has NETRAD, ET, Year, DoY, sos, eos

def gs_slice(df, rn_col):
    rows = []
    for y in sorted(df['Year'].unique()):
        # assume one sos/eos value per year
        sos = df.loc[df['Year']==y, 'sos'].iloc[0]
        eos = df.loc[df['Year']==y, 'eos'].iloc[0]
        m = (df['Year']==y) & (df['DoY']>=sos) & (df['DoY']<=eos)
        if m.any():
            rows.append({
                "Year": y,
                "sos": sos,
                "eos": eos,
                "Rn_mean": df.loc[m, rn_col].mean(),
                "Rn_median": df.loc[m, rn_col].median(),
                "ET_mean": df.loc[m, "ET"].mean(),
                "ET_median": df.loc[m, "ET"].median()
            })
    return pd.DataFrame(rows)

ems_gs = gs_slice(dfe, "NETRAD_fill")
hem_gs = gs_slice(dfh, "NETRAD")

# align by year and compute differences
cmp = (ems_gs.merge(hem_gs, on="Year", suffixes=("_EMS","_HEM"))
              .assign(
                  dRn_mean = lambda x: x.Rn_mean_EMS - x.Rn_mean_HEM,
                  dET_mean = lambda x: x.ET_mean_EMS - x.ET_mean_HEM,
                  dRn_pct  = lambda x: 100*(x.Rn_mean_EMS - x.Rn_mean_HEM)/x.Rn_mean_HEM,
                  dET_pct  = lambda x: 100*(x.ET_mean_EMS - x.ET_mean_HEM)/x.ET_mean_HEM
              ))

# print headline numbers you can cite
print("Growing-season EMS vs HEM (means across overlapping years)")
print(cmp[["Year","Rn_mean_EMS","Rn_mean_HEM","dRn_mean","dRn_pct",
           "ET_mean_EMS","ET_mean_HEM","dET_mean","dET_pct"]].round(2))

# also a single multi-year summary you can paste verbatim
summary = {
    "Rn_EMS_mean": ems_gs["Rn_mean"].mean(),
    "Rn_HEM_mean": hem_gs["Rn_mean"].mean(),
    "ET_EMS_mean": ems_gs["ET_mean"].mean(),
    "ET_HEM_mean": hem_gs["ET_mean"].mean(),
}
print("\nMulti-year GS means:")
for k,v in summary.items():
    print(f"{k}: {v:.2f}")
print(f"\nΔRn (EMS-HEM): {summary['Rn_EMS_mean']-summary['Rn_HEM_mean']:.2f} W m^-2")
print(f"ΔET (EMS-HEM): {summary['ET_EMS_mean']-summary['ET_HEM_mean']:.2f} (units of ET column)")

###########################################################################################################

## calculate roughness values






#########################################################################################################

os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")

df=pd.read_csv("EMS_daily.csv")


df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
daily_ha1=df
df= df.groupby(['water_year']).mean()
df=df.reset_index()
annual_mean_ha1=df






plt.plot(df["water_year"],df["Rg_fill"])

#0.19-0.36
#1993-2021

plt.rcParams['figure.figsize'] = (5,5)
fig, ax = plt.subplots()
plt.plot(df['water_year'], df['VPD_kpa'],color="brown",label="VPD,p=2.9*10$^{-5}$, Tau=0.53, slope=0.038",marker='8',markersize=7,linestyle='--')

#plt.errorbar(df['water_year'],df['VPD_fill'], var, color='blue', ls='-', capsize=2, capthick=0.5,marker='o',markersize=5, label='Pearson Correlation (r)=0.40' )

ax.tick_params(axis='y', labelcolor='black')
ax.legend(loc='upper left', fontsize = '8',labelcolor='linecolor')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Mean Annual VPD (kpa)')  # we already handled the x-label with ax1
#ax.set_ylim(0,18)

z1 = np.polyfit(df['water_year'], df['VPD_kpa'], 1)
p1 = np.poly1d(z1)
plt.plot(df['water_year'], p1(df['water_year']),color='brown')
ax.legend()
ax.legend(loc='upper left', fontsize = '8')
ax.set_ylim(2.8,5.0)
ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
plt.plot(df['water_year'], df['PAR_fill'],color="forestgreen",label="PAR,  p=1.2*10$^{-6}$, Tau=0.62, slope=1.75",marker='+',markersize=10,linestyle='--')
#ax2.scatter(df["seasonal"],df.prec_inch, color='purple', label="Precipitation")
ax2.legend()
ax2.legend(loc='lower right', fontsize = '8')
ax2.set_ylim(260,380)
#ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 10)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 10)
ax2.yaxis.label.set_size(10)
ax2.xaxis.label.set_size(10)
ax2.set_ylabel('Mean Annual PAR (µmol $m^{-2}$ $s^{-1}$)')  # we already handled the x-label with ax1

z2 = np.polyfit(df['water_year'], df['PAR_fill'], 1)
p2= np.poly1d(z2)
plt.plot(df['water_year'], p2(df['water_year']),color='forestgreen')

#ax.set_title("Harvard Forest",fontsize=10)
ax.set_title("US-Ha1: Harvard Forest EMS Tower,Deciduous Broadleaf Forest",fontsize=9)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()
########################################################################################################33


### with Rg instead of PAR


os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")
df=pd.read_csv("EMS_daily.csv")
EMS_daily=df
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
daily_ha1=df
df= df.groupby(['water_year']).mean()
df=df.reset_index()
annual_mean_ha1=df
df_m=annual_mean_ha1


os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")
df=pd.read_csv("EMS_daily.csv")
EMS_daily=df
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
daily_ha1=df
df= df.groupby(['water_year']).sum()
df=df.reset_index()
annual_mean_ha1=df
df_s=annual_mean_ha1



#0.19-0.36
#1993-2021

x1=df['water_year']
y1=df['Rg_fill'].astype(float)
#(x1, y1)
mk.original_test(y1)

plt.plot(df["Rg_fill"])


plt.rcParams['figure.figsize'] = (5,5)
fig, ax = plt.subplots()
plt.plot(df['water_year'], df['VPD_fill']*10,color="brown",label="VPD,p=2.9*10$^{-5}$, Tau=0.53, slope=0.038",marker='8',markersize=7,linestyle='--')

#plt.errorbar(df['water_year'],df['VPD_fill'], var, color='blue', ls='-', capsize=2, capthick=0.5,marker='o',markersize=5, label='Pearson Correlation (r)=0.40' )

ax.tick_params(axis='y', labelcolor='black')
ax.legend(loc='upper left', fontsize = '8',labelcolor='linecolor')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Mean Annual VPD (kpa)')  # we already handled the x-label with ax1
#ax.set_ylim(0,18)

z1 = np.polyfit(df['water_year'], df['VPD_fill']*10, 1)
p1 = np.poly1d(z1)
plt.plot(df['water_year'], p1(df['water_year']),color='brown')
ax.legend()
ax.legend(loc='upper left', fontsize = '8')
#ax.set_ylim(2.8,5.0)
ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
plt.plot(df['water_year'], df['Rg_fill'],color="forestgreen",label="Rg,  p=1.6*10$^{-2}$, Tau=0.31, slope=0.79",marker='+',markersize=10,linestyle='--')
#ax2.scatter(df["seasonal"],df.prec_inch, color='purple', label="Precipitation")
ax2.legend()
ax2.legend(loc='lower right', fontsize = '8')
#ax2.set_ylim(260,380)
#ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 10)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 10)
ax2.yaxis.label.set_size(10)
ax2.xaxis.label.set_size(10)
#ax2.set_ylabel('Mean Annual PAR (µmol $m^{-2}$ $s^{-1}$)')  # we already handled the x-label with ax1
ax2.set_ylabel('Mean Annual Incoming Shortwave radiation (W$m^{-2}$')  # we already handled the x-label with ax1

z2 = np.polyfit(df['water_year'], df['PAR_fill'], 1)
p2= np.poly1d(z2)
plt.plot(df['water_year'], p2(df['water_year']),color='forestgreen')

#ax.set_title("Harvard Forest",fontsize=10)
ax.set_title("US-Ha1: Harvard Forest EMS Tower,Deciduous Broadleaf Forest",fontsize=9)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


((0.79 * 31)/ 148.4) *100  # 1993-2008    # 16.5

####################################################################################################

# wind speed graph

df=EMS_daily
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
daily_ha1=df
df= df.groupby(['water_year']).mean()
df=df.reset_index()
annual_mean_ha1=df


df=annual_mean_ha1
x1=df['water_year']
y1=df['WS_Rd'].astype(float)
#(x1, y1)
mk.original_test(y1)


plt.rcParams['figure.figsize'] = (5,5)
fig, ax = plt.subplots()
plt.plot(df['water_year'], df['WS_Rd'],color="black",label="WS, p=1.4*10$^{-7}$, Tau=0.67, slope=-0.02",marker='^',markersize=7,linestyle='--')
#plt.errorbar(df['water_year'],df['VPD_fill'], var, color='blue', ls='-', capsize=2, capthick=0.5,marker='o',markersize=5, label='Pearson Correlation (r)=0.40' )
ax.tick_params(axis='y', labelcolor='black')
ax.legend(loc='upper left', fontsize = '8',labelcolor='linecolor')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Mean Annual Wind Speed (ms$^{-1}$)')  # we already handled the x-label with ax1
#ax.set_ylim(0,18)

z1 = np.polyfit(df['water_year'], df['WS_Rd'], 1)
p1 = np.poly1d(z1)
plt.plot(df['water_year'], p1(df['water_year']),color='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '8')
#ax.set_ylim(2.8,5.0)
#ax.set_title("Harvard Forest",fontsize=10)
ax.set_title("US-Ha1: Harvard Forest EMS Tower,Deciduous Broadleaf Forest",fontsize=9)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()
#######################################################################################################
# next run EMS and HEM ML daily 
EMS
# EMS comparison periods: 1993-2008 and 2008-2023.
# HEM comparison periods: 2005-2013 and 2014-2023.
# HWA periods: 2006-2013 pre/during HWA and 2014-2022 post-HWA.

#######################################################################################################

# test periods here for ET

df=EMS_daily
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
#mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
#mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2008)

mask = (df['water_year'] >= 2009) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
daily_ha1=df
df= df.groupby(['water_year']).sum()
df=df.reset_index()
#annual_em1=df
annual_em2=df

#####################################################################################################33


## check if slopes are different for two periods

df=annual_em1

df['Year'] = 'Period1'

df2= annual_em2 

df2['Year'] = 'Period2'

df3 = pd.concat([df, df2], ignore_index=True)


# Fit the linear model with interaction
model = smf.ols('ET ~ water_year * Year', data=df3).fit()

# Print the summary
print(model.summary())

#####################################################################################

df=annual_hem1

df['Year'] = 'Period1'

df2= annual_hem2

df2['Year'] = 'Period2'

df3 = pd.concat([df, df2], ignore_index=True)


# Fit the linear model with interaction
model = smf.ols('ET ~ water_year * Year', data=df3).fit()

# Print the summary
print(model.summary())






###########################################################################################
df_e=annual_ha1

#hg.pettitt_test(df_e.ET)

df=annual_ha1
x1=df['water_year']
y1=df['ET'].astype(float)
#(x1, y1)
mk.original_test(y1)

#Mann_Kendall_Test(trend='decreasing', h=True, p=0.017, Tau=-0.45, slope=-9.21

((4.11 * 16)/ 546.9446) *100  # 1993-2008    # -31.5

# 1993-2008    # -31.5
# 2008-2023    # 12.02


# ammara MD

#HEM_ML_fill_2023
HEM_daily=df



df=HEM_daily
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
#mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2023)
#mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2013)
#mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2014)
#mask = (df['water_year'] >= 2013) & (df['water_year'] <= 2023)
mask = (df['water_year'] >= 2015) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
daily_ha2=df
df= df.groupby(['water_year']).sum()
df=df.reset_index()
#annual_hem1=df
annual_hem2=df

#df_h=annual_ha2
##################################################################################################

278.9-424
## dont run beyond this point if making graph

df=annual_ha2
x1=df['water_year']
y1=df['ET'].astype(float)
#(x1, y1)
mk.original_test(y1)


# Mann-Kendall output: decreasing trend, p=0.029, tau=-0.612,
# slope=-26.268899162410897.

((3.33 * 11)/ 424.74) *100
# 2005-2013  percent change -60.65
# 2013-2023  percent change 8.6

# Mann-Kendall output repeated for comparison: decreasing trend, p=0.029,
# tau=-0.612, slope=-26.268899162410897.
# 2013-2023  percent change -60.65
#####################################################################################################

annual_ha1=df
df_e=annual_ha1
#####################################################################################################

## bring HEM 

#df=HEM_daily_ET

#df=HEM_daily
df.isnull().values.any()
df=df.groupby(['water_year']).sum()
#df=df.groupby(['Year']).sum()
df=df.reset_index()


#mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2013)
mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2014)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_hem_sum=df


x1=df['water_year']
y1=df['ET'].astype(float)
#y1=df_e_y['ET'].astype(float)
pearsonr(x1, y1)
mk.original_test(y1)


((-26.3 * 9)/389.8) *100


#df=annual_hem_sum
mask = (df['Year'] >= 2014) & (df['Year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))


df_e_y=df


mk.original_test(y1)


# EMS periods: 1993-2008 and 2008-2023; change point evaluated near 2010.
# HEM periods: 2005-2014 and 2015-2023.
# HWA periods: 2006-2014 pre/during HWA and 2015-2023 post-HWA.

#change point 2015

#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(df_e['water_year'].head(17), df_e['ET'].head(17),color="blue",label="EMS (1993-2008), p=1.7*10$^{-2}$, slope=-9.2",linestyle='-', marker='*',markersize=7)
plt.plot(df_e['water_year'].tail(15), df_e['ET'].tail(15),color="cornflowerblue",label="EMS (2008-2023), p=6.9*10$^{-1}$, slope=4.1",linestyle='-', marker='*',markersize=7)

plt.plot(df_h['water_year'].head(9), df_h['ET'].head(9),color="darkgreen",label="HEM (2005-2013), p=2.9*10$^{-2}$, slope=-26",linestyle='-', marker='.',markersize=8)
plt.plot(df_h['water_year'].tail(11), df_h['ET'].tail(11),color="lime",label="HEM (2013-2023), p=7.6*10$^{-1}$, slope=3.3",linestyle='-', marker='.',markersize=8)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
#ax.legend(loc='upper left', fontsize = '6',labelspacing=0.0005)
ax.legend(loc='upper left', fontsize = '6',labelcolor='linecolor',labelspacing=0.0005)
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Total annual ET (mm)')  # we already handled the x-label with ax1

ax.set_ylim(250,800)

z1 = np.polyfit(df_e['water_year'].head(16), df_e['ET'].head(16), 1)
p1 = np.poly1d(z1)
plt.plot(df_e['water_year'].head(16), p1(df_e['water_year'].head(16)),color='blue')


z2 = np.polyfit(df_e['water_year'].tail(16), df_e['ET'].tail(16), 1)
p2 = np.poly1d(z2)
plt.plot(df_e['water_year'].tail(16), p2(df_e['water_year'].tail(16)),color='cornflowerblue')


z3 = np.polyfit(df_h['water_year'].head(9), df_h['ET'].head(9), 1)
p3 = np.poly1d(z3)
plt.plot(df_h['water_year'].head(9), p3(df_h['water_year'].head(9)),color='darkgreen')


z4 = np.polyfit(df_h['water_year'].tail(11), df_h['ET'].tail(11), 1)
p4 = np.poly1d(z4)
plt.plot(df_h['water_year'].tail(11), p4(df_h['water_year'].tail(11)),color='lime')


ax.set_title("Harvard Forest, EMS Tower (US-Ha1), HEM Tower (US-Ha2)",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
#fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
#plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

####################################################################################################

## annual 
### growing season
from scipy import stats
###################################################################################################

df=EMS_daily

df['Grow_sea'] = df.apply(Grow_sea, axis = 1)

mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))   


df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
#mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
#mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2008)
mask = (df['water_year'] >= 2009) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
daily_ha1=df
df= df.groupby(['water_year']).sum()
df=df.reset_index()
#ann_h1=df
#ann1_h1=df
#ann2_h1=df
#ann_gh1=df  #GS ET not changing
#ann_ngh1=df NGS ET is changing 

df=ann2_h1
x1=df['water_year']
y1=df['ET'].astype(float)
#(x1, y1)
mk.original_test(y1)

stats.ttest_ind(ann1_h1, ann2_h1)

# EMS overall no ET trend , split in two parts: both are significantly different 
# EMS annual GS no trend, 
# EMS annul NGS increasing 
# bar plot 
#ann1_h1=df
#ann2_h1=df

#time series
#ann_gh1=df
#ann_ngh1
###################################################################################################


df=HEM_daily

df['Grow_sea'] = df.apply(Grow_sea, axis = 1)

mask = (df['Grow_sea'] == 'NGS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))   


df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
#mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2023)
#mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2014)
mask = (df['water_year'] >= 2015) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df= df.groupby(['water_year']).sum()
df=df.reset_index()
#ann_h2=df
#ann1_h2=df
#ann2_h2=df
#ann_gh2=df  #GS ET not changing
ann_ngh2=df #NGS ET is changing 

df=ann_ngh2
x1=df['water_year']
y1=df['ET'].astype(float)
#(x1, y1)
mk.original_test(y1)

#stats.ttest_ind(ann1_h1, ann2_h1)

# HEM overall no ET trend , split in two parts: both are significantly different 
# HEM annual GS no trend, 
# HEM annul NGS no trend 
# bar plot 
#ann1_h2=df
#ann2_h2=df

#time series
#ann_gh2=df
#ann_ngh2

###############################################################################################

my_dict1={'EMS (1993-2008)':ann1_h1.ET}
my_dict2={'EMS (2009-2023)':ann2_h1.ET}
my_dict3={'HEM (2005-2014)':ann1_h2.ET}
my_dict4={'HEM (2015-2023)':ann2_h2.ET}


#my_dict2={'gs_iPM':after_iPM, 'gs_FG':after_FG}

#ax.set_xticklabels(['gs_iPM (1992-2008)','gs_FG (1992-2008)','gs_iPM (2009-2023)','gs_FG (2009-2023)'])
c1="C0"
c2="C1"
c3="C2"
c4="C3"




plt.rcParams['figure.figsize'] = (5,4)
fig, ax = plt.subplots()

bp1=ax.boxplot(my_dict1.values(),sym='.',positions =[1],notch=False, widths=3, 
patch_artist=True,whiskerprops = dict(linestyle='--'
,color='mediumblue', linewidth=0.5),boxprops=dict(facecolor='blue',edgecolor='mediumblue'))

bp2=ax.boxplot(my_dict2.values(),sym='.',positions = [4],notch=False, widths=3, 
 patch_artist=True, boxprops=dict(facecolor='cornflowerblue',edgecolor='royalblue'),whiskerprops = dict(linestyle='--'
               ,color='royalblue' , linewidth=0.5))

plt.setp(bp1['medians'][0], color='black')
plt.setp(bp2['medians'][0], color='royalblue')

ax.set_title("Harvard Forest, US-Ha1: EMS Tower, US-Ha2: HEM Tower",fontsize=10)

bp3=ax.boxplot(my_dict3.values(),sym='.',positions =[8],notch=False, widths=3, 
patch_artist=True,whiskerprops = dict(linestyle='--'
,color='forestgreen', linewidth=0.5),boxprops=dict(facecolor='darkgreen',edgecolor='forestgreen'))

bp4=ax.boxplot(my_dict4.values(),sym='.',positions = [11],notch=False, widths=3, 
 patch_artist=True, boxprops=dict(facecolor='lime',edgecolor='mediumseagreen'),whiskerprops = dict(linestyle='--'
               ,color='mediumseagreen' , linewidth=0.5))
#ax.set_ylim(250,550)

plt.setp(bp3['medians'][0], color='black')
plt.setp(bp4['medians'][0], color='mediumseagreen')


plt.xticks([0, 4, 8,12], ['1993-2008','2009-2023','2005-2014','2015-2023'])

ax.text(0.35, 0.75, 'p=0.002',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=10)

ax.text(0.82, 0.50, 'p=0.036',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=10)

ax.legend([bp1["boxes"][0], bp2["boxes"][0],bp3["boxes"][0],bp4["boxes"][0]], ['EMS','EMS', 'HEM','HEM'], loc='upper left',prop={'size': 8})
#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Pre-during HWA:  2006-2013', 'Post HWA:            2014-2022'], loc='upper left',prop={'size': 5})

ax.set_ylim(250,800)
ax.set_ylabel('Total annual ET (mm)')  # we already handled the x-label with ax1
ax.tick_params(axis = 'y', which = 'major', labelsize = 12,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 12,direction='in')
#ax.set_ylim(-0.5,1.5)
#ax.set_title("Training Sites, Potatoes: (2018-2020), n=309",fontsize=6)
#ax.set_title("whole year",fontsize=7)
ax.yaxis.label.set_size(10)
ax.yaxis.label.set_size(10)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
#ax.locator_params(axis='y', nbins=4)
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


#####################################################################################################
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")
df=pd.read_csv("daily_ems_pet_ET.csv") 
daily_ems_pet_ET=df

df=daily_ems_pet_ET
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df=df
back_daily_ems=df

df=back_daily_ems
df=df.groupby(['water_year']).sum()
#df=df.groupby(['water_year']).mean()
df=df.reset_index()
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_ems_sum=df

#
###PET graphs
############################################################################################

df=annual_ems_sum

df_e_y=df

x1=df_e_y['water_year']
y1=df_e_y['PET_ems'].astype(float)
#y1=df_e_y['ET'].astype(float)
#pearsonr(x1, y1)
mk.original_test(y1)


#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (2.8, 4.5)
fig, ax = plt.subplots()
#plt.plot(df_e_y['water_year'], df_e_y['PET_ems'],color="blue"   ,label="PET, p=2.2*10$^{-3}$, Tau=0.33, slope=4.4",linestyle='--', marker='o',markersize=5)
#plt.plot(df_e_y['water_year'], df_e_y['ET'],color="lightcoral",label="ET,  p=0.20, Tau=0.17, slope=2.5",linestyle='-', marker='*',markersize=7)

plt.scatter(df_e_y['water_year'], df_e_y['PET_ems'],color="lightcoral"  ,marker='*' , s=40,label="PET, p= 0.002, slope= 4.8")
plt.scatter(df_e_y['water_year'], df_e_y['ET'],color="blue", marker='3', s=70, label="ET,   p= 0.17,   slope= 2.6")

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '8',labelspacing=0.0005,labelcolor='linecolor')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Total annual PET or ET (mm)')  # we already handled the x-label with ax1

z1 = np.polyfit(df_e_y['water_year'], df_e_y['PET_ems'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='lightcoral',linestyle='-')

ax.set_ylim(300,1100)
z1 = np.polyfit(df_e_y['water_year'], df_e_y['ET'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='blue',linestyle='--')

ax.set_title("US-Ha1, EMS Tower, Annual",fontsize=8)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
#ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
#ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
#ax.yaxis.set_major_locator(mdates.DayLocator(interval=50))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#351.2-779.8

#################################################################################################
# growing


df=back_daily_ems
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))   
GS=df
#NGS=df


df=GS

#df=NGS
df=df.groupby(['water_year']).sum()
#df=df.groupby(['water_year']).mean()
df=df.reset_index()
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_ems_sum=df

#GS: PET 280.08-592.03
#   ET  229.6-599.9
##NGS:  PET  241.5-430.5
#NGS ET  90.9-241.5
############################################################################################

df=annual_ems_sum

df_e_y=df

x1=df_e_y['water_year']
y1=df_e_y['PET_ems'].astype(float)
#y1=df_e_y['ET'].astype(float)
pearsonr(x1, y1)
mk.original_test(y1)



#Sen's slope * length of period)/mean *100
#((4.84 * 31)/770.78) *100

#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (2.8, 4.5)
fig, ax = plt.subplots()

plt.scatter(df_e_y['water_year'], df_e_y['PET_ems'],color="lightcoral"  ,marker='X' , s=30,label="PET, p= 0.048, slope= 2.63")
plt.scatter(df_e_y['water_year'], df_e_y['ET'],color="blue", marker='H' , s=30, label="ET,   p= 0.57,   slope= 0.55")

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '8',labelspacing=0.0005,labelcolor='linecolor')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Annual PET or ET (mm)')  # we already handled the x-label with ax1

z1 = np.polyfit(df_e_y['water_year'], df_e_y['PET_ems'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='lightcoral',linestyle='-')

ax.set_ylim(200,700)
z1 = np.polyfit(df_e_y['water_year'], df_e_y['ET'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='blue',linestyle='--')

ax.set_title("US-Ha1, EMS Tower, Growing Season",fontsize=8)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


#################################################################################################

### nongrowing season

df=back_daily_ems
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))   
GS=df


df=GS
df=df.groupby(['water_year']).sum()
#df=df.groupby(['water_year']).mean()
df=df.reset_index()
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_ems_sum=df
#
###PET graphs
############################################################################################

df=annual_ems_sum

df_e_y=df

x1=df_e_y['water_year']
#y1=df_e_y['PET_ems'].astype(float)
y1=df_e_y['ET'].astype(float)
pearsonr(x1, y1)
mk.original_test(y1)





#Sen's slope * length of period)/mean *100
ET=((2.025 * 31)/156.38) *100 # 44%

#Sen's slope * length of period)/mean *100
PET=((2.88 * 31)/340.3) *100  # 35%


# Generic Sen-slope percentage-change calculation was retained as a note:
# end_value = start_value + sen_slope * (len(time_series) - 1)
# percentage_change = ((end_value - start_value) / start_value) * 100



#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (2.8, 4.5)
fig, ax = plt.subplots()

plt.scatter(df_e_y['water_year'], df_e_y['PET_ems'],color="lightcoral"  ,marker='d' ,s=40, label="PET, p= 0.004, slope= 2.88")
plt.scatter(df_e_y['water_year'], df_e_y['ET'],color="blue", marker='v' ,s=40, label="ET,   p= 0.019, slope= 2.02")

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '8',labelspacing=0.0005,labelcolor='linecolor')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Annual PET or ET (mm)')  # we already handled the x-label with ax1

z1 = np.polyfit(df_e_y['water_year'], df_e_y['PET_ems'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='lightcoral',linestyle='-')

ax.set_ylim(60,500)
z1 = np.polyfit(df_e_y['water_year'], df_e_y['ET'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='blue',linestyle='--')

ax.set_title("US-Ha1, EMS site, Non Growing Season",fontsize=8)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()
###########################################################################################################33333
# march fig
## fraction of growing and nongrowin



GS["ET_f"]=GS['ET']/(GS['ET']+NGS["ET"])

NGS["ET_f"]=NGS['ET']/(GS['ET']+NGS["ET"])



x1=GS['water_year']
y1=GS['ET_f'].astype(float)
mk.original_test(y1)  # significant decareasing trend for growing season


x1=NGS['water_year']
y1=NGS['ET_f'].astype(float)
mk.original_test(y1)   # # significant increasing ET trend for non growing season


### two axis
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(GS['water_year'], GS['ET_f'],color="deeppink",label="EMS (Growing Season)",linestyle='--', linewidth=0.5, marker='1',markersize=10)
#ax.tick_params(axis='y', labelcolor='black')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 

z1 = np.polyfit(GS['water_year'], GS['ET_f'], 1)
p1 = np.poly1d(z1)
plt.plot(GS['water_year'], p1(GS['water_year']),color='deeppink',linestyle='--')


ax2 = ax.twinx() 
#plt.plot(NGS['water_year'], NGS['ET_f'],color="red",label="EMS (Non-Growing Season)",linestyle='--', marker='*',markersize=7)
plt.plot(NGS['water_year'], NGS['ET_f'],color="orange",label="EMS (Non-Growing Season)",linestyle='--', linewidth=0.5, marker='o',markersize=6)
z2 = np.polyfit(NGS['water_year'], NGS['ET_f'], 1)
p2 = np.poly1d(z2)
plt.plot(NGS['water_year'], p2(NGS['water_year']),color='orange',linestyle='--')

ax2.tick_params(axis='y', labelcolor='orange')
ax.xaxis.label.set_size(12) # 

ax2.spines['right'].set_color('orange') 
ax.locator_params(axis='y', nbins=6)
ax.set_ylabel('Growing Season', color='hotpink')  # we already handled the x-label with ax1

ax2.set_ylabel('Non-Growing Season', color='orange')  # we already handled the x-label with ax1
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1

ax.yaxis.label.set_size(14)
ax2.yaxis.label.set_size(14)
#ax2.locator_params(axis='y', nbins=6)
ax.set_title("EMS Tower (US-Ha1), Fraction of Annual ET",fontsize=10)
#ax.set_title("HEM Tower (US-Ha2), Fraction of Annual ET",fontsize=10)

#ax.set_title("EMS Tower (US-Ha1), ET Fraction",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
#ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
#fig.autofmt_xdate() 
ax.tick_params(axis='y', labelcolor='hotpink')
ax.spines['left'].set_color('hotpink')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
#plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#####################################################################################################################
### same axis 


plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(GS['water_year'], GS['ET_f'],color="deeppink",label="Growing Season",linestyle='--', linewidth=0.5, marker='1',markersize=10)
#ax.tick_params(axis='y', labelcolor='black')
ax.xaxis.label.set_size(12)
ax.yaxis.label.set_size(12) 

z1 = np.polyfit(GS['water_year'], GS['ET_f'], 1)
p1 = np.poly1d(z1)
plt.plot(GS['water_year'], p1(GS['water_year']),color='deeppink',linestyle='--')


#plt.plot(NGS['water_year'], NGS['ET_f'],color="red",label="EMS (Non-Growing Season)",linestyle='--', marker='*',markersize=7)
plt.plot(NGS['water_year'], NGS['ET_f'],color="orange",label="Non-Growing Season",linestyle='--', linewidth=0.5, marker='o',markersize=6)
z2 = np.polyfit(NGS['water_year'], NGS['ET_f'], 1)
p2 = np.poly1d(z2)
plt.plot(NGS['water_year'], p2(NGS['water_year']),color='orange',linestyle='--')

ax.legend()
#ax.legend(loc='center', fontsize = '8',labelspacing=0.0005,labelcolor='linecolor')
ax.legend(loc='center', fontsize = '10',labelcolor='linecolor')

ax.xaxis.label.set_size(12) # 

ax.locator_params(axis='y', nbins=6)
ax.set_ylabel(' Fraction of Annual ET', color='black')  # we already handled the x-label with ax1

ax.set_xlabel('Water Year')  # we already handled the x-label with ax1

ax.yaxis.label.set_size(14)
ax.set_title("EMS Tower (US-Ha1)",fontsize=12)
#ax.set_title("HEM Tower (US-Ha2), Fraction of Annual ET",fontsize=10)

#ax.set_title("EMS Tower (US-Ha1), ET Fraction",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
#ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 12,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =12, width=1)
#fig.autofmt_xdate() 
ax.tick_params(axis='y', labelcolor='black')
ax.spines['left'].set_color('black')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
#plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()












#####################################################################################################################

### hemlock fraction do calculations below

## two axis graph
x1=GS['water_year']
y1=GS['ET_f'].astype(float)
mk.original_test(y1)  # significant decareasing trend for growing season


x1=NGS['water_year']
y1=NGS['ET_f'].astype(float)
mk.original_test(y1)   # # significant increasing ET trend for non growing season


plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(GS['water_year'], GS['ET_f'],color="deeppink",label="EMS (Growing Season)",linestyle='--', linewidth=0.5, marker='1',markersize=10)
#ax.tick_params(axis='y', labelcolor='black')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 

z1 = np.polyfit(GS['water_year'], GS['ET_f'], 1)
p1 = np.poly1d(z1)
plt.plot(GS['water_year'], p1(GS['water_year']),color='deeppink',linestyle='--')


ax2 = ax.twinx() 
#plt.plot(NGS['water_year'], NGS['ET_f'],color="red",label="EMS (Non-Growing Season)",linestyle='--', marker='*',markersize=7)
plt.plot(NGS['water_year'], NGS['ET_f'],color="orange",label="EMS (Non-Growing Season)",linestyle='--', linewidth=0.5, marker='o',markersize=6)
z2 = np.polyfit(NGS['water_year'], NGS['ET_f'], 1)
p2 = np.poly1d(z2)
plt.plot(NGS['water_year'], p2(NGS['water_year']),color='orange',linestyle='--')

ax2.tick_params(axis='y', labelcolor='orange')
ax.xaxis.label.set_size(12) # 

ax2.spines['right'].set_color('orange') 
ax.locator_params(axis='y', nbins=6)
ax.set_ylabel('Growing Season', color='hotpink')  # we already handled the x-label with ax1

ax2.set_ylabel('Non-Growing Season', color='orange')  # we already handled the x-label with ax1
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1

ax.yaxis.label.set_size(14)
ax2.yaxis.label.set_size(14)
ax2.locator_params(axis='y', nbins=6)
ax.set_title("HEM Tower (US-Ha2), Fraction of Annual ET",fontsize=10)
#ax.set_title("EMS Tower (US-Ha1), ET Fraction",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
#ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
#fig.autofmt_xdate() 
ax.tick_params(axis='y', labelcolor='hotpink')
ax.spines['left'].set_color('hotpink')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.tick_params(axis='x', rotation=0)
#plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2g}'))
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

########################################################################################################################

## same axis

GS["ET_f"]=GS['ET']/(GS['ET']+NGS["ET"])

NGS["ET_f"]=NGS['ET']/(GS['ET']+NGS["ET"])


plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(GS['water_year'], GS['ET_f'],color="deeppink",label="Growing Season",linestyle='--', linewidth=0.5, marker='1',markersize=10)
#ax.tick_params(axis='y', labelcolor='black')
ax.xaxis.label.set_size(12)
ax.yaxis.label.set_size(12) 

z1 = np.polyfit(GS['water_year'], GS['ET_f'], 1)
p1 = np.poly1d(z1)
plt.plot(GS['water_year'], p1(GS['water_year']),color='deeppink',linestyle='--')


#plt.plot(NGS['water_year'], NGS['ET_f'],color="red",label="EMS (Non-Growing Season)",linestyle='--', marker='*',markersize=7)
plt.plot(NGS['water_year'], NGS['ET_f'],color="orange",label="Non-Growing Season",linestyle='--', linewidth=0.5, marker='o',markersize=6)
z2 = np.polyfit(NGS['water_year'], NGS['ET_f'], 1)
p2 = np.poly1d(z2)
plt.plot(NGS['water_year'], p2(NGS['water_year']),color='orange',linestyle='--')

ax.legend()
#ax.legend(loc='center', fontsize = '8',labelspacing=0.0005,labelcolor='linecolor')
ax.legend(loc='center', fontsize = '10',labelcolor='linecolor')

ax.xaxis.label.set_size(12) # 

ax.locator_params(axis='y', nbins=3)
ax.set_ylabel(' Fraction of Annual ET', color='black')  # we already handled the x-label with ax1

ax.set_xlabel('Water Year')  # we already handled the x-label with ax1

ax.yaxis.label.set_size(14)
ax.set_title("HEM Tower (US-Ha2)",fontsize=12)
#ax.set_title("HEM Tower (US-Ha2), Fraction of Annual ET",fontsize=10)

#ax.set_title("EMS Tower (US-Ha1), ET Fraction",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
#ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 12,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =12, width=1)
#fig.autofmt_xdate() 
ax.tick_params(axis='y', labelcolor='black')
ax.spines['left'].set_color('black')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.locator_params(axis='y', nbins=7)  
#ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.tick_params(axis='x', rotation=0)
#plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#####################################################################################################################

## HEM PET, ET

#################################################################################################################
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")
df=pd.read_csv("daily_hem_pet_ET.csv") 
daily_hem_pet_ET=df

df=daily_hem_pet_ET
mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df=df
back_daily_hem=df


### annual total
df=back_daily_hem
df=df.groupby(['water_year']).sum()
#df=df.groupby(['water_year']).mean()
df=df.reset_index()
mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_hem_sum=df


df=back_daily_hem
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))  
df=df.groupby(['water_year']).sum()
#df=df.groupby(['water_year']).mean()
df=df.reset_index()
mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_hem_sum=df 
GS=df


#NGS=df

df=back_daily_hem
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'NGS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))  

df=df.groupby(['water_year']).sum()
#df=df.groupby(['water_year']).mean()
df=df.reset_index()
mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_hem_sum=df
NGS=df


#
###PET graphs
#####################################################################################################################
df=annual_hem_sum

df_e_y=df

x1=df_e_y['water_year']
#y1=df_e_y['PET_hem'].astype(float)
y1=df_e_y['ET'].astype(float)
#pearsonr(x1, y1)
mk.original_test(y1)




#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (2.8, 4.5)
fig, ax = plt.subplots()
#plt.plot(df_e_y['water_year'], df_e_y['PET_ems'],color="blue"   ,label="PET, p=2.2*10$^{-3}$, Tau=0.33, slope=4.4",linestyle='--', marker='o',markersize=5)
#plt.plot(df_e_y['water_year'], df_e_y['ET'],color="lightcoral",label="ET,  p=0.20, Tau=0.17, slope=2.5",linestyle='-', marker='*',markersize=7)

plt.scatter(df_e_y['water_year'], df_e_y['PET_hem'],color="lightcoral"  ,marker='*' , s=40,label="PET, p= 0.48, slope= -1.98")
plt.scatter(df_e_y['water_year'], df_e_y['ET'],color="blue", marker='3', s=70, label="ET,   p= 0.53, slope=  2.30")

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '8',labelspacing=0.0005,labelcolor='linecolor')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Total annual PET or ET (mm)')  # we already handled the x-label with ax1

z1 = np.polyfit(df_e_y['water_year'], df_e_y['PET_hem'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='lightcoral',linestyle='-')
#ax.set_ylim(300,700)
ax.set_xlim(2004,2025)

#ax.set_ylim(300,1100)
z1 = np.polyfit(df_e_y['water_year'], df_e_y['ET'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='blue',linestyle='--')

ax.set_title("US-Ha2, HEM Tower, Annual",fontsize=8)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
#ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
#ax.yaxis.set_major_locator(mdates.DayLocator(interval=50))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#351.2-779.8

#################################################################################################
# growing and non growing 


df=back_daily_hem
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))   
GS=df
#NGS=df


df=GS

#df=NGS
df=df.groupby(['water_year']).sum()
#df=df.groupby(['water_year']).mean()
df=df.reset_index()
mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_hem_sum=df

#GS: PET 280.08-592.03
#   ET  229.6-599.9
##NGS:  PET  241.5-430.5
#NGS ET  90.9-241.5
############################################################################################

df=annual_hem_sum

df_e_y=df

x1=df_e_y['water_year']
y1=df_e_y['PET_hem'].astype(float)
#y1=df_e_y['ET'].astype(float)
pearsonr(x1, y1)
mk.original_test(y1)



#Sen's slope * length of period)/mean *100
#((4.84 * 31)/770.78) *100

#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (2.8, 4.5)
fig, ax = plt.subplots()

plt.scatter(df_e_y['water_year'], df_e_y['PET_hem'],color="lightcoral"  ,marker='X' , s=30,label="PET, p= 0.67, slope= 1.64")
plt.scatter(df_e_y['water_year'], df_e_y['ET'],color="blue", marker='H' , s=30, label="ET,   p= 0.44, slope= 1.79")

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '8',labelspacing=0.0005,labelcolor='linecolor')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Annual PET or ET (mm)')  # we already handled the x-label with ax1

z1 = np.polyfit(df_e_y['water_year'], df_e_y['PET_hem'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='lightcoral',linestyle='-')

#ax.set_ylim(200,700)
z1 = np.polyfit(df_e_y['water_year'], df_e_y['ET'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='blue',linestyle='--')
ax.set_xlim(2003,2025)
ax.set_title("US-Ha2, HEM Tower, Growing Season",fontsize=8)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


#################################################################################################
df=back_daily_hem
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'NGS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))   

NGS=df

df=NGS
df=df.groupby(['water_year']).sum()
#df=df.groupby(['water_year']).mean()
df=df.reset_index()
mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_hem_sum=df

#
###PET graphs
############################################################################################

df=annual_hem_sum

df_e_y=df

x1=df_e_y['water_year']
#y1=df_e_y['PET_hem'].astype(float)
y1=df_e_y['ET'].astype(float)
pearsonr(x1, y1)
mk.original_test(y1)



#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (2.8, 4.5)
fig, ax = plt.subplots()

plt.scatter(df_e_y['water_year'], df_e_y['PET_hem'],color="lightcoral"  ,marker='d' ,s=40, label="PET, p= 0.20,  slope= -1.70")
plt.scatter(df_e_y['water_year'], df_e_y['ET'],color="blue", marker='v' ,s=40, label="ET,   p= 0.53,  slope= 0.95")

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '8',labelspacing=0.0005,labelcolor='linecolor')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Annual PET or ET (mm)')  # we already handled the x-label with ax1

z1 = np.polyfit(df_e_y['water_year'], df_e_y['PET_hem'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='lightcoral',linestyle='-')

#ax.set_ylim(60,500)
z1 = np.polyfit(df_e_y['water_year'], df_e_y['ET'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='blue',linestyle='--')
ax.set_xlim(2003,2025)
ax.set_title("US-Ha2, HEM Tower, Non Growing Season",fontsize=8)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#####################################################################################################




x1=GS['water_year']
y1=GS['ET_f'].astype(float)
mk.original_test(y1)  # significant decareasing trend for growing season


x1=NGS['water_year']
y1=NGS['ET_f'].astype(float)
mk.original_test(y1)   # # significant increasing ET trend for non growing season



plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(GS['water_year'], GS['ET_f'],color="deeppink",label="EMS (Growing Season)",linestyle='--', linewidth=0.5, marker='1',markersize=10)
#ax.tick_params(axis='y', labelcolor='black')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 

z1 = np.polyfit(GS['water_year'], GS['ET_f'], 1)
p1 = np.poly1d(z1)
plt.plot(GS['water_year'], p1(GS['water_year']),color='deeppink',linestyle='--')


ax2 = ax.twinx() 
#plt.plot(NGS['water_year'], NGS['ET_f'],color="red",label="EMS (Non-Growing Season)",linestyle='--', marker='*',markersize=7)
plt.plot(NGS['water_year'], NGS['ET_f'],color="orange",label="EMS (Non-Growing Season)",linestyle='--', linewidth=0.5, marker='o',markersize=6)
z2 = np.polyfit(NGS['water_year'], NGS['ET_f'], 1)
p2 = np.poly1d(z2)
plt.plot(NGS['water_year'], p2(NGS['water_year']),color='orange',linestyle='--')

ax2.tick_params(axis='y', labelcolor='orange')
ax.xaxis.label.set_size(12) # 

ax2.spines['right'].set_color('orange') 
ax.locator_params(axis='y', nbins=6)
ax.set_ylabel('Growing Season', color='hotpink')  # we already handled the x-label with ax1

ax2.set_ylabel('Non-Growing Season', color='orange')  # we already handled the x-label with ax1
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1

ax.yaxis.label.set_size(14)
ax2.yaxis.label.set_size(14)
ax2.locator_params(axis='y', nbins=6)
ax.set_title("EMS Tower (US-Ha1), Fraction of Annual ET",fontsize=10)
#ax.set_title("EMS Tower (US-Ha1), ET Fraction",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
#ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
#fig.autofmt_xdate() 
ax.tick_params(axis='y', labelcolor='hotpink')
ax.spines['left'].set_color('hotpink')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
#plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

























###############################################################################################################3
# canopy conductance 

os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")
gs_ems=pd.read_csv("gs_FG_all.csv") 
gs_hem=pd.read_csv("gs_FG_all_hem.csv")

#########################################################################################################

#df=gs_ems
#std_e=df.groupby(df["water_year"])['gs_FG'].std()

#std_e=std_e.iloc[1:]

df=gs_ems
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))

df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'NGS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))


#### decide to use mean or median 

df= df.groupby(['water_year']).median()
df=df.reset_index()

df_ems=df


#gs_e=df        # annual
gs_e_gs=df     # GS  #0.45-0.83
gs_e_ngs=df    # NGS #0.33-0.49

#df=gs_hem
df=gs_hem
mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2023) #original mask

#mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2014) #original mask
#mask = (df['water_year'] >= 2015) & (df['water_year'] <= 2023) #original mask
# 2005-2014
#

df = df.loc[mask]
df.index = np.arange(0, len(df))


df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
#mask = (df['Grow_sea'] == 'GS') 
mask = (df['Grow_sea'] == 'NGS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))


df= df.groupby(['water_year']).median()
df=df.reset_index()
df_hem=df

from scipy import stats
t_stat, p_value = stats.ttest_ind(df_ems['gs_FG'], df_hem['gs_FG'])


##################################################################################################
## t test to check if hem gs is different than ems 

#############################################################################################

#((3.33 * 31)/ 424.74) *100

#gs_h=df        # annual   0.23-0.42

y1=df['gs_FG'].astype(float)
#pearsonr(x1, y1)
hg.pettitt_test(y1)





gs_h_gs=df     # GS  0.24-0.47
gs_h_ngs=df    # NGS 0.19-0.6












#df=gs_h
df=gs_h_gs
#df=gs_h_ngs

x1=df['water_year']
y1=df['gs_FG'].astype(float)
#pearsonr(x1, y1)
mk.original_test(y1)



df=gs_h_ngs
x1=df['water_year']
y1=df['gs_FG'].astype(float)
pearsonr(x1, y1)
mk.original_test(y1)

#########################################################################################################

################################################################################################

plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(gs_e['water_year'], gs_e['gs_FG'],markersize=7,color="red", label="Annual,          p=0.007,  slope=-0.004",linestyle='--', marker='o')
plt.scatter(gs_e_gs['water_year'], gs_e_gs['gs_FG'],s=70,color="lightcoral", label="Growing,        p=0.002, slope=-0.006",linestyle='--', marker='2')
plt.scatter(gs_e_ngs['water_year'],gs_e_ngs['gs_FG'],s=70,color="lightsalmon", label="Non Growing, p=0.15,   slope= 0.001",linestyle='--', marker='*')

ax.tick_params(axis='y', labelcolor='black')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('$g_{c}$ ($mol$ $m^{-2}$ $s^{-1}$)')  # we already handled the x-label with ax1

z1 = np.polyfit(gs_e['water_year'], gs_e['gs_FG'], 1)
p1 = np.poly1d(z1)
plt.plot(gs_e['water_year'], p1(gs_e['water_year']),color='red')

z2 = np.polyfit(gs_e_gs['water_year'], gs_e_gs['gs_FG'], 1)
p2 = np.poly1d(z2)
plt.plot(gs_e_gs['water_year'], p2(gs_e_gs['water_year']),color='lightcoral', linestyle='--')

z3 = np.polyfit(gs_e_ngs['water_year'], gs_e_ngs['gs_FG'], 1)
p3 = np.poly1d(z3)
plt.plot(gs_e_ngs['water_year'], p3(gs_e_ngs['water_year']),color='lightsalmon', linestyle='--')

ax.legend()
ax.legend(loc='upper right', fontsize = '8',labelspacing=0.0005)
ax.set_title("Harvard Forest, US-Ha1, Median Canopy Conductance ($g_{c}$)",fontsize=10)

#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 10)
ax.tick_params(axis = 'y', which = 'major', labelsize =10)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

########################################################################################################
## not use

plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(gs_e['water_year'], gs_e['gs_FG'],markersize=7,color="red", label="EMS: annual,          p=0.007,  slope=-0.004",linestyle='--', marker='o')
plt.scatter(gs_e_gs['water_year'], gs_e_gs['gs_FG'],s=70,color="lightcoral", label="EMS: Growing,        p=0.002, slope=-0.006",linestyle='--', marker='2')
plt.scatter(gs_e_ngs['water_year'],gs_e_ngs['gs_FG'],s=70,color="lightsalmon", label="EMS: Non Growing, p=0.15,   slope= 0.001",linestyle='--', marker='*')

ax.tick_params(axis='y', labelcolor='black')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('$G_{c}$ ($mol$ $m^{-2}$ $s^{-1}$)')  # we already handled the x-label with ax1

z1 = np.polyfit(gs_e['water_year'], gs_e['gs_FG'], 1)
p1 = np.poly1d(z1)
plt.plot(gs_e['water_year'], p1(gs_e['water_year']),color='red')

z2 = np.polyfit(gs_e_gs['water_year'], gs_e_gs['gs_FG'], 1)
p2 = np.poly1d(z2)
plt.plot(gs_e_gs['water_year'], p2(gs_e_gs['water_year']),color='lightcoral', linestyle='--')

z3 = np.polyfit(gs_e_ngs['water_year'], gs_e_ngs['gs_FG'], 1)
p3 = np.poly1d(z3)
plt.plot(gs_e_ngs['water_year'], p3(gs_e_ngs['water_year']),color='lightsalmon', linestyle='--')

#l1 = plt.legend([p1, p2,p3],["EMS: annual,          p=0.007, slope=-0.004", "EMS: Growing,        p=0.002, slope=-0.006", "EMS: Non Growing, p=0.15, slope= 0.001"], labelcolor='linecolor',loc='upper left',labelspacing=0.0005)
#plt.gca().add_artist(l1)

plt.bar(gs_h['water_year'], gs_h['gs_FG'],color="teal", label="HEM: annual,          p=0.54,   slope= 0.002",linestyle='--')
plt.plot(gs_h['water_year'], gs_h['gs_FG'],color="teal",marker="d", markersize="7")
plt.bar(gs_h_gs['water_year'], gs_h_gs['gs_FG'],color="aqua", label="HEM: Growing,        p=0.65,   slope= 0.002",linestyle='--')
plt.bar(gs_h_ngs['water_year'],gs_h_ngs['gs_FG'],color="aquamarine", label="HEM: Non Growing, p=0.04,   slope= 0.003",linestyle='--')

ax.legend()
ax.legend(loc='upper right', fontsize = '8',labelcolor='linecolor',labelspacing=0.0005)
ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.set_title("Harvard Forest, Median Canopy Conductance ($G_{c}$)",fontsize=12)

z4 = np.polyfit(gs_h['water_year'], gs_h['gs_FG'], 1)
p4 = np.poly1d(z4)
plt.plot(gs_h['water_year'], p4(gs_h['water_year']),color='teal')

z5 = np.polyfit(gs_h_gs['water_year'], gs_h_gs['gs_FG'], 1)
p5 = np.poly1d(z5)
plt.plot(gs_h_gs['water_year'], p5(gs_h_gs['water_year']),color='aqua', linestyle='--')

z6 = np.polyfit(gs_h_ngs['water_year'], gs_h_ngs['gs_FG'], 1)
p6 = np.poly1d(z6)
plt.plot(gs_h_ngs['water_year'], p6(gs_h_ngs['water_year']),color='aquamarine', linestyle='--')

#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 10)
ax.tick_params(axis = 'y', which = 'major', labelsize =10)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()





############################################################################################3333333
df=gs_h
df=gs_h_gs
#df=gs_h_ngs

x1=df['water_year']
y1=df['gs_FG'].astype(float)
#pearsonr(x1, y1)
mk.original_test(y1)

## if hemlock does not work use this


# --- HEM ---
df = gs_hem
mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2023)
df = df.loc[mask].copy()
df.index = np.arange(0, len(df))

df['Grow_sea'] = df.apply(Grow_sea, axis=1)

# Annual (all seasons)
gs_h = df.groupby(['water_year']).median().reset_index()

# Growing season only
gs_h_gs = df[df['Grow_sea'] == 'GS'].groupby(['water_year']).median().reset_index()

# Non-growing season only
gs_h_ngs = df[df['Grow_sea'] == 'NGS'].groupby(['water_year']).median().reset_index()


## fix it


plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()

plt.plot(gs_h['water_year'], gs_h['gs_FG'],color="teal", label="Annual,          p=0.48,   slope= 0.002",linestyle='--',marker='o', markersize=7)
plt.scatter(gs_h_gs['water_year'], gs_h_gs['gs_FG'],color="aqua", label="Growing,        p=0.62,   slope= 0.002",linestyle='--',marker='2')
plt.scatter(gs_h_ngs['water_year'],gs_h_ngs['gs_FG'],color="deepskyblue", label="Non Growing, p=0.45,   slope= 0.001",linestyle='--',marker='*')


ax.tick_params(axis='y', labelcolor='black')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('$g_{c}$ ($mol$ $m^{-2}$ $s^{-1}$)')  # we already handled the x-label with ax1

z4 = np.polyfit(gs_h['water_year'], gs_h['gs_FG'], 1)
p4 = np.poly1d(z4)
plt.plot(gs_h['water_year'], p4(gs_h['water_year']),color='teal')

z5 = np.polyfit(gs_h_gs['water_year'], gs_h_gs['gs_FG'], 1)
p5 = np.poly1d(z5)
plt.plot(gs_h_gs['water_year'], p5(gs_h_gs['water_year']),color='aqua', linestyle='--')

z6 = np.polyfit(gs_h_ngs['water_year'], gs_h_ngs['gs_FG'], 1)
p6 = np.poly1d(z6)
plt.plot(gs_h_ngs['water_year'], p6(gs_h_ngs['water_year']),color='deepskyblue', linestyle='--')


ax.legend()
ax.legend(loc='upper left', fontsize = '8',labelspacing=0.0005)
ax.set_title("Harvard Forest, US-Ha2, Median Canopy Conductance ($g_{c}$)",fontsize=10)

#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 10)
ax.tick_params(axis = 'y', which = 'major', labelsize =10)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#######################################################################################################
# ammaramd
#separate canopy

df=gs_hem
#mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2023) #original mask
#mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2014) #original mask
mask = (df['water_year'] >= 2015) & (df['water_year'] <= 2023) #original mask
# 2005-2014
#

df = df.loc[mask]
df.index = np.arange(0, len(df))

df= df.groupby(['water_year']).median()
df=df.reset_index()
#((3.33 * 31)/ 424.74) *100
#gs_h=df        # annual   0.23-0.42
y1=df['gs_FG'].astype(float)
#pearsonr(x1, y1)
#hg.pettitt_test(y1)
mk.original_test(y1)

#gs_h1=df  # (p=0.02, slope= -0.009, year=10, mean=0.292)

gs_h2=df # (p=0.029, slope=-0.009, year 9, mean=0.36)

######################################################################################################

df=gs_hem
#mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2023) #original mask
#mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2014) #original mask
mask = (df['water_year'] >= 2015) & (df['water_year'] <= 2023) #original mask
# 2005-2014
#
df = df.loc[mask]
df.index = np.arange(0, len(df))

df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))


df= df.groupby(['water_year']).median()
df=df.reset_index()
#((3.33 * 31)/ 424.74) *100
#gs_h=df        # annual   0.23-0.42
y1=df['gs_FG'].astype(float)
#pearsonr(x1, y1)
mk.original_test(y1)

#gs_h_gs1=df  # (p=0.04, slope= -0.01, year=10, mean=0.32)
#gs_h_gs2=df # (p=0.029, slope=-0.009, year 9, mean=0.36)

#########################################################################################################
# new revised code

import numpy as np
import matplotlib.pyplot as plt

# --- HEM ---
df = gs_hem
mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2023)
df = df.loc[mask].copy()
df.index = np.arange(0, len(df))

df['Grow_sea'] = df.apply(Grow_sea, axis=1)

# Annual (all seasons)
gs_h = df.groupby(['water_year']).median().reset_index()

# Growing season only
gs_h_gs = df[df['Grow_sea'] == 'GS'].groupby(['water_year']).median().reset_index()

# Non-growing season only
gs_h_ngs = df[df['Grow_sea'] == 'NGS'].groupby(['water_year']).median().reset_index()

# Hide 2023 value(s) by setting NaN (keeps the tick but no marker)
gs_h.loc[gs_h['water_year'] == 2023, 'gs_FG'] = np.nan
gs_h_gs.loc[gs_h_gs['water_year'] == 2023, 'gs_FG'] = np.nan
gs_h_ngs.loc[gs_h_ngs['water_year'] == 2023, 'gs_FG'] = np.nan

# --- Plot ---
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()

# Annual
plt.plot(gs_h['water_year'], gs_h['gs_FG'], color="teal",
         label="Annual,          p=0.48,   slope= 0.002",
         linestyle='--', marker='o', markersize=7)

# Growing
plt.scatter(gs_h_gs['water_year'], gs_h_gs['gs_FG'], color="aqua",
            label="Growing,        p=0.62,   slope= 0.002",
            marker='2', s=70)

# Non-growing
plt.scatter(gs_h_ngs['water_year'], gs_h_ngs['gs_FG'], color="deepskyblue",
            label="Non Growing, p=0.45,   slope= 0.001",
            marker='*', s=70)

ax.set_ylabel('$g_{c}$ ($mol$ $m^{-2}$ $s^{-1}$)')
ax.tick_params(axis='y', which='major', labelsize=10)

# Trend lines (skip NaNs)
def linefit(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 2:
        z = np.polyfit(x[m], y[m], 1)
        return np.poly1d(z)
    return None

p4 = linefit(gs_h['water_year'].values, gs_h['gs_FG'].values)
if p4 is not None:
    plt.plot(gs_h['water_year'], p4(gs_h['water_year']), color='teal')

p5 = linefit(gs_h_gs['water_year'].values, gs_h_gs['gs_FG'].values)
if p5 is not None:
    plt.plot(gs_h_gs['water_year'], p5(gs_h_gs['water_year']), color='aqua', linestyle='--')

p6 = linefit(gs_h_ngs['water_year'].values, gs_h_ngs['gs_FG'].values)
if p6 is not None:
    plt.plot(gs_h_ngs['water_year'], p6(gs_h_ngs['water_year']), color='deepskyblue', linestyle='--')

# Legend + title
ax.legend(loc='upper left', fontsize=8, labelspacing=0.0005)
ax.set_title("Harvard Forest, US-Ha2, Median Canopy Conductance ($g_{c}$)", fontsize=10)

# X-axis: ticks every 3 years, padded limits so first/last points aren’t on the edge
ax.set_xticks([2005, 2008, 2011, 2014, 2017, 2020, 2023])
ax.set_xlim(2004.5, 2023.5)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.tick_params(axis='x', which='major', labelsize=10)

plt.tight_layout()
plt.show()
fig.savefig('hemlock_gc_fixed.jpg', dpi=600, bbox_inches="tight")







#######################################################################################################
gs_h_gs=df     # GS  0.24-0.47
gs_h_ngs=df    # NGS 0.19-0.6

#df=gs_h
df=gs_h_gs
#df=gs_h_ngs

x1=df['water_year']
y1=df['gs_FG'].astype(float)
#pearsonr(x1, y1)
mk.original_test(y1)

df=gs_h_ngs
x1=df['water_year']
y1=df['gs_FG'].astype(float)
pearsonr(x1, y1)
mk.original_test(y1)





#######################################################################################################
plt.bar(gs_h['water_year'], gs_h['gs_FG'],color="teal", label="HEM: annual,          p=0.54,   slope= 0.002",linestyle='--')
plt.plot(gs_h['water_year'], gs_h['gs_FG'],color="teal",marker="d", markersize="7")
plt.bar(gs_h_gs['water_year'], gs_h_gs['gs_FG'],color="aqua", label="HEM: Growing,        p=0.65,   slope= 0.002",linestyle='--')
plt.bar(gs_h_ngs['water_year'],gs_h_ngs['gs_FG'],color="aquamarine", label="HEM: Non Growing, p=0.04,   slope= 0.003",linestyle='--')

ax.legend()
ax.legend(loc='upper right', fontsize = '8',labelcolor='linecolor',labelspacing=0.0005)
ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.set_title("Harvard Forest, Median Canopy Conductance ($G_{s}$)",fontsize=12)

z4 = np.polyfit(gs_h['water_year'], gs_h['gs_FG'], 1)
p4 = np.poly1d(z4)
plt.plot(gs_h['water_year'], p4(gs_h['water_year']),color='teal')

z5 = np.polyfit(gs_h_gs['water_year'], gs_h_gs['gs_FG'], 1)
p5 = np.poly1d(z5)
plt.plot(gs_h_gs['water_year'], p5(gs_h_gs['water_year']),color='aqua', linestyle='--')

z6 = np.polyfit(gs_h_ngs['water_year'], gs_h_ngs['gs_FG'], 1)
p6 = np.poly1d(z6)
plt.plot(gs_h_ngs['water_year'], p6(gs_h_ngs['water_year']),color='aquamarine', linestyle='--')




#######################################################################################################
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")

xls = pd.ExcelFile("variable_impor.xlsx")

ems_hourly_all = pd.read_excel(xls, 'ems_hourly_all')
ems_hourly_GS = pd.read_excel(xls, 'ems_hourly_GS').sort_values(by="importance",ascending=False)
ems_hourly_NGS = pd.read_excel(xls, 'ems_hourly_NGS').sort_values(by="importance",ascending=False)

hem_hourly_all= pd.read_excel(xls, 'hem_hourly_all').sort_values(by="importance",ascending=False)
hem_hourly_GS= pd.read_excel(xls, 'hem_hourly_GS').sort_values(by="importance",ascending=False)
hem_hourly_NGS= pd.read_excel(xls, 'hem_hourly_NGS').sort_values(by="importance",ascending=False)


ems_daily_all= pd.read_excel(xls, 'ems_daily_all').sort_values(by="importance",ascending=False)
ems_daily_GS= pd.read_excel(xls, 'ems_daily_GS').sort_values(by="importance",ascending=False)
ems_daily_NGS= pd.read_excel(xls, 'ems_daily_NGS').sort_values(by="importance",ascending=False)

HEM_daily_all= pd.read_excel(xls, 'HEM_daily_all').sort_values(by="importance",ascending=False)
HEM_daily_GS= pd.read_excel(xls, 'HEM_daily_GS').sort_values(by="importance",ascending=False)
HEM_daily_NGS= pd.read_excel(xls, 'HEM_daily_NGS').sort_values(by="importance",ascending=False)


#################################################################################################

## hourly ET model
################################################################################################3

df=ems_hourly_all
xe=list(ems_hourly_all["importance_e"])
xh=list(ems_hourly_all["importance_h"])
ye=list(ems_hourly_all["variable"])
y_pose = np.arange(len(ye))
labe= ['NETRAD', 'Rg', 'PAR', '$T_{air}$', '$T_{soil}$', 'NEE', 'GPP','H', 'VPD', 'PA', 'Year',
 'DoY', 'Month','Hour', 'sos', 'eos', 'WS', 'WD', 'u*']

width = 0.35
N=19 # number of bars
ind = np.arange(N)
#{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}


fig = plt.figure()
plt.rcParams['figure.figsize'] = (8,11)
fig, ax = plt.subplots()
# Example data
ax.barh(y_pose+0.2, xe, width,color='hotpink',label='US-Ha1: EMS', hatch='/')
ax.barh(y_pose+width+0.2, xh,width, align='center',color='mediumseagreen',label='US-Ha2: HEM', hatch='/')
ax.set_yticks(y_pose)
#ax.set_yticklabels(labe)

#ax.set_yticklabels(labe)
plt.yticks(ind + width / 1, (labe))
ax.set_xlim(0,0.6)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature Importance',size=18)
ax.set_title('Drivers Importance (All seasons: Hourly ET)',size=15)

ax.tick_params(axis = 'y', which = 'major', labelsize = 18,direction='out')
ax.tick_params(axis = 'x', which = 'major', labelsize = 18,direction='out')
#ax.set_xlim(0,0.4)
#ax.set_yticks(ind + width / 2)
ax.legend()
ax.legend(loc='lower right', fontsize = '15',labelcolor='linecolor')

plt.show()
fig.savefig('testfig.png',dpi=600,bbox_inches="tight")
plt.tight_layout()

##################################################################################################

df=ems_hourly_all
xe=list(ems_hourly_all["importance_e_gs"])
xh=list(ems_hourly_all["importance_h_gs"])
ye=list(ems_hourly_all["variable"])
y_pose = np.arange(len(ye))
labe= ['NETRAD', 'Rg', 'PAR', '$T_{air}$', '$T_{soil}$', 'NEE', 'GPP','H', 'VPD', 'PA', 'Year',
 'DoY', 'Month','Hour', 'sos', 'eos', 'WS', 'WD', 'u*']

width = 0.35
N=19 # number of bars
ind = np.arange(N)
#{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}


fig = plt.figure()
plt.rcParams['figure.figsize'] = (8,11)
fig, ax = plt.subplots()
# Example data
ax.barh(y_pose+0.2, xe, width,color='hotpink',label='US-Ha1: EMS', hatch='+')
ax.barh(y_pose+width+0.2, xh,width, align='center',color='mediumseagreen',label='US-Ha2: HEM', hatch='+')
ax.set_yticks(y_pose)
#ax.set_yticklabels(labe)
ax.set_xlim(0,0.5)
#ax.set_yticklabels(labe)
plt.yticks(ind + width / 1, (labe))

ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature Importance',size=18)
ax.set_title('Drivers Importance (Growing season: Hourly ET)',size=15)

ax.tick_params(axis = 'y', which = 'major', labelsize = 18,direction='out')
ax.tick_params(axis = 'x', which = 'major', labelsize = 18,direction='out')
#ax.set_xlim(0,0.4)
#ax.set_yticks(ind + width / 2)
ax.legend()
ax.legend(loc='lower right', fontsize = '15',labelcolor='linecolor')

plt.show()
fig.savefig('testfig.png',dpi=600,bbox_inches="tight")
plt.tight_layout()

##################################################################################################


df=ems_hourly_all
xe=list(ems_hourly_all["importance_e_ngs"])
xh=list(ems_hourly_all["importance_h_ngs"])
ye=list(ems_hourly_all["variable"])
y_pose = np.arange(len(ye))
labe= ['NETRAD', 'Rg', 'PAR', '$T_{air}$', '$T_{soil}$', 'NEE', 'GPP','H', 'VPD', 'PA', 'Year',
 'DoY', 'Month','Hour', 'sos', 'eos', 'WS', 'WD', 'u*']

width = 0.35
N=19 # number of bars
ind = np.arange(N)
#{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}


fig = plt.figure()
plt.rcParams['figure.figsize'] = (8,11)
fig, ax = plt.subplots()
# Example data
ax.barh(y_pose+0.2, xe, width,color='hotpink',label='US-Ha1: EMS', hatch='-')
ax.barh(y_pose+width+0.2, xh,width, align='center',color='mediumseagreen',label='US-Ha2: HEM', hatch='-')
ax.set_yticks(y_pose)
#ax.set_yticklabels(labe)
ax.set_xlim(0,0.5)
#ax.set_yticklabels(labe)
plt.yticks(ind + width / 1, (labe))

ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature Importance',size=18)
ax.set_title('Drivers Importance (Non Growing seaso: Hourly ET)',size=15)

ax.tick_params(axis = 'y', which = 'major', labelsize = 18,direction='out')
ax.tick_params(axis = 'x', which = 'major', labelsize = 18,direction='out')
#ax.set_xlim(0,0.4)
#ax.set_yticks(ind + width / 2)
ax.legend()
ax.legend(loc='lower right', fontsize = '15',labelcolor='linecolor')

plt.show()
fig.savefig('testfig.png',dpi=600,bbox_inches="tight")
plt.tight_layout()


############################################################################################################
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")

xls = pd.ExcelFile("variable_impor.xlsx")


ems_daily_all= pd.read_excel(xls, 'ems_daily_all')
ems_daily_GS= pd.read_excel(xls, 'ems_daily_GS').sort_values(by="importance",ascending=False)
ems_daily_NGS= pd.read_excel(xls, 'ems_daily_NGS').sort_values(by="importance",ascending=False)

HEM_daily_all= pd.read_excel(xls, 'HEM_daily_all').sort_values(by="importance",ascending=False)
HEM_daily_GS= pd.read_excel(xls, 'HEM_daily_GS').sort_values(by="importance",ascending=False)
HEM_daily_NGS= pd.read_excel(xls, 'HEM_daily_NGS').sort_values(by="importance",ascending=False)


#################################################################################################

## daily ET model
################################################################################################3

df=ems_daily_all
xe=list(ems_daily_all["d_importance_e"])
xh=list(ems_daily_all["d_importance_h"])
ye=list(ems_daily_all["variable"])
y_pose = np.arange(len(ye))
labe= ['NETRAD', 'Rg', 'PAR', '$T_{air}$', '$T_{soil}$', 'NEE', 'GPP','H', 'VPD', 'PA', 'Year',
 'DoY', 'Month', 'sos', 'eos', 'WS', 'WD', 'u*']

width = 0.35
N=18 # number of bars
ind = np.arange(N)
#{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}


fig = plt.figure()
plt.rcParams['figure.figsize'] = (8,11)
fig, ax = plt.subplots()
# Example data
ax.barh(y_pose+0.2, xe, width,color='hotpink',label='US-Ha1: EMS', hatch='*')
ax.barh(y_pose+width+0.2, xh,width, align='center',color='mediumseagreen',label='US-Ha2: HEM', hatch='*')
ax.set_yticks(y_pose)
#ax.set_yticklabels(labe)

#ax.set_yticklabels(labe)
plt.yticks(ind + width / 1, (labe))

ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature Importance',size=18)
ax.set_title('Drivers Importance (All seasons: Daily ET)',size=15)
ax.set_xlim(0,0.6)
ax.tick_params(axis = 'y', which = 'major', labelsize = 18,direction='out')
ax.tick_params(axis = 'x', which = 'major', labelsize = 18,direction='out')
#ax.set_xlim(0,0.4)
#ax.set_yticks(ind + width / 2)
ax.legend()
ax.legend(loc='lower right', fontsize = '15',labelcolor='linecolor')

plt.show()
fig.savefig('testfig.png',dpi=600,bbox_inches="tight")
plt.tight_layout()

##################################################################################################

df=ems_daily_all
xe=list(ems_daily_all["d_importance_e_gs"])
xh=list(ems_daily_all["d_importance_h_gs"])
ye=list(ems_daily_all["variable"])
y_pose = np.arange(len(ye))
labe= ['NETRAD', 'Rg', 'PAR', '$T_{air}$', '$T_{soil}$', 'NEE', 'GPP','H', 'VPD', 'PA', 'Year',
 'DoY', 'Month', 'sos', 'eos', 'WS', 'WD', 'u*']

width = 0.35
N=18 # number of bars
ind = np.arange(N)
#{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}


fig = plt.figure()
plt.rcParams['figure.figsize'] = (8,11)
fig, ax = plt.subplots()
# Example data
ax.barh(y_pose+0.2, xe, width,color='hotpink',label='US-Ha1: EMS', hatch='.')
ax.barh(y_pose+width+0.2, xh,width, align='center',color='mediumseagreen',label='US-Ha2: HEM', hatch='.')
ax.set_yticks(y_pose)
#ax.set_yticklabels(labe)

#ax.set_yticklabels(labe)
plt.yticks(ind + width / 1, (labe))

ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature Importance',size=18)
ax.set_title('Drivers Importance (Growing season: Daily ET)',size=15)
ax.set_xlim(0,0.8)
ax.tick_params(axis = 'y', which = 'major', labelsize = 18,direction='out')
ax.tick_params(axis = 'x', which = 'major', labelsize = 18,direction='out')
#ax.set_xlim(0,0.4)
#ax.set_yticks(ind + width / 2)
ax.legend()
ax.legend(loc='lower right', fontsize = '15',labelcolor='linecolor')

plt.show()
fig.savefig('testfig.png',dpi=600,bbox_inches="tight")
plt.tight_layout()

##################################################################################################


df=ems_daily_all
xe=list(ems_daily_all["d_importance_e_ngs"])
xh=list(ems_daily_all["d_importance_h_ngs"])
ye=list(ems_daily_all["variable"])
y_pose = np.arange(len(ye))
labe= ['NETRAD', 'Rg', 'PAR', '$T_{air}$', '$T_{soil}$', 'NEE', 'GPP','H', 'VPD', 'PA', 'Year',
 'DoY', 'Month', 'sos', 'eos', 'WS', 'WD', 'u*']

width = 0.35
N=18 # number of bars
ind = np.arange(N)
#{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}


fig = plt.figure()
plt.rcParams['figure.figsize'] = (8,11)
fig, ax = plt.subplots()
# Example data
ax.barh(y_pose+0.2, xe, width,color='hotpink',label='US-Ha1: EMS', hatch='o')
ax.barh(y_pose+width+0.2, xh,width, align='center',color='mediumseagreen',label='US-Ha2: HEM', hatch='o')
ax.set_yticks(y_pose)
#ax.set_yticklabels(labe)

#ax.set_yticklabels(labe)
plt.yticks(ind + width / 1, (labe))

ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature Importance',size=18)
ax.set_title('Drivers Importance (Non Growing season: Daily ET)',size=15)
ax.set_xlim(0,0.5)
ax.tick_params(axis = 'y', which = 'major', labelsize = 18,direction='out')
ax.tick_params(axis = 'x', which = 'major', labelsize = 18,direction='out')
#ax.set_xlim(0,0.4)
#ax.set_yticks(ind + width / 2)
ax.legend()
ax.legend(loc='lower right', fontsize = '15',labelcolor='linecolor')

plt.show()
fig.savefig('testfig.png',dpi=600,bbox_inches="tight")
plt.tight_layout()
###########################################################################################################
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")
xls = pd.ExcelFile("variable_impor.xlsx")
ems_daily_all= pd.read_excel(xls, 'pre_EMS_daily_all')
df=ems_daily_all



xe=list(ems_daily_all["importance_e"])
xh=list(ems_daily_all["importance_h"])
ye=list(ems_daily_all["variable"])
y_pose = np.arange(len(ye))
labe= ['NETRAD', 'Rg', 'PAR', '$T_{air}$', '$T_{soil}$', 'NEE', 'GPP','H', 'VPD', 'PA', 'Year',
 'DoY', 'Month', 'sos', 'eos', 'WS', 'WD', 'u*', '$Precip_{7}$', '$Precip_{14}$']

width = 0.35
N=20 # number of bars
ind = np.arange(N)
#{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}


fig = plt.figure()
plt.rcParams['figure.figsize'] = (8,11)
fig, ax = plt.subplots()
# Example data
ax.barh(y_pose+0.2, xe, width,color='hotpink',label='US-Ha1: EMS', hatch='*')
ax.barh(y_pose+width+0.2, xh,width, align='center',color='mediumseagreen',label='US-Ha2: HEM', hatch='*')
ax.set_yticks(y_pose)
#ax.set_yticklabels(labe)

#ax.set_yticklabels(labe)
plt.yticks(ind + width / 1, (labe))

ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature Importance',size=18)
ax.set_title('Predictors Importance (All seasons: Daily ET)',size=15)
ax.set_xlim(0,0.7)
ax.tick_params(axis = 'y', which = 'major', labelsize = 18,direction='out')
ax.tick_params(axis = 'x', which = 'major', labelsize = 18,direction='out')
#ax.set_xlim(0,0.4)
#ax.set_yticks(ind + width / 2)
ax.legend()
ax.legend(loc='lower right', fontsize = '15',labelcolor='linecolor')

plt.show()
fig.savefig('testfig.png',dpi=600,bbox_inches="tight")
plt.tight_layout()

########################################################################################################

xe=list(ems_daily_all["importance_e_gs"])
xh=list(ems_daily_all["importance_h_gs"])
ye=list(ems_daily_all["variable"])
y_pose = np.arange(len(ye))
labe= ['NETRAD', 'Rg', 'PAR', '$T_{air}$', '$T_{soil}$', 'NEE', 'GPP','H', 'VPD', 'PA', 'Year',
 'DoY', 'Month', 'sos', 'eos', 'WS', 'WD', 'u*', '$Precip_{7}$', '$Precip_{14}$']

width = 0.35
N=20 # number of bars
ind = np.arange(N)
#{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}


fig = plt.figure()
plt.rcParams['figure.figsize'] = (8,11)
fig, ax = plt.subplots()
# Example data
ax.barh(y_pose+0.2, xe, width,color='hotpink',label='US-Ha1: EMS', hatch='.')
ax.barh(y_pose+width+0.2, xh,width, align='center',color='mediumseagreen',label='US-Ha2: HEM', hatch='.')
ax.set_yticks(y_pose)
#ax.set_yticklabels(labe)

#ax.set_yticklabels(labe)
plt.yticks(ind + width / 1, (labe))

ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature Importance',size=18)
ax.set_title('Predictors Importance (Growing season: Daily ET)',size=15)
ax.set_xlim(0,0.8)
ax.tick_params(axis = 'y', which = 'major', labelsize = 18,direction='out')
ax.tick_params(axis = 'x', which = 'major', labelsize = 18,direction='out')
#ax.set_xlim(0,0.4)
#ax.set_yticks(ind + width / 2)
ax.legend()
ax.legend(loc='lower right', fontsize = '15',labelcolor='linecolor')

plt.show()
fig.savefig('testfig.png',dpi=600,bbox_inches="tight")
plt.tight_layout()

###################################################################################################

xe=list(ems_daily_all["importance_e_ngs"])
xh=list(ems_daily_all["importance_h_ngs"])
ye=list(ems_daily_all["variable"])
y_pose = np.arange(len(ye))
labe= ['NETRAD', 'Rg', 'PAR', '$T_{air}$', '$T_{soil}$', 'NEE', 'GPP','H', 'VPD', 'PA', 'Year',
 'DoY', 'Month', 'sos', 'eos', 'WS', 'WD', 'u*', '$Precip_{7}$', '$Precip_{14}$']

width = 0.35
N=20 # number of bars
ind = np.arange(N)
#{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}


fig = plt.figure()
plt.rcParams['figure.figsize'] = (8,11)
fig, ax = plt.subplots()
# Example data
ax.barh(y_pose+0.2, xe, width,color='hotpink',label='US-Ha1: EMS', hatch='o')
ax.barh(y_pose+width+0.2, xh,width, align='center',color='mediumseagreen',label='US-Ha2: HEM', hatch='o')
ax.set_yticks(y_pose)
#ax.set_yticklabels(labe)

#ax.set_yticklabels(labe)
plt.yticks(ind + width / 1, (labe))

ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature Importance',size=18)
ax.set_title('Predictors Importance (Non Growing season: Daily ET)',size=15)
ax.set_xlim(0,0.5)
ax.tick_params(axis = 'y', which = 'major', labelsize = 18,direction='out')
ax.tick_params(axis = 'x', which = 'major', labelsize = 18,direction='out')
#ax.set_xlim(0,0.4)
#ax.set_yticks(ind + width / 2)
ax.legend()
ax.legend(loc='lower right', fontsize = '15',labelcolor='linecolor')

plt.show()
fig.savefig('testfig.png',dpi=600,bbox_inches="tight")
plt.tight_layout()




########################################################################################################
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")

ems_da_all_test=pd.read_csv('ems_da_all_test.csv')
ems_da_all_test['lambda']=(3149000-2370*(ems_da_all_test['Tair_fill']+273.16))*1e-6
ems_da_all_test['ET_xg']=(ems_da_all_test['xg_model']/ems_da_all_test['lambda'])*(1/1e6)*3600
ems_da_all_test['ET_obs']=(ems_da_all_test['obs']/ems_da_all_test['lambda'])*(1/1e6)*3600



ems_all_da_GS_test=pd.read_csv('ems_all_da_GS_test.csv')
ems_all_da_GS_test['lambda']=(3149000-2370*(ems_all_da_GS_test['Tair_fill']+273.16))*1e-6
ems_all_da_GS_test['ET_xg']=(ems_all_da_GS_test['xg_model']/ems_all_da_GS_test['lambda'])*(1/1e6)*3600
ems_all_da_GS_test['ET_obs']=(ems_all_da_GS_test['obs']/ems_all_da_GS_test['lambda'])*(1/1e6)*3600



ems_da_NGS_test=pd.read_csv('ems_da_NGS_test.csv')
ems_da_NGS_test['lambda']=(3149000-2370*(ems_da_NGS_test['Tair_fill']+273.16))*1e-6
ems_da_NGS_test['ET_xg']=(ems_da_NGS_test['xg_model']/ems_da_NGS_test['lambda'])*(1/1e6)*3600
ems_da_NGS_test['ET_obs']=(ems_da_NGS_test['obs']/ems_da_NGS_test['lambda'])*(1/1e6)*3600



hem_da_all_test=pd.read_csv('hem_da_all_test.csv')
hem_da_all_test['lambda']=(3149000-2370*(hem_da_all_test['Tair']+273.16))*1e-6
hem_da_all_test['ET_xg']=(hem_da_all_test['xg_model']/hem_da_all_test['lambda'])*(1/1e6)*1800
hem_da_all_test['ET_obs']=(hem_da_all_test['obs']/hem_da_all_test['lambda'])*(1/1e6)*1800



hem_da_GS_test=pd.read_csv('hem_da_GS_test.csv')
hem_da_GS_test['lambda']=(3149000-2370*(hem_da_GS_test['Tair']+273.16))*1e-6
hem_da_GS_test['ET_xg']=(hem_da_GS_test['xg_model']/hem_da_GS_test['lambda'])*(1/1e6)*1800
hem_da_GS_test['ET_obs']=(hem_da_GS_test['obs']/hem_da_GS_test['lambda'])*(1/1e6)*1800



hem_da_NGS_test=pd.read_csv('hem_da_NGS_test.csv')
hem_da_NGS_test['lambda']=(3149000-2370*(hem_da_NGS_test['Tair']+273.16))*1e-6
hem_da_NGS_test['ET_xg']=(hem_da_NGS_test['xg_model']/hem_da_NGS_test['lambda'])*(1/1e6)*1800
hem_da_NGS_test['ET_obs']=(hem_da_NGS_test['obs']/hem_da_NGS_test['lambda'])*(1/1e6)*1800


########################################################################################################

def stat (obs,pre):
    nse=NSE(pre,obs)
   # pear=pearsonr(obs,pre)
    wil=will(pre,obs)   
    MAE=mean_absolute_error(obs,pre)
    RMSE=sqrt(mean_squared_error(obs,pre))
   # reg_=reg.score(x1, y1)
    pbias=(np.sum(pre-obs)/np.sum(obs))*100
    return nse, MAE
#    return nse, RMSE



df=hem_da_NGS_test
obs=df["ET_obs"] # already in meters. no need to convert
pre=df["ET_xg"]

x1=obs
y1=pre
#reg = LinearRegression(fit_intercept=True).fit(x1, y1)
stat(obs,pre) #nse, will, pears, MAE, RMSE, pbias

df=hem_da_NGS_test
obs=df["ET_obs"] # already in meters. no need to convert
pre=df["ET_xg"]
x1=obs
y1=pre
#reg = LinearRegression(fit_intercept=True).fit(x1, y1)
stat(obs,pre) #nse, will, pears, MAE, RMSE, pbias

# Recorded NSE and RMSE pairs: (0.69, 0.39) and (0.67, 0.52).

#plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['figure.figsize'] = (5,5)
fig, ax = plt.subplots()
ax.plot(ems_da_all_test["ET_obs"],ems_da_all_test["ET_xg"],'x', color="orange",markersize=7,marker='^',label="All seasons (n= 3534)",mfc='none')
ax.plot(x1,0.8066*x1+0.2425,'-', linewidth=2, color='darkorange', label='Fitted line (All seasons)')#overall trend line 

ax.plot(ems_all_da_GS_test["ET_obs"],ems_all_da_GS_test["ET_xg"],'x', color="violet",markersize=7,marker='o',label="Growing season (n= 1403)",mfc='none')
ax.plot(x1,0.7781*x1+0.5824,'--', linewidth=2, color='darkviolet', label='Fitted line (Growing season)')#overall trend line 


ax.plot(ems_da_NGS_test["ET_obs"],ems_da_NGS_test["ET_xg"],'x', color="salmon",markersize=10,marker='*',label="Non Growing season (n= 2131)",mfc='none')
ax.plot(x1,0.6445*x1+0.3158,'--', linewidth=2, color='tomato', label='Fitted line (Non Growing season)')#overall trend line 

#ax.plot(CS4["lw_net"],CS4["lwnet_idso"],'x', color="cyan",markersize=6,marker='1',label="US-CS4 (2020)",mfc='none')
#ax.grid(which='major', axis='both', linestyle='--', linewidth=1)
line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
#ax.plot(x1,0.8449*x1+0.0048,'-', linewidth=2)#overall trend line 

#y= 0.8078x + 0.0031 shallow
# y = 0.9022x + 0.0046   deep
ax.set_title("US-Ha1: EMS Tower, Daily ET Model, Testing",fontsize=13)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend(loc='upper left',prop={'size': '9'})

ax.legend()
ax.legend(loc='upper right', fontsize = '8',labelcolor='linecolor',labelspacing=0.0005)
ax.legend(bbox_to_anchor=(1.1, 1.05))

ax.set_xlabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_ylabel('Model ET (mm)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(13)
ax.xaxis.label.set_size(13) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 13,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 13,direction='in')
ax.set_aspect('auto')
ax.set_ylim([-1,9])
ax.set_xlim([-1,9])

#R2	   NSE	   will	   pear  	MAE  	RMSE	pbias
#0.91	0.90	0.97	0.95	0.113	0.160	-1.30


ax.text(0.45, 0.95, 'n = 3534',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='orange', fontsize=12)

ax.text(0.45, 0.90, 'NSE = 0.76',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='orange', fontsize=12)

ax.text(0.45, 0.85, '$\mathregular{R^{2}}$ = 0.87',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='orange', fontsize=12)

ax.text(0.45, 0.80, 'Willmotts = 0.93',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='orange', fontsize=12)

ax.text(0.45, 0.75, 'Pbias = -3.5 (%)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='orange', fontsize=12)

ax.text(0.45, 0.70, 'MAE = 0.46 (mm)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='orange', fontsize=12)
ax.text(0.45, 0.65, 'RMSE = 0.62 (mm)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='orange', fontsize=12)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


#######################################################################################################

## timeseries


df=ems_da_all_test
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
df.set_index(('TIMESTAMP'), inplace=True)


plt.rcParams['figure.figsize'] = (7.5, 3.3)
fig, ax = plt.subplots()
#plt.plot(df.index,df.obs,'x', color="purple", markersize=8,marker='*',label="Observed Anomaly",mfc='none')
plt.plot(df.index,df["ET_obs"], color="grey", linestyle='--',label="Observed ET",mfc='none')

plt.plot(df.index,df.ET_xg,'x', color="orange",markersize=3,marker='^',label="Model ET",mfc='none')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '12')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.locator_params(axis='y', nbins=6)
#ax.set_ylim([-.6,.6])
ax.set_ylabel('ET (mm) ')  # we already handled the x-label with ax1
ax.legend(loc='upper left', fontsize = '8',labelcolor='linecolor')
ax.set_title("US-Ha1: EMS Tower, Daily ET Model, Testing",fontsize=13, color='black')
#ax.set_xlabel('Year-Month')  # we already handled the x-label with ax1

ax.tick_params(axis = 'x', which = 'major', labelsize = 10)
ax.tick_params(axis = 'y', which = 'major', labelsize = 10)


ax.xaxis.set_major_locator(mdates.DayLocator(interval=300))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


########################################################################################################
## HEM
#######################################################################################################

hem_da_all_test=pd.read_csv('hem_da_all_test.csv')
hem_da_all_test['lambda']=(3149000-2370*(hem_da_all_test['Tair']+273.16))*1e-6
hem_da_all_test['ET_xg']=(hem_da_all_test['xg_model']/hem_da_all_test['lambda'])*(1/1e6)*1800
hem_da_all_test['ET_obs']=(hem_da_all_test['obs']/hem_da_all_test['lambda'])*(1/1e6)*1800


hem_da_GS_test=pd.read_csv('hem_da_GS_test.csv')
hem_da_GS_test['lambda']=(3149000-2370*(hem_da_GS_test['Tair']+273.16))*1e-6
hem_da_GS_test['ET_xg']=(hem_da_GS_test['xg_model']/hem_da_GS_test['lambda'])*(1/1e6)*1800
hem_da_GS_test['ET_obs']=(hem_da_GS_test['obs']/hem_da_GS_test['lambda'])*(1/1e6)*1800


hem_da_NGS_test=pd.read_csv('hem_da_NGS_test.csv')
hem_da_NGS_test['lambda']=(3149000-2370*(hem_da_NGS_test['Tair']+273.16))*1e-6
hem_da_NGS_test['ET_xg']=(hem_da_NGS_test['xg_model']/hem_da_NGS_test['lambda'])*(1/1e6)*1800
hem_da_NGS_test['ET_obs']=(hem_da_NGS_test['obs']/hem_da_NGS_test['lambda'])*(1/1e6)*1800



df=hem_da_NGS_test
obs=df["ET_obs"] # already in meters. no need to convert
pre=df["ET_xg"]

x1=obs
y1=pre
stat(obs,pre) #nse, will, pears, MAE, RMSE, pbias




#plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['figure.figsize'] = (5,5)
fig, ax = plt.subplots()
ax.plot(hem_da_all_test["ET_obs"],hem_da_all_test["ET_xg"],'x', color="crimson",markersize=10,marker='1',label="All seasons (n= 2192)",mfc='none')
ax.plot(x1,0.7329*x1+0.0819,'-', linewidth=2, color='orangered', label='Fitted line (All seasons)')#overall trend line 


ax.plot(hem_da_GS_test["ET_obs"],hem_da_GS_test["ET_xg"],'x', color="aqua",markersize=10,marker='+',label="Growing season (n= 1081)",mfc='none')
ax.plot(x1,0.5947*x1+0.5359,'--', linewidth=2, color='darkturquoise', label='Fitted line (Growing season)')#overall trend line 


ax.plot(ems_da_NGS_test["ET_obs"],ems_da_NGS_test["ET_xg"],'x', color="cornflowerblue",markersize=7,marker='d',label="Non Growing season (n= 1110)",mfc='none')
ax.plot(x1,0.7133*x1+0.0442,'--', linewidth=2, color='royalblue', label='Fitted line (Non Growing season)')#overall trend line 

#ax.plot(CS4["lw_net"],CS4["lwnet_idso"],'x', color="cyan",markersize=6,marker='1',label="US-CS4 (2020)",mfc='none')
#ax.grid(which='major', axis='both', linestyle='--', linewidth=1)
line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
#ax.plot(x1,0.8449*x1+0.0048,'-', linewidth=2)#overall trend line 

#y= 0.8078x + 0.0031 shallow
# y = 0.9022x + 0.0046   deep
ax.set_title("US-Ha2: HEM Tower, Daily ET Model, Testing",fontsize=13)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend(loc='upper left',prop={'size': '9'})

ax.legend()
ax.legend(loc='upper right', fontsize = '8',labelcolor='linecolor',labelspacing=0.0005)
ax.legend(bbox_to_anchor=(1.1, 1.05))

ax.set_xlabel('Observed ET (mm)')  # we already handled the x-label with ax1
ax.set_ylabel('Model ET (mm)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(13)
ax.xaxis.label.set_size(13) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 13,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 13,direction='in')
ax.set_aspect('auto')
ax.set_ylim([-1,7])
ax.set_xlim([-1,7])

#R2	   NSE	   will	   pear  	MAE  	RMSE	pbias
#0.91	0.90	0.97	0.95	0.113	0.160	-1.30


ax.text(0.45, 0.95, 'n = 2192',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='crimson', fontsize=12)

ax.text(0.45, 0.90, 'NSE = 0.74',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='crimson', fontsize=12)

ax.text(0.45, 0.85, '$\mathregular{R^{2}}$ = 0.89',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='crimson', fontsize=12)

ax.text(0.45, 0.80, 'Willmotts = 0.92',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='crimson', fontsize=12)

ax.text(0.45, 0.75, 'Pbias = -19.9 (%)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='crimson', fontsize=12)

ax.text(0.45, 0.70, 'MAE = 0.42 (mm)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='crimson', fontsize=12)

ax.text(0.45, 0.65, 'RMSE = 0.60 (mm)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='crimson', fontsize=12)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


#######################################################################################################

## timeseries


df=hem_da_all_test
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
df.set_index(('TIMESTAMP'), inplace=True)







plt.rcParams['figure.figsize'] = (7.5, 3.3)
fig, ax = plt.subplots()
#plt.plot(df.index,df.obs,'x', color="purple", markersize=8,marker='*',label="Observed Anomaly",mfc='none')
plt.plot(df.index,df["ET_obs"], color="grey", linestyle='--',label="Observed ET",mfc='none')

plt.plot(df.index,df.ET_xg,'x', color="crimson",markersize=4,marker='1',label="Model ET",mfc='none')
ax.tick_params(axis='y', labelcolor='black')

 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.locator_params(axis='y', nbins=5)
ax.set_ylim([-1,6])
ax.set_ylabel('ET (mm) ')  # we already handled the x-label with ax1
ax.legend(loc='upper left', fontsize = '8',labelcolor='linecolor')
ax.set_title("US-Ha2: HEM Tower, Daily ET Model, Testing",fontsize=13, color='black')
#ax.set_xlabel('Year-Month')  # we already handled the x-label with ax1

ax.tick_params(axis = 'x', which = 'major', labelsize = 10)
ax.tick_params(axis = 'y', which = 'major', labelsize = 10)


ax.xaxis.set_major_locator(mdates.DayLocator(interval=300))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


####################################################################################################

## water use efficiency
###############################################################################################

df=EMS_daily
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
df= df.groupby(['water_year']).sum()
df=df.reset_index()

mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))

#plt.plot(df["GPP_g"])

df["WUE"]=(df["GPP_g"]/df["ET"])

x1=df['water_year']
y1=df['WUE'].astype(float)
#pearsonr(x1, y1)
mk.original_test(y1)
hg.pettitt_test(y1)

#std=df['WUE'].std()

#Sen's slope * length of period)/mean *100
# peaked at 2008

plt.rcParams['figure.figsize'] = (4, 4)
fig, ax = plt.subplots()
plt.plot(x1, y1, color='crimson', ls='-',marker='o',markersize=5 )
#plt.errorbar(x1, y1, color='crimson', ls='-', capsize=2, capthick=0.5,marker='o',markersize=5 )

#plt.fill_between(x1, y1-76.87,y1+76.87, color='lightgrey')
ax.tick_params(axis='y', labelcolor='black')
#ax.legend()
#ax.legend(loc='upper left', fontsize = '9')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),labelspacing=0.0005,labelcolor='linecolor')

ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('Total Annual WUE (g C per kg $H_{2}$O)')  # we already handled the x-label with ax1
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
#z1 = np.polyfit(x1, y1, 1)
#p1 = np.poly1d(z1)
#plt.plot(x1, p1(x1),color='black',ls='--')
#ax.set_ylim(1,6)
#ax.set_title("US-Ha1: Annual water use efficiency (WUE)",fontsize=8)
#ax.set_title("US-Ha1: Growing season water use efficiency (WUE)",fontsize=8)
ax.set_title("US-Ha1: Nongrowing season water use efficiency (WUE)",fontsize=8)

#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 10)
ax.tick_params(axis = 'y', which = 'major', labelsize =10)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#####################################################################################################

df=EMS_daily
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
#mask = (df['water_year'] >= 2008) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
daily_ha1=df
df= df.groupby(['water_year']).sum()
df=df.reset_index()

df["WUE"]=(df["GPP_g"]/df["ET"])
x1=df['water_year']
y1=df['WUE'].astype(float)

WUE_e=df
df_e=WUE_e

mk.original_test(y1)
#std=df['WUE'].std()
# peaked at 2008
plt.plot(df["WUE"])
df_e=df

df=df_e
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2008)
#mask = (df['water_year'] >= 2008) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
x1=df['water_year']
y1=df['WUE'].astype(float)
mk.original_test(y1)
df_e1=df
x1=df['water_year']
y1=df['WUE'].astype(float)
mk.original_test(y1)


#Sen's slope * length of period)/mean *100
((0.1229*16)/3.186221) *100

df=df_e
mask = (df['water_year'] >= 2008) & (df['water_year'] <= 2023)
#mask = (df['water_year'] >= 2008) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
x1=df['water_year']
y1=df['WUE'].astype(float)
mk.original_test(y1)
#df_e2=df

((-0.08878*16)/2.9759) *100
###################################################################################################


df=HEM_daily
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
#mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2022)
#mask = (df['water_year'] >= 2008) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))

df= df.groupby(['water_year']).sum()
df=df.reset_index()

df["WUE"]=(df["GPP_g"]/df["ET"])
x1=df['water_year']
y1=df['WUE'].astype(float)

mk.original_test(y1)
#std=df['WUE'].std()
# peaked at 2008
plt.plot(df["WUE"])
df = df.iloc[:-2]
df_h=df

df=df_h
WUE_h=df_h

df_h=WUE_h

mask = (df['water_year'] >= 2005) & (df['water_year'] <= 2012)
#mask = (df['water_year'] >= 2008) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
x1=df['water_year']
y1=df['WUE'].astype(float)
mk.original_test(y1)
df_h1=df

((0.3556*8)/3.53676) *100

df=df_h
mask = (df['water_year'] >= 2012) & (df['water_year'] <= 2022)
#mask = (df['water_year'] >= 2008) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
x1=df['water_year']
y1=df['WUE'].astype(float)
mk.original_test(y1)
df_h2=df


###################################################################################################

#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(df_e['water_year'].head(17), df_e['WUE'].head(17),color="deeppink",label="EMS (1993-2008), p=1.3*10$^{-4}$, slope=0.12",linestyle='-', marker='.',markersize=8)
plt.plot(df_e['water_year'].tail(15), df_e['WUE'].tail(15),color="pink",label="EMS (2008-2023), p=1.0*10$^{-2}$, slope=-0.09",linestyle='--', marker='.',markersize=8)

plt.plot(df_h['water_year'].head(8), df_h['WUE'].head(8),color="teal",label="HEM (2005-2012), p=4.4*10$^{-3}$, slope=0.36",linestyle='-', marker='+',markersize=8)
plt.plot(df_h['water_year'].tail(11), df_h['WUE'].tail(11),color="mediumturquoise",label="HEM (2012-2023), p=0.44,        slope=-0.03",linestyle='--', marker='+',markersize=8)

ax.tick_params(axis='y', labelcolor='black')

ax.legend()
#ax.legend(loc='upper left', fontsize = '6',labelspacing=0.0005)
#ax.legend(loc='lower right', fontsize = '5',labelcolor='linecolor',labelspacing=0.0005)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),labelspacing=0.0005,labelcolor='linecolor')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Total Annual WUE (g C per kg $H_{2}$O)')  # we already handled the x-label with ax1

#ax.set_ylim(250,800)

z1 = np.polyfit(df_e['water_year'].head(16), df_e['WUE'].head(16), 1)
p1 = np.poly1d(z1)
plt.plot(df_e['water_year'].head(16), p1(df_e['water_year'].head(16)),color='deeppink')


z2 = np.polyfit(df_e['water_year'].tail(16), df_e['WUE'].tail(16), 1)
p2 = np.poly1d(z2)
plt.plot(df_e['water_year'].tail(16), p2(df_e['water_year'].tail(16)),color='pink',linestyle='--')


z3 = np.polyfit(df_h['water_year'].head(9), df_h['WUE'].head(9), 1)
p3 = np.poly1d(z3)
plt.plot(df_h['water_year'].head(9), p3(df_h['water_year'].head(9)),color='teal')


z4 = np.polyfit(df_h['water_year'].tail(11), df_h['WUE'].tail(11), 1)
p4 = np.poly1d(z4)
plt.plot(df_h['water_year'].tail(11), p4(df_h['water_year'].tail(11)),color='mediumturquoise',linestyle='--')

ax.set_title("Harvard Forest, EMS Tower (US-Ha1), HEM Tower (US-Ha2)",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
#fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
#plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


#####################################################################################################
# revised version  Februrary 

###GPP and ET changes slope by year
from scipy.stats import linregress, pearsonr

def yearly_stats(group):
    if len(group) < 2:  # Ensure enough data points
        return pd.Series({'slope': None, 'pearson_r': None})

    slope, _, _, _, _ = linregress(group['GPP_g'], group['ET'])
    pearson_r = group['GPP_g'].corr(group['ET'])

    return pd.Series({'slope': slope, 'pearson_r': pearson_r})

# Group by year and calculate stats
#yearly_metrics_ems  = df.groupby('water_year').apply(yearly_stats)
# Group by year and apply function





os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")

df=pd.read_csv("EMS_daily.csv")
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])

df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
dfe=df
df["month"]=df['TIMESTAMP'].dt.month
dfe=df


df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
mask = (df['water_year'] >=1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))

df["day_of_year"] = df["DoY"]  

# Function to compute slope and correlation

# Group by year and apply function
yearly_metrics_ems = df.groupby("water_year").apply(yearly_stats).reset_index()

#########################################################################################
##hem 
################################################################################


df=pd.read_csv("HEM_daily.csv")

df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])

df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
dfh=df
df["month"]=df['TIMESTAMP'].dt.month
dfh=df


df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
mask = (df['water_year'] >=2005) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))

# Function to compute slope and correlation

# Group by year and apply function
yearly_metrics_hem = df.groupby("water_year").apply(yearly_stats).reset_index()

yearly_metrics_hem = yearly_metrics_hem.iloc[1:-1].reset_index(drop=True)



#plt.plot(yearly_metrics_ems["water_year"],yearly_metrics_ems["slope"] )
#plt.plot(yearly_metrics_hem["water_year"],yearly_metrics_hem["slope"] )



#plt.plot(yearly_metrics_ems["water_year"],yearly_metrics_ems["pearson_r"] )
#plt.plot(yearly_metrics_hem["water_year"],yearly_metrics_hem["pearson_r"] )

#############################################################################################
df_e=yearly_metrics_ems
df_h=yearly_metrics_hem


### ems

x1=df_e['water_year'].head(16)
y1=df_e['slope'].head(16)
mk.original_test(y1)


x1=df_e['water_year'].tail(16)
y1=df_e['slope'].tail(16)
mk.original_test(y1)
#########################################

### hem

x1=df_h['water_year'].head(8)
y1=df_h['slope'].head(8)
mk.original_test(y1)



x1=df_h['water_year'].tail(10)
y1=df_h['slope'].tail(10)
mk.original_test(y1)


plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(df_e['water_year'].head(17), df_e['slope'].head(17),color="deeppink",label="EMS (1993-2008), p=6.0*10$^{-3}$, slope=-0.01",linestyle='-', marker='*',markersize=7)
plt.plot(df_e['water_year'].tail(15), df_e['slope'].tail(15),color="pink",label="EMS (2008-2023), p=1.0*10$^{-2}$, slope= 0.01",linestyle='--', marker='*',markersize=7)

plt.plot(df_h['water_year'].head(8), df_h['slope'].head(8),color="teal",label="HEM (2006-2013), p=9.3*10$^{-3}$, slope=-0.03",linestyle='-', marker='.',markersize=8)
plt.plot(df_h['water_year'].tail(10), df_h['slope'].tail(10),color="mediumturquoise",label="HEM (2013-2023), p=7.6*10$^{-1}$, slope= 3.3",linestyle='--', marker='.',markersize=8)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
#ax.legend(loc='upper left', fontsize = '6',labelspacing=0.0005)
#ax.legend(loc='lower right', fontsize = '5',labelcolor='linecolor',labelspacing=0.0005)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),labelspacing=0.0005,labelcolor='linecolor')

ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('ET/GPP')  # we already handled the x-label with ax1

#ax.set_ylim(250,800)

z1 = np.polyfit(df_e['water_year'].head(16), df_e['slope'].head(16), 1)
p1 = np.poly1d(z1)
plt.plot(df_e['water_year'].head(16), p1(df_e['water_year'].head(16)),color='deeppink')


z2 = np.polyfit(df_e['water_year'].tail(16), df_e['slope'].tail(16), 1)
p2 = np.poly1d(z2)
plt.plot(df_e['water_year'].tail(16), p2(df_e['water_year'].tail(16)),color='pink',linestyle='--')


z3 = np.polyfit(df_h['water_year'].head(8), df_h['slope'].head(8), 1)
p3 = np.poly1d(z3)
plt.plot(df_h['water_year'].head(8), p3(df_h['water_year'].head(8)),color='teal')


z4 = np.polyfit(df_h['water_year'].tail(10), df_h['slope'].tail(10), 1)
p4 = np.poly1d(z4)
plt.plot(df_h['water_year'].tail(10), p4(df_h['water_year'].tail(10)),color='mediumturquoise',linestyle='--')


#ax.set_title("Harvard Forest, EMS Tower (US-Ha1), HEM Tower (US-Ha2)",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
#ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
#fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
#plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()
#######################################################################################################


plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(df_e['water_year'].head(17), df_e['pearson_r'].head(17),color="deeppink",label="EMS (1993-2008)",linestyle='-', marker='*',markersize=7)
plt.plot(df_e['water_year'].tail(15), df_e['pearson_r'].tail(15),color="pink",label="EMS (2008-2023)",linestyle='--', marker='*',markersize=7)

plt.plot(df_h['water_year'].head(8), df_h['pearson_r'].head(8),color="teal",label="HEM (2006-2013)",linestyle='-', marker='.',markersize=8)
plt.plot(df_h['water_year'].tail(10), df_h['pearson_r'].tail(10),color="mediumturquoise",label="HEM (2013-2023)",linestyle='--', marker='.',markersize=8)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='lower left', fontsize = '10',labelspacing=0.005)
#ax.legend(loc='lower right', fontsize = '5',labelcolor='linecolor',labelspacing=0.0005)
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),labelspacing=0.0005,labelcolor='linecolor')

ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('pearson correlation (r)')  # we already handled the x-label with ax1



#z1 = np.polyfit(df_e['water_year'].head(16), df_e['pearson_r'].head(16), 1)
#p1 = np.poly1d(z1)
#plt.plot(df_e['water_year'].head(16), p1(df_e['water_year'].head(16)),color='deeppink')


#z2 = np.polyfit(df_e['water_year'].tail(16), df_e['pearson_r'].tail(16), 1)
#p2 = np.poly1d(z2)
#plt.plot(df_e['water_year'].tail(16), p2(df_e['water_year'].tail(16)),color='pink',linestyle='--')


#z3 = np.polyfit(df_h['water_year'].head(8), df_h['pearson_r'].head(8), 1)
#p3 = np.poly1d(z3)
#plt.plot(df_h['water_year'].head(8), p3(df_h['water_year'].head(8)),color='teal')


#z4 = np.polyfit(df_h['water_year'].tail(10), df_h['pearson_r'].tail(10), 1)
#p4 = np.poly1d(z4)
#plt.plot(df_h['water_year'].tail(10), p4(df_h['water_year'].tail(10)),color='mediumturquoise',linestyle='--')


ax.set_title("Harvard Forest, EMS Tower (US-Ha1), HEM Tower (US-Ha2)",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
#ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
#fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
#plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()














#####################################################################################################
### pearson

df_e=yearly_metrics_ems
df_h=yearly_metrics_hem


### ems

x1=df_e['water_year'].head(16)
y1=df_e['pearson_r'].head(16)
mk.original_test(y1)


x1=df_e['water_year'].tail(16)
y1=df_e['pearson_r'].tail(16)
mk.original_test(y1)
#########################################

### hem

x1=df_h['water_year'].head(8)
y1=df_h['pearson_r'].head(8)
mk.original_test(y1)



x1=df_h['water_year'].tail(10)
y1=df_h['pearson_r'].tail(10)
mk.original_test(y1)


plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(df_e['water_year'].head(17), df_e['pearson_r'].head(17),color="blue",label="EMS (1993-2008), p=6.0*10$^{-3}$, slope=-0.01",linestyle='-', marker='*',markersize=7)
plt.plot(df_e['water_year'].tail(15), df_e['pearson_r'].tail(15),color="cornflowerblue",label="EMS (2008-2023), p=1.0*10$^{-2}$, slope= 0.01",linestyle='-', marker='*',markersize=7)

plt.plot(df_h['water_year'].head(9), df_h['pearson_r'].head(9),color="darkgreen",label="HEM (2006-2013), p=9.3*10$^{-3}$, slope=-0.03",linestyle='-', marker='.',markersize=8)
plt.plot(df_h['water_year'].tail(8), df_h['pearson_r'].tail(8),color="lime",label="HEM (2013-2023), p=7.6*10$^{-1}$, slope= 3.3",linestyle='-', marker='.',markersize=8)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
#ax.legend(loc='upper left', fontsize = '6',labelspacing=0.0005)
#ax.legend(loc='lower right', fontsize = '5',labelcolor='linecolor',labelspacing=0.0005)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),labelspacing=0.0005,labelcolor='linecolor')

ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('ET/GPP')  # we already handled the x-label with ax1

#ax.set_ylim(250,800)

z1 = np.polyfit(df_e['water_year'].head(16), df_e['pearson_r'].head(16), 1)
p1 = np.poly1d(z1)
plt.plot(df_e['water_year'].head(16), p1(df_e['water_year'].head(16)),color='blue')


z2 = np.polyfit(df_e['water_year'].tail(16), df_e['pearson_r'].tail(16), 1)
p2 = np.poly1d(z2)
plt.plot(df_e['water_year'].tail(16), p2(df_e['water_year'].tail(16)),color='cornflowerblue')


z3 = np.polyfit(df_h['water_year'].head(8), df_h['pearson_r'].head(8), 1)
p3 = np.poly1d(z3)
plt.plot(df_h['water_year'].head(8), p3(df_h['water_year'].head(8)),color='darkgreen')


z4 = np.polyfit(df_h['water_year'].tail(9), df_h['pearson_r'].tail(9), 1)
p4 = np.poly1d(z4)
plt.plot(df_h['water_year'].tail(9), p4(df_h['water_year'].tail(9)),color='lime')


ax.set_title("Harvard Forest, EMS Tower (US-Ha1), HEM Tower (US-Ha2)",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
#ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
#fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
#plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()














#########################################################################################
##GPP part 
################################################################################
def compute_yearly_metrics(group):
    x = group["day_of_year"].values  # Days in the year (1 to 365)
    y = group["GPP_g"].values  # Variable values

    # Linear regression to compute slope
    slope, _, _, _, _ = linregress(x, y)
    
    # Pearson correlation between day_of_year and var1
    corr, _ = pearsonr(x, y)
    
    return pd.Series({"slope": slope, "pearson_corr": corr})
# Group by year and apply function
yearly_metrics_gpp_hem = df.groupby("water_year").apply(compute_yearly_metrics).reset_index()
# Display results


plt.plot(yearly_metrics_ET_hem["water_year"],yearly_metrics_ET_hem["slope"] )

plt.plot(yearly_metrics_ET_hem["water_year"],yearly_metrics_ET_hem["slope"] )
plt.plot(yearly_metrics_gpp_hem["water_year"],yearly_metrics_gpp_hem["slope"] )

#######################################################################################################

df=pd.read_csv("HEM_daily.csv")

df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])

df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
dfh=df
df["month"]=df['TIMESTAMP'].dt.month
dfh=df


df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
mask = (df['water_year'] >=2005) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))

df["day_of_year"] = df["DoY"]  

# Function to compute slope and correlation

# Group by year and apply function
yearly_metrics_ET_hem = df.groupby("water_year").apply(compute_yearly_metrics).reset_index()

#########################################################################################
##GPP part 
################################################################################
def compute_yearly_metrics(group):
    x = group["day_of_year"].values  # Days in the year (1 to 365)
    y = group["GPP_g"].values  # Variable values

    # Linear regression to compute slope
    slope, _, _, _, _ = linregress(x, y)
    
    # Pearson correlation between day_of_year and var1
    corr, _ = pearsonr(x, y)
    
    return pd.Series({"slope": slope, "pearson_corr": corr})
# Group by year and apply function
yearly_metrics_gpp_hem = df.groupby("water_year").apply(compute_yearly_metrics).reset_index()
# Display results


plt.plot(yearly_metrics_ET_hem["water_year"],yearly_metrics_ET_hem["slope"] )
plt.plot(yearly_metrics_gpp_hem["water_year"],yearly_metrics_gpp_hem["slope"] )
#y1e=pd.DataFrame(dfe['ET'])

#############################################################################################################


x1=yearly_metrics_ET_hem["water_year"]
y1=yearly_metrics_ET_hem["slope"]

mk.original_test(y1)



def compute_yearly_metrics(group):
    x = group["day_of_year"].values  # Days in the year (1 to 365)
    y = group["GPP_g"].values  # Variable values

    # Linear regression to compute slope
    slope, _, _, _, _ = linregress(x, y)
    
    # Pearson correlation between day_of_year and var1
    corr, _ = pearsonr(x, y)
    
    return pd.Series({"slope": slope, "pearson_corr": corr})
# Group by year and apply function
yearly_metrics_gpp_ems = df.groupby("water_year").apply(compute_yearly_metrics).reset_index()
# Display results


plt.plot(yearly_metrics_ET_ems["water_year"],yearly_metrics_ET_ems["slope"] )
plt.plot(yearly_metrics_gpp_ems["water_year"],yearly_metrics_gpp_ems["slope"] )
#y1e=pd.DataFrame(dfe['ET'])
#x1e=pd.DataFrame(dfe['gee.e.6mol.m2.s'])


x1=df['water_year']
y1=df['WUE'].astype(float)

mk.original_test(y1)



##########################################################################################################
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
mask = (df['TIMESTAMP'] >='2005-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
df = df.loc[mask]
df.index = np.arange(0, len(df))

df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
dfh=df
df["month"]=df['TIMESTAMP'].dt.month
dfh=df

dfh=df[df['GPP'].notnull()]

y1h=pd.DataFrame(dfh['ET'])
x1h=pd.DataFrame(dfh['GPP'])


reg = LinearRegression(fit_intercept=True).fit(x1h, y1h)
reg.score(x1h,y1h)

pearsonr(dfh['ET'],dfh['GPP'])


plt.rcParams['figure.figsize'] = (3,3)
fig, ax = plt.subplots()

ax.plot(dfe['gee.e.6mol.m2.s']*-1,dfe['ET'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.40, Pearson=0.64",mfc='none')
ax.plot(dfh['GPP'],dfh['ET'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.46, Pearson=0.68",mfc='none')


z1 = np.polyfit(dfe['gee.e.6mol.m2.s'],dfe['ET'], 1)
p1 = np.poly1d(z1)
plt.plot(dfe['gee.e.6mol.m2.s']*-1, p1(dfe['gee.e.6mol.m2.s']),color='violet')
z2 = np.polyfit(dfh['GPP'], dfh['ET'],1)

p2 = np.poly1d(z2)
plt.plot(dfh['GPP'],p2(dfh['GPP']),color='salmon')
line = mlines.Line2D([1, 0], [1, 0], color='grey',linestyle='--',linewidth=1)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend()
ax.legend(loc='upper left', fontsize = '6')
ax.set_ylabel('ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('GPP (gC$m^{-2}$$day^{-1}$)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=8)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize =9,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 9,direction='in')
ax.set_aspect('auto')
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

##########################################################################################################


























#####################################################################################################
EMS_daily.to_csv('EMS_daily.csv', index=False, header=True)
HEM_daily.to_csv('HEM_daily.csv', index=False, header=True)


os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")

EMS_hr.to_csv('EMS_hr.csv', index=False, header=True)
HEM_hr.to_csv('HEM_hr.csv', index=False, header=True)


dfe=pd.read_csv("EMS_daily.csv")
dfh=pd.read_csv("HEM_daily.csv")

dfeh=pd.read_csv("EMS_hr.csv")
dfhh=pd.read_csv("HEM_hr.csv")

################################################################################################33
df=pd.read_csv("EMS_daily.csv")
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
mask = (df['TIMESTAMP'] >='1992-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
df = df.loc[mask]
df.index = np.arange(0, len(df))
dfe=df
x1=pd.DataFrame(dfe['ET'])
y1=pd.DataFrame(dfe['NETRAD_fill'])
reg = LinearRegression(fit_intercept=True).fit(x1, y1)
reg.score(x1,y1)
pearsonr(dfe['ET'],dfe['NETRAD_fill'])


plt.rcParams['figure.figsize'] = (5,5)
fig, ax = plt.subplots()
ax.plot(dfe['ET'],dfe['NETRAD_fill'],'x', color="crimson",markersize=10,marker='1',label="R2=0.59, Pearson=0.77",mfc='none')
#ax.plot(dfe['ET'],0.7329*dfe['ET']+0.0819,'-', linewidth=2, color='orangered', label='Fitted line (All seasons)')#overall trend line 
z1 = np.polyfit(dfe['ET'], dfe['NETRAD_fill'], 1)
p1 = np.poly1d(z1)
plt.plot(dfe['ET'], p1(dfe['ET']),color='deeppink')
line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend()
ax.legend(loc='upper left', fontsize = '12')

ax.set_xlabel('ET (mm)')  # we already handled the x-label with ax1
ax.set_ylabel('NETRAD (Wm-2)')  # we already handled the x-label with ax1


ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(13)
ax.xaxis.label.set_size(13) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 13,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 13,direction='in')
ax.set_aspect('auto')

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

##################################################################################################

ndex(['TIMESTAMP', 'DoY', 'Year', 'NETRAD_fill', 'VPD_fill', 'Tair_fill',
       'PAR_fill', 'WD_gaps', 'WS_Rd', 'PA_Rd', 'H_Rd', 'Rg_fill', 'TAU_gaps',
       'Ustar_gaps', 'nee.e.6mol.m2.s', 'gee.e.6mol.m2.s', 'sos', 'eos',
       'Month_cos', 'tsoil_upper_mean_gaps', 'LE_xg_Rd', 'ET', 'GPP_g'],
      dtype='object')

#####################################################################################################


#######################################################################################################


# growing season


os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")

df=pd.read_csv("EMS_daily.csv")
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
mask = (df['TIMESTAMP'] >='1992-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
df = df.loc[mask]
df.index = np.arange(0, len(df))

df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
dfe=df
df["month"]=df['TIMESTAMP'].dt.month
dfe=df

#dfe=df[df['WD_gaps'].notnull()]
#dfe=df[df['WS_Rd'].notnull()]
#dfe=df[df['PA_Rd'].notnull()]
#dfe=df[df['H_Rd'].notnull()]
#dfe=df[df['Rg_fill'].notnull()]
#dfe=df[df['PAR_fill'].notnull()]
dfe=df[df['nee.e.6mol.m2.s'].notnull()]
#dfe=df[df['gee.e.6mol.m2.s'].notnull()]


y1e=pd.DataFrame(dfe['ET'])
#x1e=pd.DataFrame(dfe['NETRAD_fill'])
#x1e=pd.DataFrame(dfe['VPD_fill'])
#x1e=pd.DataFrame(dfe['Tair_fill'])
#x1e=pd.DataFrame(dfe['PAR_fill'])
#x1e=pd.DataFrame(dfe['WD_fill'])
#x1e=pd.DataFrame(dfe['WS_Rd'])
#x1e=pd.DataFrame(dfe['PA_Rd'])
#x1e=pd.DataFrame(dfe['H_Rd'])
#x1e=pd.DataFrame(dfe['Rg_fill'])
x1e=pd.DataFrame(dfe['nee.e.6mol.m2.s'])
#x1e=pd.DataFrame(dfe['gee.e.6mol.m2.s'])

reg = LinearRegression(fit_intercept=True).fit(x1e, y1e)
reg.score(x1e,y1e)

#pearsonr(dfe['NETRAD_fill'],dfe['ET'])
#pearsonr(dfe['ET'],dfe['VPD_fill'])
#pearsonr(dfe['ET'],dfe['Tair_fill'])
#pearsonr(dfe['ET'],dfe['PAR_fill'])
#y1e=pd.DataFrame(dfe['WD_fill'])
#pearsonr(dfe['ET'],dfe['WS_Rd'])
#pearsonr(dfe['ET'],dfe['PA_Rd'])
#pearsonr(dfe['ET'],dfe['H_Rd'])
#pearsonr(dfe['ET'],dfe['Rg_fill'])
pearsonr(dfe['ET'],dfe['nee.e.6mol.m2.s'])
#pearsonr(dfe['ET'],dfe['gee.e.6mol.m2.s']*-1)

#######################################################################################################
#['TIMESTAMP', 'DoY', 'Year', 'WD', 'Ustar', 'TS_PI_1', 'NEE', 'H', 'WS',
 #      'Tair', 'VPD', 'PPFD', 'GPP', 'sos', 'eos', 'Month_cos', 'PA', 'Rg',
  #     'NETRAD', 'ET', 'GPP_g', 'LE_fill'],
   #   dtype='object')


#######################################################################################################

df=pd.read_csv("HEM_daily.csv")
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
mask = (df['TIMESTAMP'] >='2005-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
df = df.loc[mask]
df.index = np.arange(0, len(df))

df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
dfh=df
df["month"]=df['TIMESTAMP'].dt.month
dfh=df

#mask = (df['VPD'] <= 2.5) 
#df = df.loc[mask]
#df.index = np.arange(0, len(df))
#dfh=df

#dfh=df[df['WD'].notnull()]
#dfh=df[df['WS'].notnull()]
#dfh=df[df['PA'].notnull()]
#dfh=df[df['H'].notnull()]
#dfh=df[df['Rg'].notnull()]
dfh=df[df['NEE'].notnull()]
#dfh=df[df['GPP'].notnull()]
#dfh=df[df['PPFD'].notnull()]

y1h=pd.DataFrame(dfh['ET'])
#x1h=pd.DataFrame(dfh['NETRAD'])
#x1h=pd.DataFrame(dfh['VPD'])
#x1h=pd.DataFrame(dfh['Tair'])
#x1h=pd.DataFrame(dfh['PPFD'])
#x1h=pd.DataFrame(dfh['WD'])
#x1h=pd.DataFrame(dfh['WS'])
#x1h=pd.DataFrame(dfh['PA'])
#x1h=pd.DataFrame(dfh['H'])
#x1h=pd.DataFrame(dfh['Rg'])
x1h=pd.DataFrame(dfh['NEE'])
#x1h=pd.DataFrame(dfh['GPP'])


reg = LinearRegression(fit_intercept=True).fit(x1h, y1h)
reg.score(x1h,y1h)

#pearsonr(dfh['NETRAD'],dfh['ET'])
#pearsonr(dfh['ET'],dfh['VPD'])
#pearsonr(dfh['ET'],dfh['Tair'])
#pearsonr(dfh['ET'],dfh['PPFD'])
#pearsonr(dfh['ET'],dfh['WD'])
#pearsonr(dfh['ET'],dfh['WS'])
#pearsonr(dfh['ET'],dfh['PA'])
#pearsonr(dfh['ET'],dfh['H'])
#pearsonr(dfh['Rg'],dfh['ET'])
pearsonr(dfh['ET'],dfh['NEE'])
#pearsonr(dfh['ET'],dfh['GPP'])


plt.rcParams['figure.figsize'] = (3,3)
fig, ax = plt.subplots()

#ax.plot(dfe['NETRAD_fill'],dfe['ET'],'x', color="violet",markersize=8,marker='1',label="EMS: R$^{2}$=0.60, Pearson=0.78",mfc='none')
#ax.plot(dfh['NETRAD'],dfh['ET'],'x', color="salmon",markersize=6,marker='s',label="HEM: R$^{2}$=0.79, Pearson=0.89",mfc='none')
#ax.plot(dfe['Rg_fill'],dfe['ET'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.58, Pearson=0.76",mfc='none')
#ax.plot(dfh['Rg'],dfh['ET'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.75, Pearson=0.86",mfc='none')

#ax.plot(dfe['ET'],dfe['VPD_fill'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.32, Pearson=0.57",mfc='none')
#ax.plot(dfh['ET'],dfh['VPD'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.35, Pearson=0.60",mfc='none')

#ax.plot(dfe['ET'],dfe['Tair_fill'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.48, Pearson=0.70",mfc='none')
#ax.plot(dfh['ET'],dfh['Tair'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.56, Pearson=0.75",mfc='none')

#ax.plot(dfe['PAR_fill'],dfe['ET'], 'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.60, Pearson=0.77",mfc='none')
#ax.plot(dfh['PPFD'],dfh['ET'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.62, Pearson=0.79",mfc='none')

#ax.plot(dfe['ET'],dfe['WD_gaps'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.11, Pearson=0.20",mfc='none')
#ax.plot(dfh['ET'],dfh['WD'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.0075, Pearson=0.087",mfc='none')

#ax.plot(dfe['ET'],dfe['WS_Rd'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.002, Pearson=-0.04",mfc='none')
#ax.plot(dfh['ET'],dfh['WS'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.01, Pearson=-0.10",mfc='none')

#ax.plot(dfe['ET'],dfe['PA_Rd'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=2.4*10^-6, Pearson=0.0016",mfc='none')
#ax.plot(dfh['ET'],dfh['PA'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=7.7*106-5, Pearson=0.009",mfc='none')

#ax.plot(dfe['ET'],dfe['H_Rd'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.26, Pearson=0.51",mfc='none')
#ax.plot(dfh['ET'],dfh['H'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.30, Pearson=0.55",mfc='none')

#ax.plot(dfe['nee.e.6mol.m2.s'],dfe['ET'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.30, Pearson=-0.55",mfc='none')
#ax.plot(dfh['NEE'], dfh['ET'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.24, Pearson=-0.49",mfc='none')

ax.plot(dfe['gee.e.6mol.m2.s']*-1,dfe['ET'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.40, Pearson=0.64",mfc='none')
ax.plot(dfh['GPP'],dfh['ET'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.46, Pearson=0.68",mfc='none')


#z1 = np.polyfit(dfe['NETRAD_fill'],dfe['ET'], 1)
#z1 = np.polyfit(dfe['Rg_fill'],dfe['ET'], 1)
#z1 = np.polyfit(dfe['PAR_fill'],dfe['ET'], 1)
#z1 = np.polyfit(dfe['nee.e.6mol.m2.s'],dfe['ET'], 1)
z1 = np.polyfit(dfe['gee.e.6mol.m2.s'],dfe['ET'], 1)
#z1 = np.polyfit(dfe['ET'], dfe['H_Rd'], 1)
p1 = np.poly1d(z1)
#plt.plot(dfe['NETRAD_fill'], p1(dfe['NETRAD_fill']),color='violet')
#plt.plot(dfe['Rg_fill'], p1(dfe['Rg_fill']),color='violet')
#plt.plot(dfe['PAR_fill'], p1(dfe['PAR_fill']),color='violet')
#plt.plot(dfe['nee.e.6mol.m2.s'], p1(dfe['nee.e.6mol.m2.s']),color='violet')
plt.plot(dfe['gee.e.6mol.m2.s']*-1, p1(dfe['gee.e.6mol.m2.s']),color='violet')


#z2 = np.polyfit(dfh['ET'], dfh['NETRAD'], 1)
#z2 = np.polyfit(dfh['Rg'],dfh['ET'], 1)
#z2 = np.polyfit(dfh['PPFD'],dfh['ET'], 1)
#z2 = np.polyfit(dfh['NEE'],dfh['ET'], 1)
z2 = np.polyfit(dfh['GPP'], dfh['ET'],1)
#z2 = np.polyfit(dfh['NETRAD'], dfh['ET'], 1)

p2 = np.poly1d(z2)
#plt.plot(dfh['NETRAD'],p2(dfh['NETRAD']),color='salmon')
#plt.plot(dfh['Rg'],p2(dfh['Rg']),color='salmon')
#plt.plot(dfh['PPFD'],p2(dfh['PPFD']),color='salmon')
#plt.plot(dfh['NEE'],p2(dfh['NEE']),color='salmon')

#plt.plot(dfh['NEE'],p2(dfh['NEE']),color='salmon')
plt.plot(dfh['GPP'],p2(dfh['GPP']),color='salmon')
#line = mlines.Line2D([0, 1], [1, 0], color='grey',linestyle='--',linewidth=1)
line = mlines.Line2D([1, 0], [1, 0], color='grey',linestyle='--',linewidth=1)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend()
#ax.legend(loc='lower left', fontsize = '6')
ax.legend(loc='upper left', fontsize = '6')
ax.set_ylabel('ET (mm)')  # we already handled the x-label with ax1
#ax.set_xlabel('NETRAD (Wm$^{-2}$)')  # we already handled the x-label with ax1
#ax.set_xlabel('Rg (Wm$^{-2}$)')  # we already handled the x-label with ax1
#ax.set_xlabel('PAR (µmol $m^{-2}$ $day^{-1}$)')  # we already handled the x-label with ax1
#gCm-2 day-1
#ax.set_ylim([-1,8])
#ax.set_xlabel('VPD (kpa)')  # we already handled the x-label with ax1
#ax.set_xlabel('Tair (oc)')  # we already handled the x-label with ax1
#ax.set_xlabel('WD')  # we already handled the x-label with ax1
#ax.set_xlabel('WS')  # we already handled the x-label with ax1
#ax.set_xlabel('Pressure (kap)')  # we already handled the x-label with ax1
#ax.set_xlabel('H (wm-2)')  # we already handled the x-label with ax1
#ax.set_xlabel('NEE (gC$m^{-2}$$day^{-1}$)')  # we already handled the x-label with ax1
ax.set_xlabel('GPP (gC$m^{-2}$$day^{-1}$)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=8)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize =9,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 9,direction='in')
ax.set_aspect('auto')
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

#####################################################################################################

### non growing seaon

####################################################################################################



os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")

df=pd.read_csv("EMS_daily.csv")
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
mask = (df['TIMESTAMP'] >='1992-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
df = df.loc[mask]
df.index = np.arange(0, len(df))

df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'NGS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
dfe=df
df["month"]=df['TIMESTAMP'].dt.month
dfe=df

#dfe=df[df['WD_gaps'].notnull()]
#dfe=df[df['WS_Rd'].notnull()]
#dfe=df[df['PA_Rd'].notnull()]
#dfe=df[df['H_Rd'].notnull()]
#dfe=df[df['Rg_fill'].notnull()]
#dfe=df[df['PAR_fill'].notnull()]
#dfe=df[df['nee.e.6mol.m2.s'].notnull()]
dfe=df[df['gee.e.6mol.m2.s'].notnull()]


y1e=pd.DataFrame(dfe['ET'])
#x1e=pd.DataFrame(dfe['NETRAD_fill'])
#x1e=pd.DataFrame(dfe['VPD_fill'])
#x1e=pd.DataFrame(dfe['Tair_fill'])
#x1e=pd.DataFrame(dfe['PAR_fill'])
#x1e=pd.DataFrame(dfe['WD_fill'])
#x1e=pd.DataFrame(dfe['WS_Rd'])
#x1e=pd.DataFrame(dfe['PA_Rd'])
#x1e=pd.DataFrame(dfe['H_Rd'])
#x1e=pd.DataFrame(dfe['Rg_fill'])
#x1e=pd.DataFrame(dfe['nee.e.6mol.m2.s'])
x1e=pd.DataFrame(dfe['gee.e.6mol.m2.s'])

reg = LinearRegression(fit_intercept=True).fit(x1e, y1e)
reg.score(x1e,y1e)

#pearsonr(dfe['NETRAD_fill'],dfe['ET'])
#pearsonr(dfe['ET'],dfe['VPD_fill'])
#pearsonr(dfe['ET'],dfe['Tair_fill'])
#pearsonr(dfe['ET'],dfe['PAR_fill'])
#y1e=pd.DataFrame(dfe['WD_fill'])
#pearsonr(dfe['ET'],dfe['WS_Rd'])
#pearsonr(dfe['ET'],dfe['PA_Rd'])
#pearsonr(dfe['ET'],dfe['H_Rd'])
#pearsonr(dfe['ET'],dfe['Rg_fill'])
#pearsonr(dfe['ET'],dfe['nee.e.6mol.m2.s'])
pearsonr(dfe['ET'],dfe['gee.e.6mol.m2.s']*-1)

#######################################################################################################
#['TIMESTAMP', 'DoY', 'Year', 'WD', 'Ustar', 'TS_PI_1', 'NEE', 'H', 'WS',
 #      'Tair', 'VPD', 'PPFD', 'GPP', 'sos', 'eos', 'Month_cos', 'PA', 'Rg',
  #     'NETRAD', 'ET', 'GPP_g', 'LE_fill'],
   #   dtype='object')


#######################################################################################################

df=pd.read_csv("HEM_daily.csv")
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
mask = (df['TIMESTAMP'] >='2005-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
df = df.loc[mask]
df.index = np.arange(0, len(df))

df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'NGS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
dfh=df
df["month"]=df['TIMESTAMP'].dt.month
dfh=df

#mask = (df['VPD'] <= 2.5) 
#df = df.loc[mask]
#df.index = np.arange(0, len(df))
#dfh=df

#dfh=df[df['WD'].notnull()]
#dfh=df[df['WS'].notnull()]
#dfh=df[df['PA'].notnull()]
#dfh=df[df['H'].notnull()]
#dfh=df[df['Rg'].notnull()]
#dfh=df[df['NEE'].notnull()]
dfh=df[df['GPP'].notnull()]
#dfh=df[df['PPFD'].notnull()]

y1h=pd.DataFrame(dfh['ET'])
#x1h=pd.DataFrame(dfh['NETRAD'])
#x1h=pd.DataFrame(dfh['VPD'])
#x1h=pd.DataFrame(dfh['Tair'])
#x1h=pd.DataFrame(dfh['PPFD'])
#x1h=pd.DataFrame(dfh['WD'])
#x1h=pd.DataFrame(dfh['WS'])
#x1h=pd.DataFrame(dfh['PA'])
#x1h=pd.DataFrame(dfh['H'])
#x1h=pd.DataFrame(dfh['Rg'])
#x1h=pd.DataFrame(dfh['NEE'])
x1h=pd.DataFrame(dfh['GPP'])


reg = LinearRegression(fit_intercept=True).fit(x1h, y1h)
reg.score(x1h,y1h)

#pearsonr(dfh['NETRAD'],dfh['ET'])
#pearsonr(dfh['ET'],dfh['VPD'])
#pearsonr(dfh['ET'],dfh['Tair'])
#pearsonr(dfh['ET'],dfh['PPFD'])
#pearsonr(dfh['ET'],dfh['WD'])
#pearsonr(dfh['ET'],dfh['WS'])
#pearsonr(dfh['ET'],dfh['PA'])
#pearsonr(dfh['ET'],dfh['H'])
#pearsonr(dfh['Rg'],dfh['ET'])
#pearsonr(dfh['ET'],dfh['NEE'])
pearsonr(dfh['ET'],dfh['GPP'])



plt.rcParams['figure.figsize'] = (3,3)
fig, ax = plt.subplots()

#ax.plot(dfe['NETRAD_fill'],dfe['ET'],'x', color="violet",markersize=8,marker='1',label="EMS: R$^{2}$=0.31, Pearson=0.56",mfc='none')
#ax.plot(dfh['NETRAD'],dfh['ET'],'x', color="salmon",markersize=6,marker='s',label="HEM: R$^{2}$=0.62, Pearson=0.79",mfc='none')

#ax.plot(dfe['Rg_fill'],dfe['ET'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.33, Pearson=0.57",mfc='none')
#ax.plot(dfh['Rg'],dfh['ET'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.56, Pearson=0.74",mfc='none')

#ax.plot(dfe['ET'],dfe['VPD_fill'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.32, Pearson=0.57",mfc='none')
#ax.plot(dfh['ET'],dfh['VPD'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.35, Pearson=0.60",mfc='none')

#ax.plot(dfe['ET'],dfe['Tair_fill'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.48, Pearson=0.70",mfc='none')
#ax.plot(dfh['ET'],dfh['Tair'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.56, Pearson=0.75",mfc='none')

#ax.plot(dfe['PAR_fill'],dfe['ET'], 'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.34, Pearson=0.58",mfc='none')
#ax.plot(dfh['PPFD'],dfh['ET'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.51, Pearson=0.71",mfc='none')

#ax.plot(dfe['ET'],dfe['WD_gaps'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.11, Pearson=0.20",mfc='none')
#ax.plot(dfh['ET'],dfh['WD'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.0075, Pearson=0.087",mfc='none')

#ax.plot(dfe['ET'],dfe['WS_Rd'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.002, Pearson=-0.04",mfc='none')
#ax.plot(dfh['ET'],dfh['WS'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.01, Pearson=-0.10",mfc='none')

#ax.plot(dfe['ET'],dfe['PA_Rd'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=2.4*10^-6, Pearson=0.0016",mfc='none')
#ax.plot(dfh['ET'],dfh['PA'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=7.7*106-5, Pearson=0.009",mfc='none')

#ax.plot(dfe['ET'],dfe['H_Rd'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.26, Pearson=0.51",mfc='none')
#ax.plot(dfh['ET'],dfh['H'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.30, Pearson=0.55",mfc='none')

#ax.plot(dfe['nee.e.6mol.m2.s'],dfe['ET'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.01, Pearson=-0.11",mfc='none')
#ax.plot(dfh['NEE'], dfh['ET'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.56, Pearson=-0.75",mfc='none')

ax.plot(dfe['gee.e.6mol.m2.s']*-1,dfe['ET'],'x', color="violet",markersize=8,marker='1',label="EMS, R$^{2}$=0.31, Pearson=0.55",mfc='none')
ax.plot(dfh['GPP'],dfh['ET'],'x', color="salmon",markersize=6,marker='s',label="HEM, R$^{2}$=0.66, Pearson=0.81",mfc='none')


#z1 = np.polyfit(dfe['NETRAD_fill'],dfe['ET'], 1)
#z1 = np.polyfit(dfe['Rg_fill'],dfe['ET'], 1)
#z1 = np.polyfit(dfe['PAR_fill'],dfe['ET'], 1)
#z1 = np.polyfit(dfe['nee.e.6mol.m2.s'],dfe['ET'], 1)
z1 = np.polyfit(dfe['gee.e.6mol.m2.s'],dfe['ET'], 1)
#z1 = np.polyfit(dfe['ET'], dfe['H_Rd'], 1)
p1 = np.poly1d(z1)
#plt.plot(dfe['NETRAD_fill'], p1(dfe['NETRAD_fill']),color='violet')
#plt.plot(dfe['Rg_fill'], p1(dfe['Rg_fill']),color='violet')
#plt.plot(dfe['PAR_fill'], p1(dfe['PAR_fill']),color='violet')
#plt.plot(dfe['nee.e.6mol.m2.s'], p1(dfe['nee.e.6mol.m2.s']),color='violet')
plt.plot(dfe['gee.e.6mol.m2.s']*-1, p1(dfe['gee.e.6mol.m2.s']),color='violet')


#z2 = np.polyfit(dfh['ET'], dfh['NETRAD'], 1)
#z2 = np.polyfit(dfh['Rg'],dfh['ET'], 1)
#z2 = np.polyfit(dfh['PPFD'],dfh['ET'], 1)
#z2 = np.polyfit(dfh['NEE'],dfh['ET'], 1)
z2 = np.polyfit(dfh['GPP'], dfh['ET'],1)
#z2 = np.polyfit(dfh['NETRAD'], dfh['ET'], 1)

p2 = np.poly1d(z2)
#plt.plot(dfh['NETRAD'],p2(dfh['NETRAD']),color='salmon')
#plt.plot(dfh['Rg'],p2(dfh['Rg']),color='salmon')
#plt.plot(dfh['PPFD'],p2(dfh['PPFD']),color='salmon')
#plt.plot(dfh['NEE'],p2(dfh['NEE']),color='salmon')

#plt.plot(dfh['NEE'],p2(dfh['NEE']),color='salmon')
plt.plot(dfh['GPP'],p2(dfh['GPP']),color='salmon')
#line = mlines.Line2D([0, 1], [1, 0], color='grey',linestyle='--',linewidth=1)
line = mlines.Line2D([1, 0], [1, 0], color='grey',linestyle='--',linewidth=1)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend()
#ax.legend(loc='lower left', fontsize = '6')
ax.legend(loc='upper left', fontsize = '6')
ax.set_ylabel('ET (mm)')  # we already handled the x-label with ax1
#ax.set_xlabel('NETRAD (Wm$^{-2}$)')  # we already handled the x-label with ax1
#ax.set_xlabel('Rg (Wm$^{-2}$)')  # we already handled the x-label with ax1
#ax.set_xlabel('PAR (µmol $m^{-2}$ $day^{-1}$)')  # we already handled the x-label with ax1
#gCm-2 day-1
#ax.set_ylim([-1,8])
#ax.set_xlabel('VPD (kpa)')  # we already handled the x-label with ax1
#ax.set_xlabel('Tair (oc)')  # we already handled the x-label with ax1
#ax.set_xlabel('WD')  # we already handled the x-label with ax1
#ax.set_xlabel('WS')  # we already handled the x-label with ax1
#ax.set_xlabel('Pressure (kap)')  # we already handled the x-label with ax1
#ax.set_xlabel('H (wm-2)')  # we already handled the x-label with ax1
#ax.set_xlabel('NEE (gC$m^{-2}$$day^{-1}$)')  # we already handled the x-label with ax1
ax.set_xlabel('GPP (gC$m^{-2}$$day^{-1}$)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=8)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize =9,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 9,direction='in')
ax.set_aspect('auto')
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


#######################################################################################################
































########################################################################################################

# by month 
###############################################################################################
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")

df=pd.read_csv("EMS_daily.csv")
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
mask = (df['TIMESTAMP'] >='1992-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["month"]=df['TIMESTAMP'].dt.month

mask = (df['month'] ==6) 
df = df.loc[mask]
df.index = np.arange(0, len(df))
df=df
dfe=df

#dfe=df[df['Rg_fill'].notnull()]
#dfe=df[df['nee.e.6mol.m2.s'].notnull()]
#dfe=df[df['gee.e.6mol.m2.s'].notnull()]
dfe=df[df['H_Rd'].notnull()]

y1e=pd.DataFrame(dfe['ET'])
x1e=pd.DataFrame(dfe['NETRAD_fill'])
#y1e=pd.DataFrame(dfe['Rg_fill'])
#y1e=pd.DataFrame(dfe['PAR_fill'])
#y1e=pd.DataFrame(dfe['nee.e.6mol.m2.s'])
#y1e=pd.DataFrame(dfe['gee.e.6mol.m2.s'])
y1e=pd.DataFrame(dfe['ET'])
reg = LinearRegression(fit_intercept=True).fit(x1e, y1e)
reg.score(x1e,y1e)


#pearsonr(dfe['ET'],dfe['NETRAD_fill'])
#pearsonr(dfe['ET'],dfe['Rg_fill'])
#pearsonr(dfe['ET'],dfe['PAR_fill'])
#pearsonr(dfe['ET'],dfe['nee.e.6mol.m2.s'])
#pearsonr(dfe['ET'],dfe['gee.e.6mol.m2.s']*-1)
#pearsonr(dfe['ET'],dfe['H_Rd'])



df=pd.read_csv("HEM_daily.csv")
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
mask = (df['TIMESTAMP'] >='2005-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
df = df.loc[mask]
df.index = np.arange(0, len(df))

df["month"]=df['TIMESTAMP'].dt.month

mask = (df['month'] ==6) 
df = df.loc[mask]
df.index = np.arange(0, len(df))
df=df
dfh=df


#mask = (df['VPD'] <= 2.5) 
#df = df.loc[mask]
#df.index = np.arange(0, len(df))
#dfh=df


#dfh=df[df['Rg'].notnull()]
#dfh=df[df['NEE'].notnull()]
#dfh=df[df['GPP'].notnull()]
dfh=df[df['H'].notnull()]


y1h=pd.DataFrame(dfh['ET'])
x1h=pd.DataFrame(dfh['NETRAD'])
#y1h=pd.DataFrame(dfh['Rg'])
#y1h=pd.DataFrame(dfh['PPFD'])
#y1h=pd.DataFrame(dfh['NEE'])
#y1h=pd.DataFrame(dfh['GPP'])
#y1h=pd.DataFrame(dfh['H'])
reg = LinearRegression(fit_intercept=True).fit(x1h, y1h)
reg.score(x1h,y1h)


#pearsonr(dfh['ET'],dfh['NETRAD'])
#pearsonr(dfh['ET'],dfh['Rg'])
#pearsonr(dfh['ET'],dfh['PPFD'])
#pearsonr(dfh['ET'],dfh['NEE'])
#pearsonr(dfh['ET'],dfh['GPP'])
#pearsonr(dfh['ET'],dfh['H'])



plt.rcParams['figure.figsize'] = (5,5)
fig, ax = plt.subplots()
ax.plot(dfe['NETRAD_fill'],dfe['ET'],'x', color="violet",markersize=10,marker='1',label="EMS: R2=0.54, Pearson=0.76",mfc='none')
ax.plot(dfh['NETRAD'],dfh['ET'],'x', color="salmon",markersize=7,marker='s',label="HEM: R2=0.50, Pearson=0.70",mfc='none')

#ax.plot(dfe['ET'],dfe['Rg_fill'],'x', color="violet",markersize=10,marker='1',label="EMS, R2=0.52, Pearson=0.72",mfc='none')
#ax.plot(dfh['ET'],dfh['Rg'],'x', color="salmon",markersize=7,marker='s',label="HEM, R2=0.47, Pearson=0.68",mfc='none')


#ax.plot(dfe['ET'],dfe['PAR_fill'],'x', color="violet",markersize=10,marker='1',label="EMS, R2=0.35, Pearson=0.59",mfc='none')
#ax.plot(dfh['ET'],dfh['PPFD'],'x', color="salmon",markersize=7,marker='s',label="HEM, R2=0.32, Pearson=0.56",mfc='none')



#ax.plot(dfe['ET'],dfe['nee.e.6mol.m2.s'],'x', color="violet",markersize=10,marker='1',label="EMS, R2=0.14, Pearson=-0.37",mfc='none')
#ax.plot(dfh['ET'],dfh['NEE'],'x', color="salmon",markersize=7,marker='s',label="HEM, R2=0.17, Pearson=-0.41",mfc='none')

#ax.plot(dfe['ET'],dfe['gee.e.6mol.m2.s']*-1,'x', color="violet",markersize=10,marker='1',label="EMS, R2=0.32, Pearson=0.56",mfc='none')
#ax.plot(dfh['ET'],dfh['GPP'],'x', color="salmon",markersize=7,marker='s',label="HEM, R2=0.19, Pearson=0.43",mfc='none')

#ax.plot(dfe['ET'],dfe['H_Rd'],'x', color="violet",markersize=10,marker='1',label="EMS, R2=0.18, Pearson=0.43",mfc='none')
#ax.plot(dfh['ET'],dfh['H'],'x', color="salmon",markersize=7,marker='s',label="HEM, R2=0.46, Pearson=0.68",mfc='none')


z1 = np.polyfit(dfe['NETRAD_fill'],dfe['ET'], 1)
#z1 = np.polyfit(dfe['ET'], dfe['Rg_fill'], 1)
#z1 = np.polyfit(dfe['ET'], dfe['PAR_fill'], 1)
#z1 = np.polyfit(dfe['ET'], dfe['nee.e.6mol.m2.s'], 1)
#z1 = np.polyfit(dfe['ET'], dfe['gee.e.6mol.m2.s']*-1, 1)
#z1 = np.polyfit(dfe['ET'], dfe['H_Rd'], 1)

p1 = np.poly1d(z1)
plt.plot(dfe['NETRAD_fill'], p1(dfe['NETRAD_fill']),color='violet')

#z2 = np.polyfit(dfh['ET'], dfh['NETRAD'], 1)
#2 = np.polyfit(dfh['ET'], dfh['Rg'], 1)
#z2 = np.polyfit(dfh['ET'], dfh['PPFD'], 1)
#z2 = np.polyfit(dfh['ET'], dfh['NEE'], 1)
#z2 = np.polyfit(dfh['ET'], dfh['GPP'], 1)
z2 = np.polyfit(dfh['NETRAD'], dfh['ET'], 1)

p2 = np.poly1d(z2)
plt.plot(dfh['NETRAD'],p2(dfh['NETRAD']),color='salmon')

#line = mlines.Line2D([0, 1], [1, 0], color='grey',linestyle='--',linewidth=1)
line = mlines.Line2D([1, 0], [1, 0], color='grey',linestyle='--',linewidth=1)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.set_xlabel('ET (mm)')  # we already handled the x-label with ax1
#ax.set_ylabel('NETRAD (Wm-2)')  # we already handled the x-label with ax1
#ax.set_ylabel('Rg (wm-2)')  # we already handled the x-label with ax1
#ax.set_ylabel('PAR')  # we already handled the x-label with ax1
#ax.set_ylabel('NEE (mol.m2.s)')  # we already handled the x-label with ax1
#ax.set_ylabel('GPP')  # we already handled the x-label with ax1
ax.set_ylabel('H (wm-2)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=9)
ax.locator_params(axis='x', nbins=9)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

#######################################################################################################



#y1e=pd.DataFrame(dfe['NETRAD_fill'])
#y1e=pd.DataFrame(dfe['Rg_fill'])
#y1e=pd.DataFrame(dfe['PAR_fill'])
#y1e=pd.DataFrame(dfe['nee.e.6mol.m2.s'])
#y1e=pd.DataFrame(dfe['gee.e.6mol.m2.s'])
#y1e=pd.DataFrame(dfe['H_Rd'])


#pearsonr(dfh['ET'],dfh['NETRAD'])
#pearsonr(dfh['ET'],dfh['Rg'])
#pearsonr(dfh['ET'],dfh['PPFD'])
#pearsonr(dfh['ET'],dfh['NEE'])
#pearsonr(dfh['ET'],dfh['GPP'])
#pearsonr(dfh['ET'],dfh['H'])

#dfh=df[df['Rg'].notnull()]
#dfh=df[df['NEE'].notnull()]
#dfh=df[df['GPP'].notnull()]
#dfh=df[df['H'].notnull()]


#mk.original_test(y1)

#EMS
#1993-2008
#2008-2023

#change point 2010
#Hem
#2005-2014
#2015-2023

#2006-2014 pre-during HWA
#2015-2023 post HWA




os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")
df=pd.read_csv("EMS_daily.csv")
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
mask = (df['TIMESTAMP'] >='1992-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
df= df.groupby(['water_year']).mean()
df=df.reset_index()
dfe=df


df=pd.read_csv("HEM_daily.csv")
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
mask = (df['TIMESTAMP'] >='2005-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
df= df.groupby(['water_year']).mean()
df=df.reset_index()
dfh=df

hg.pettitt_test(dfe.NETRAD_fill)

hg.pettitt_test(dfh.NETRAD)

plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(dfe['water_year'], dfe['NETRAD_fill'],color="blue",label="EMS",linestyle='-', marker='*',markersize=7)
#plt.plot(dfe['water_year'], dfe['ET'],color="cornflowerblue",label="EMS (2008-2023), p=6.9*10$^{-1}$, slope=4.1",linestyle='-', marker='*',markersize=7)

plt.plot(dfh['water_year'], dfh['NETRAD'],color="darkgreen",label="HEM",linestyle='-', marker='.',markersize=8)
#plt.plot(dfH['water_year'], dfh['ET'],color="lime",label="HEM (2013-2023), p=7.6*10$^{-1}$, slope=3.3",linestyle='-', marker='.',markersize=8)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
#ax.legend(loc='upper left', fontsize = '6',labelspacing=0.0005)
ax.legend(loc='upper left', fontsize = '6',labelcolor='linecolor',labelspacing=0.0005)
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('NETRAD')  # we already handled the x-label with ax1

z1 = np.polyfit(dfe['water_year'], dfe['ET'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e['water_year'].head(16), p1(df_e['water_year'].head(16)),color='blue')



#ax.set_title("Harvard Forest, EMS Tower (US-Ha1), HEM Tower (US-Ha2)",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
#fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
#plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#ax.set_ylim(250,800)

z1 = np.polyfit(dfe['water_year'], dfe['ET'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e['water_year'].head(16), p1(df_e['water_year'].head(16)),color='blue')


z2 = np.polyfit(df_e['water_year'].tail(16), df_e['ET'].tail(16), 1)
p2 = np.poly1d(z2)
plt.plot(df_e['water_year'].tail(16), p2(df_e['water_year'].tail(16)),color='cornflowerblue')


z3 = np.polyfit(df_h['water_year'].head(9), df_h['ET'].head(9), 1)
p3 = np.poly1d(z3)
plt.plot(df_h['water_year'].head(9), p3(df_h['water_year'].head(9)),color='darkgreen')


z4 = np.polyfit(df_h['water_year'].tail(11), df_h['ET'].tail(11), 1)
p4 = np.poly1d(z4)
plt.plot(df_h['water_year'].tail(11), p4(df_h['water_year'].tail(11)),color='lime')

#####################################################################################################
## slope year graph

# https://stackoverflow.com/questions/56030933/compute-slope-for-each-point-in-dataframe
########################################################################################################
#import statsmodels.api as sm
#from collections import ChainMap  

os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper\results")

dfe=pd.read_csv("EMS_daily.csv")
dfh=pd.read_csv("HEM_daily.csv")


################################################################################################33

EMS_daily=dfe
HEM_daily=dfh


dfe=pd.read_csv("EMS_daily.csv")
#dfe=dfe[dfe['H_Rd'].notnull()]
dfe=dfe[dfe['nee.e.6mol.m2.s'].notnull()]
df=dfe

df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
mask = (df['TIMESTAMP'] >='1992-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
df=df.groupby(df['water_year']).mean()
df=df.reset_index()

#x1=pd.DataFrame(df['NETRAD_fill'])
#x1=pd.DataFrame(df['Rg_fill'])
#x1=pd.DataFrame(df['PAR_fill'])
x1=pd.DataFrame(df['nee.e.6mol.m2.s'])
#x1=pd.DataFrame(df['gee.e.6mol.m2.s'])
#x1=pd.DataFrame(df['H_Rd'])

y1=pd.DataFrame(df['ET'])
df['denom'] = x1.diff()
df['num']= y1.diff()
df['slope'] = (df.num/df.denom).fillna(0)
df = df.iloc[1:]
x1=df['water_year']
y1=df['slope'].astype(float)
mk.original_test(y1)

hg.pettitt_test(y1)





plt.plot(x1,y1)


######################################################################################################

plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(x1, y1,color="blue",label="EMS",linestyle='-', marker='*',markersize=7)
#plt.plot(dfe['water_year'], dfe['ET'],color="cornflowerblue",label="EMS (2008-2023), p=6.9*10$^{-1}$, slope=4.1",linestyle='-', marker='*',markersize=7)

#plt.plot(dfh['water_year'], dfh['NETRAD'],color="darkgreen",label="HEM",linestyle='-', marker='.',markersize=8)
#plt.plot(dfH['water_year'], dfh['ET'],color="lime",label="HEM (2013-2023), p=7.6*10$^{-1}$, slope=3.3",linestyle='-', marker='.',markersize=8)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
#ax.legend(loc='upper left', fontsize = '6',labelspacing=0.0005)
ax.legend(loc='upper left', fontsize = '10',labelcolor='linecolor',labelspacing=0.0005)
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('ET~NEE')  # we already handled the x-label with ax1

z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='blue')



#ax.set_title("Harvard Forest, EMS Tower (US-Ha1), HEM Tower (US-Ha2)",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
#fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
#plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#######################################################################################################
e_sl_d=df
s=df.groupby(df['water_year']).mean()
e_sl_y=s.reset_index()

x1=df['water_year']
y1=df['slope'].astype(float)
mk.original_test(y1)
hg.pettitt_test(df.NETRAD_fill)

########################################################################################################

dfh=pd.read_csv("HEM_daily.csv")
dfh=dfh[dfh['NEE'].notnull()]
#dfh=dfh[dfh['GPP'].notnull()]
#dfh=dfh[dfh['H'].notnull()]
df=dfh
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
mask = (df['TIMESTAMP'] >='2004-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
df=df.groupby(df['water_year']).mean()
df=df.reset_index()
#x1=pd.DataFrame(df['NETRAD'])
#x1=pd.DataFrame(df['Rg'])
#x1=pd.DataFrame(df['PPFD'])
#x1=pd.DataFrame(df['H'])
#x1=pd.DataFrame(df['GPP'])
x1=pd.DataFrame(df['NEE'])
y1=pd.DataFrame(df['ET'])
df['denom'] = x1.diff()
df['num']= y1.diff()
df['slope'] = (df.num/df.denom).fillna(0)
df = df.iloc[1:]
x1=df['water_year']
y1=df['slope'].astype(float)
#mk.original_test(y1)
hg.pettitt_test(y1)
#######################################################################################################



plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(x1, y1,color="blue",label="HEM",linestyle='-', marker='*',markersize=7)
#plt.plot(dfe['water_year'], dfe['ET'],color="cornflowerblue",label="EMS (2008-2023), p=6.9*10$^{-1}$, slope=4.1",linestyle='-', marker='*',markersize=7)

#plt.plot(dfh['water_year'], dfh['NETRAD'],color="darkgreen",label="HEM",linestyle='-', marker='.',markersize=8)
#plt.plot(dfH['water_year'], dfh['ET'],color="lime",label="HEM (2013-2023), p=7.6*10$^{-1}$, slope=3.3",linestyle='-', marker='.',markersize=8)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
#ax.legend(loc='upper left', fontsize = '6',labelspacing=0.0005)
ax.legend(loc='upper left', fontsize = '10',labelcolor='linecolor',labelspacing=0.0005)
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('ET~NEE')  # we already handled the x-label with ax1

z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='blue')



#ax.set_title("Harvard Forest, EMS Tower (US-Ha1), HEM Tower (US-Ha2)",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
#fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
#plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


########################################################################################################
h_sl_d=df
s=df.groupby(df['water_year']).mean()
h_sl_y=s.reset_index()

x1=df['water_year']
y1=df['slope'].astype(float)
mk.original_test(y1)
hg.pettitt_test(df.NETRAD_fill)

#######################################################################################################

#### non growing season






