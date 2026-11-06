# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:20:20 2024

@author: ammar
"""


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
import pyhomogeneity as hg
from scipy import stats
#####################################################################################################


os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")
df=pd.read_csv("EMS_ML_results.csv")
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])

mask= (df['TIMESTAMP'] >='1992-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))
xg_hr=df

plt.plot(df['LE_fill'],color='red')
plt.plot(df['Rg_fill'],color='red')

# data should have 273936

######################################################################################################

df=xg_hr
##mean
s=df.groupby([df['TIMESTAMP'].dt.date]).mean()
s=s.reset_index()
##sum
s_et=df.groupby([df['TIMESTAMP'].dt.date]).sum()
s_et=s_et.reset_index()

plt.plot(s['Rg_fill'],color='red')

df=pd.concat((s["TIMESTAMP"],s['DoY'],s["Year"], s['Hour'],s['WS_fill'],s['PA_fill'],s['Tair_fill'],s['VPD_fill'],s_et['Rg_fill'],s_et['PAR_fill'],
s['NETRAD_fill'],s["sos"],s["eos"]),axis=1)

#df['Year'] = pd.to_datetime(df["TIMESTAMP"]).dt.year
EMS_daily=df
df.isnull().values.any()

#######################################################################################################

df=EMS_daily

#######################################################################################################
df=EMS_daily
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
#df= df.groupby(['water_year']).sum()
#df=df.reset_index()
mask = (df['TIMESTAMP'] >='1992-10-02 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))
EMS_daily=df


df=EMS_daily

df.isnull().values.any()

mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))

EMS_daily_f=df
df=EMS_daily_f

##################################################################################################
# calculate annual 

df=EMS_daily_f

df= df.groupby(['water_year']).mean()
df=df.reset_index()


x1=df['water_year']
y1=df['VPD_fill'].astype(float)

# pre_y was an unused exploratory placeholder.

plt.plot(df['water_year'],df['VPD_fill'])

mk.original_test(y1)

stats.theilslopes(y1, x1, 0.95)
hg.pettitt_test(y1, alpha=0.05)

df=EMS_daily_ET_f


def Grow_sea(df):
    
    if (df['DoY'] >=df['sos']) and (df['DoY'] <=df['eos']):
        return 'GS'
    else:
        return 'NGS'
    
    
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)


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



df=gw
#df=ngw
df= df.groupby(['water_year']).mean()
df=df.reset_index()



x1=df['water_year']
y1=df['VPD_f'].astype(float)

mk.original_test(y1)
stats.theilslopes(y1, x1, 0.95)
hg.pettitt_test(y1, alpha=0.05)



df['lambda']=(3149000-2370*(df['Tair']+273.16))*1e-6

