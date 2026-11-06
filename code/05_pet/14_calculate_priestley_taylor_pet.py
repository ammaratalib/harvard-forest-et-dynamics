# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 13:53:29 2025

@author: ammar
"""

##########################################################################################################
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

###############################################################################################################
## slope of saturation vapor curve
##############################################################################################################



def priestley_taylor_et_radiation_wm2(net_radiation_wm2, air_temperature, alpha=1.26):
    """
    Calculate evapotranspiration (ET) using the Priestley-Taylor method with net radiation in W/m^2.

    Parameters:
    - net_radiation_wm2 (float): Net radiation at the surface (W/m^2).
    - air_temperature (float): Air temperature in degrees Celsius.
    - alpha (float): Priestley-Taylor coefficient (default is 1.26).

    Returns:
    - float: Estimated evapotranspiration (mm/day).
    """
    # Constants
    gamma = 0.066  # Psychrometric constant (kPa/°C)
    latent_heat_flux_density = 2.45  # Latent heat of vaporization (MJ/kg)

    # Convert net radiation from W/m^2 to MJ/m^2/day
    # 1 W/m^2 = 0.0864 MJ/m^2/day
    net_radiation_mj_m2_day = net_radiation_wm2 * 0.0864

    # Calculate slope of the saturation vapor pressure curve (kPa/°C)
    delta = (4098 * (0.6108 * np.exp((17.27 * air_temperature) / (air_temperature + 237.3)))) / ((air_temperature + 237.3) ** 2)

    # Priestley-Taylor equation
    et = (alpha * delta * net_radiation_mj_m2_day) / (delta + gamma) / latent_heat_flux_density

    # Convert ET from MJ/m^2/day to mm/day (1 MJ/m^2/day = 1 mm/day for water)
    return et





os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")
df=pd.read_csv("daily_ems_pet_ET.csv") 
daily_ems_pet_ET=df

df=daily_ems_pet_ET
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df=df
#back_daily_ems=df
 # originally rn in [MJ/m2/d] because we used it for panmontheith equation
df["Rn_wm-2"]=(df["Rn"]/3600)*1000000 # convert it back to Wm-2        
df["PET_pt"]=priestley_taylor_et_radiation_wm2(df['Rn_wm-2'], df.TA, alpha=1.26)

back_daily_ems=df


df=back_daily_ems
df=df.groupby(['water_year']).sum()
#df=df.groupby(['water_year']).mean()
df=df.reset_index()
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_ems_sum=df
df=annual_ems_sum




df_e_y=df

x1=df_e_y['water_year']
#y1=df_e_y['PET_ems'].astype(float)
y1=df_e_y['PET_pt'].astype(float)
#y1=df_e_y['ET'].astype(float)
#pearsonr(x1, y1)
mk.original_test(y1)


#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4.5)
fig, ax = plt.subplots()
#plt.scatter(df_e_y['water_year'], df_e_y['PET_ems'],color="lightcoral"  ,marker='*' , s=40,label="PET, p= 0.002, slope= 4.8")
plt.plot(df_e_y['water_year'], df_e_y['PET_ems'],color="lightcoral", marker='*', markersize=8, label="PET_PM,   p= 0.002,   slope= 4.8")
plt.plot(df_e_y['water_year'], df_e_y['PET_pt'],color="darkorange", marker='3', markersize=9,label="PET_PT,    p= 0.20,     slope= 2.2")
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='lower right', fontsize = '8',labelspacing=0.0005,labelcolor='linecolor')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Total annual PET (mm)')  # we already handled the x-label with ax1

z1 = np.polyfit(df_e_y['water_year'], df_e_y['PET_ems'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='lightcoral',linestyle='-')

#ax.set_ylim(300,1100)
z2 = np.polyfit(df_e_y['water_year'], df_e_y['PET_pt'], 1)
p2 = np.poly1d(z2)
plt.plot(df_e_y['water_year'], p2(df_e_y['water_year']),color='darkorange',linestyle='--')

ax.set_title("US-Ha1, EMS Tower, Annual Potential Evapotranspiration",fontsize=8)
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


#########################################################################################

df=back_daily_ems
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))   
GS=df

df=GS

#df=NGS
df=df.groupby(['water_year']).sum()
#df=df.groupby(['water_year']).mean()
df=df.reset_index()
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_ems_sum=df




df_e_y=df

x1=df_e_y['water_year']
#y1=df_e_y['PET_ems'].astype(float)
y1=df_e_y['PET_pt'].astype(float)
#y1=df_e_y['ET'].astype(float)
#pearsonr(x1, y1)
mk.original_test(y1)


#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4.5)
fig, ax = plt.subplots()
#plt.scatter(df_e_y['water_year'], df_e_y['PET_ems'],color="lightcoral"  ,marker='*' , s=40,label="PET, p= 0.002, slope= 4.8")
plt.plot(df_e_y['water_year'], df_e_y['PET_ems'],color="lightcoral", marker='8', markersize=6, label="PET_MT,  p= 0.048, slope= 2.63")
plt.plot(df_e_y['water_year'], df_e_y['PET_pt'],color="darkorange", marker='p', markersize=6,label="PET_PT,    p= 0.17,     slope= 1.6")
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '8',labelspacing=0.0005,labelcolor='linecolor')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Total annual PET (mm)')  # we already handled the x-label with ax1

z1 = np.polyfit(df_e_y['water_year'], df_e_y['PET_ems'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='lightcoral',linestyle='-')

#ax.set_ylim(300,1100)
z2 = np.polyfit(df_e_y['water_year'], df_e_y['PET_pt'], 1)
p2 = np.poly1d(z2)
plt.plot(df_e_y['water_year'], p2(df_e_y['water_year']),color='darkorange',linestyle='--')

#ax.set_title("US-Ha1, EMS Tower, Annual Potential Evapotranspiration",fontsize=8)
ax.set_title("US-Ha1, EMS Tower, Growing Season Annual Potential Evapotranspiration",fontsize=8)
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

############################################################################################################



df=back_daily_ems
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
mask = (df['Grow_sea'] == 'NGS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))   
NGS=df

df=NGS

#df=NGS
df=df.groupby(['water_year']).sum()
#df=df.groupby(['water_year']).mean()
df=df.reset_index()
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_ems_sum=df




df_e_y=df

x1=df_e_y['water_year']
#y1=df_e_y['PET_ems'].astype(float)
y1=df_e_y['PET_pt'].astype(float)
#y1=df_e_y['ET'].astype(float)
#pearsonr(x1, y1)
mk.original_test(y1)


#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4.5)
fig, ax = plt.subplots()
#plt.scatter(df_e_y['water_year'], df_e_y['PET_ems'],color="lightcoral"  ,marker='*' , s=40,label="PET, p= 0.002, slope= 4.8")
plt.plot(df_e_y['water_year'], df_e_y['PET_ems'],color="lightcoral", marker='x', markersize=6, label="PET_MT,  p= 0.004, slope= 2.88")
plt.plot(df_e_y['water_year'], df_e_y['PET_pt'],color="darkorange", marker='d', markersize=6,label="PET_PT,    p= 0.78,     slope= 0.22")
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='lower right', fontsize = '8',labelspacing=0.0005,labelcolor='linecolor')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Total annual PET (mm)')  # we already handled the x-label with ax1

z1 = np.polyfit(df_e_y['water_year'], df_e_y['PET_ems'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['water_year'], p1(df_e_y['water_year']),color='lightcoral',linestyle='-')

#ax.set_ylim(300,1100)
z2 = np.polyfit(df_e_y['water_year'], df_e_y['PET_pt'], 1)
p2 = np.poly1d(z2)
plt.plot(df_e_y['water_year'], p2(df_e_y['water_year']),color='darkorange',linestyle='--')

#ax.set_title("US-Ha1, EMS Tower, Annual Potential Evapotranspiration",fontsize=8)
ax.set_title("US-Ha1, EMS Tower, Non-Growing Season Annual Potential Evapotranspiration",fontsize=8)
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


