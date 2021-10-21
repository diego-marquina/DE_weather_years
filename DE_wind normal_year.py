#%%
import pandas as pd
from energyquantified.time.frequency import Frequency
import numpy as np
from datetime import date, timedelta
from razorshell.api_market_data import MarketDataAPI
from energyquantified import EnergyQuantified, time
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
init_notebook_mode(connected=True) 
pd.options.plotting.backend = "plotly"

#%%
# Initialize client
eq = EnergyQuantified(api_key='0168d6-b068b6-f41124-7108d3')

# Free-text search (filtering on attributes is also supported)
curves = eq.metadata.curves(q='DE Wind Power Production MWh/h 15min Normal')
capacity_curves = eq.metadata.curves(q='DE Wind Power Installed MW Capacity')
# curves = eq.metadata.curves(q=['DE Wind Power Production MWh/h 15min Climate','DE Wind Power Installed MW Capacity'])
# curves = eq.metadata.curves(
#    area='DE',
#    data_type='FORECAST',
#    category=['nuclear', 'production']
# )

#%%
# Load time series data
# curve = curves[0]
gen = {}
i = 0
keys = ['all','offshore','onshore']
for curve in curves:
    key = keys[i]
    i+=1
    gen[key] = eq.timeseries.load(
        curve,
        begin=date(2019,1,1),
        end=date(2022,1,2)
    )
cap = {}
i = 0
for curve in capacity_curves:
    key = keys[i]
    i+=1
    cap[key] = eq.periods.load(
        curve,
        begin=date(2019,1,1),
        end=date(2022,1,2)
    )
#%%
# Convert to Pandas data frame
for k, v in gen.items():
    gen[k] = v.to_dataframe()

for k, v in cap.items():
    cap[k] = v.to_timeseries(time.Frequency.P1M).to_dataframe()

#%%
df_gen = pd.concat(gen, axis=1)    
df_cap = pd.concat(cap, axis=1)
#%%
df_gen = df_gen.resample('h').mean()
df_cap = df_cap.resample('h').interpolate()
df_gen.index = df_gen.index.tz_convert(None)
df_cap.index = df_cap.index.tz_convert(None)
df_gen.columns = df_gen.columns.get_level_values(0)
df_cap.columns = df_cap.columns.get_level_values(0)
df_cf = df_gen/df_cap

#%%
df_out = df_cf.loc['2020',:].copy()
df_out = df_out.groupby([df_out.index.month,df_out.index.day, df_out.index.hour]).mean()
df_out.index.set_names(['month','day','hour'], inplace=True)
#%%
df_eqh.index = df_eqh.index.tz_convert(None)
df_eqh = df_eqh.loc['2021']

#%%
df_eqh.columns = df_eqh.columns.get_level_values(2)
#%%
df_eqh.columns = df_eqh.columns.str.replace('y','',regex=True)

# %%
df_epsi_year = df_epsi.loc['1990':'2019','perc50'].resample('YS').sum().to_frame().rename(columns={'perc50':'epsi'})/1e6
df_epsi_year.index = df_epsi_year.index.year.astype(str)

#%%
# df_eqh = df_eqh.loc[:,df_epsi.columns]
df_eqh = df_eqh.loc[:,df_epsi_year.index]
df_eqh.columns.name = 'Year'

# df_yr = pd.concat([df_eqh.sum()/1e6,df_epsi.sum()/1e6],axis=1).rename(columns={0:'eq',1:'epsi'})
df_yr = pd.concat([df_eqh.sum()/1e6,df_epsi_year],axis=1).rename(columns={0:'eq',1:'epsi'})

# %%
df_yr.sort_index().plot()
#%%
df_yr['epsi_over_eq'] = (df_yr['epsi']/df_yr['eq'])
df_yr = df_yr.sort_index()
df_yr.index.name = 'weather years'
fig = df_yr['epsi_over_eq'].plot(labels={
    'weather years':'weather years',
    'value':'EPSI wind output / EQ wind output'})
fig.show()
fig.write_html('yearly.html')    
# %%
# # df_epsi2 = df_epsi.resample('MS').mean().stack().reset_index()
# # df_eqh2 = df_eqh.resample('MS').mean().stack().reset_index()
# df_epsi2 = df_epsi.stack().reset_index()
df_epsi2 = df_epsi.loc['1990':'2019','perc50'].to_frame().rename(columns={'perc50':'wind_epsi'})
df_eqh2 = df_eqh.stack().reset_index()

# df_epsi2['Month']=df_epsi2.date.dt.month
# df_epsi2['Day']=df_epsi2.date.dt.day
# df_epsi2['Hour']=df_epsi2.date.dt.hour
df_epsi2['Year']=df_epsi2.index.year
df_epsi2['Month']=df_epsi2.index.month
df_epsi2['Day']=df_epsi2.index.day
df_epsi2['Hour']=df_epsi2.index.hour
df_eqh2['Month']=df_eqh2.date.dt.month
df_eqh2['Day']=df_eqh2.date.dt.day
df_eqh2['Hour']=df_eqh2.date.dt.hour

df_epsi2.rename(columns={'level_1':'Year',0:'wind_epsi'}, inplace=True)
df_eqh2.rename(columns={'level_1':'Year',0:'wind_eq'}, inplace=True)
# %%
df_epsi2['Weather_years_2022'] = pd.to_datetime(df_epsi2[['Year','Month','Day','Hour']])
df_eqh2['Weather_years_2022'] = pd.to_datetime(df_eqh2[['Year','Month','Day','Hour']])
# %%
df_ws = pd.merge(left=df_epsi2[['Weather_years_2022','wind_epsi']],
        right=df_eqh2[['Weather_years_2022','wind_eq']],
        on='Weather_years_2022',
        how='inner').set_index('Weather_years_2022')

df_ws['epsi_over_eq'] = (df_ws['wind_epsi']/df_ws['wind_eq'])
# %%
df_ws['epsi_over_eq'].resample('d').mean().plot() #there is an issue here as the daylight savings time shifts one series over summer
#%%
df_ws.resample('d').mean().plot()# %%

# %%
df_wsm = df_ws[['wind_epsi','wind_eq']].resample('MS').mean()
df_wsm['epsi_over_eq'] = (df_wsm['wind_epsi']/df_wsm['wind_eq'])
# %%
df_wsm.plot()
# %%
df_epsi_pivot = df_epsi.loc['1990':'2019','perc50'].groupby([
    df_epsi.loc['1990':'2019','perc50'].index.year,
    df_epsi.loc['1990':'2019','perc50'].index.month
    ]).mean().unstack(0)
df_epsi_pivot.columns = df_epsi_pivot.columns.astype(str)
#%%
# df_heatmap = (df_epsi.resample('MS').mean()/df_eqh.resample('MS').mean()).reset_index(drop=True)
# df_heatmap.index+=1
df_heatmap = (df_epsi_pivot/df_eqh.groupby(df_eqh.index.month).mean())
df_heatmap = df_heatmap.sort_index(axis=1)
sns.set(rc={'figure.figsize':(11.7,8.27)})
ax = sns.heatmap(df_heatmap)
ax.set(xlabel='weather years', ylabel='month', title='EPSI wind output/ EQ wind output')
# %%
df_epsi_cf_yr = df_epsi.loc['1990':'2019','perc50'].resample('YS').mean().to_frame().rename(columns={'perc50':'epsi'})/df_epsi_capacity[2021]
df_epsi_cf_yr.index = df_epsi_cf_yr.index.year.astype(str)
df_eq_cf_yr = (df_eqh.mean(axis=0)/df_cap_eq_year.iloc[0,0]).to_frame().rename(columns={0:'eq'})
df_cf_yr = pd.concat([df_epsi_cf_yr,df_eq_cf_yr], axis=1)

#%%
(df_cf_yr['epsi']/df_cf_yr['eq']).plot()
# %%
df_epsi_cf_month = df_epsi_pivot/df_epsi_capacity[2021]
df_eq_cf_month = df_eqh.groupby(df_eqh.index.month).mean()/df_cap_eq_year.iloc[0,0]
df_heatmap_cf = df_epsi_cf_month/df_eq_cf_month
# %%
ax1 =sns.heatmap(df_heatmap_cf)
ax1.set(xlabel='weather years', ylabel='month', title='EPSI cf/ EQ cf')
# %%
df_cf_hourly = df_ws[['wind_epsi','wind_eq']].div([df_epsi_capacity[2021],df_cap_eq_year.iloc[0,0]], axis=1).rename(columns={'wind_epsi':'wind_cf_epsi','wind_eq':'wind_cf_eq'})
# df_cf_hourly['wind_eq_shifted_1'] = df_cf_hourly['wind_eq'].shift(-1)
# df_cf_hourly['wind_eq_shifted_2'] = df_cf_hourly['wind_eq'].shift(-2)
df_cf_hourly['epsi_over_eq'] = df_cf_hourly['wind_cf_epsi']/df_cf_hourly['wind_cf_eq']
df_cf_hourly['epsi_less_eq'] = df_cf_hourly['wind_cf_epsi']-df_cf_hourly['wind_cf_eq']

# %%
df_cf_hourly[['wind_cf_epsi','wind_cf_eq']].groupby([df_cf_hourly.index.year,df_cf_hourly.index.month]).mean().unstack(0).to_csv('monthly_cf.csv')
df_cf_hourly[['wind_cf_epsi','wind_cf_eq']].groupby(df_cf_hourly.index.year).mean().to_csv('yearly_cf.csv')

#%%
hourly_data = pd.concat([df_cf_hourly.iloc[:,:-2],df_ws.iloc[:,:-1]],axis=1)
hourly_data.index.name = 'Weather years'
hourly_data.columns = [
    'load factor (%) - EPSI',
     'load factor (%) - EQ',
      '2021 Wind output (MW) - EPSI',
       '2021 Wind output (MW) - EQ'
       ]
hourly_data.to_csv('hourly_data.csv')
# %%
df_cf_hourly.plot(kind='scatter',x='epsi_less_eq',y='epsi_over_eq', opacity=0.2)
#%%
df_cf_hourly['epsi_over_eq'].hist(title='epsi over eq')
#%%
df_cf_hourly['epsi_less_eq'].hist(title='epsi less eq')
#%%
# sm.qqplot(df_cf_hourly['epsi_over_eq'], line='45')
# sm.qqplot(df_cf_hourly['epsi_less_eq'], line='45')
# df_cf_hourly[['epsi_less_eq','epsi_over_eq']].boxplot()
fig, axes = plt.subplots()
sns.violinplot( data=df_cf_hourly[['epsi_less_eq','epsi_over_eq']], ax = axes)
# %%
normal_dist_relative = stats.norm.rvs(
    loc=df_cf_hourly['epsi_over_eq'].mean(),
    scale=df_cf_hourly['epsi_over_eq'].std(),
    size=len(df_cf_hourly['epsi_over_eq'])
    )

normal_dist_absolute = stats.norm.rvs(
    loc=df_cf_hourly['epsi_less_eq'].mean(),
    scale=df_cf_hourly['epsi_less_eq'].std(),
    size=len(df_cf_hourly['epsi_less_eq'])
    )
# %%
act_vs_norm_rel_df = pd.concat(
    [
        df_cf_hourly['epsi_over_eq'].reset_index(drop=True),
        pd.DataFrame(data=normal_dist_relative)
    ],
    axis=1)

act_vs_norm_abs_df = pd.concat(
    [
        df_cf_hourly['epsi_less_eq'].reset_index(drop=True),
        pd.DataFrame(data=normal_dist_absolute)
    ],
    axis=1)

qq_df_rel=act_vs_norm_rel_df.quantile(np.linspace(1, 0, 100, 0))
qq_df_abs=act_vs_norm_abs_df.quantile(np.linspace(1, 0, 100, 0))

#%%
range_rel=[0.5,4]
figqq_rel = qq_df_rel.plot(kind='scatter',x=0,y='epsi_over_eq',labels={
    '0':'theoretical normal distribution',
    'epsi_over_eq':'actual values',}, title='epsi over eq')
# figqq_rel.add_scatter(x=qq_df_abs['epsi_less_eq'], y=qq_df_abs[0], mode='markers')
figqq_rel.add_scatter(x=range_rel, y=range_rel,mode='lines')
figqq_rel.update_layout(xaxis_range=range_rel, yaxis_range=range_rel)
#%%
range_abs=[-0.5,0.5]
figqq_abs = qq_df_abs.plot(kind='scatter',x=0,y='epsi_less_eq',labels={
    '0':'theoretical normal distribution',
    'epsi_less_eq':'actual values',}, title='epsi less eq')
figqq_abs.add_scatter(x=range_abs, y=range_abs,mode='lines')
figqq_abs.update_layout(xaxis_range=range_abs, yaxis_range=range_abs)
# %%
