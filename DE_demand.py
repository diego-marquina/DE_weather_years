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
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
init_notebook_mode(connected=True) 
pd.options.plotting.backend = "plotly"

#%%
api_client = MarketDataAPI("diego.marquina@shell.com", "mddpwd_Diego1")
df_epsi = api_client.get_time_series(
    group_name="DE_demand_wx",
    start="2022-01-01",
    end="2023-01-01", 
    granularity='hours'
    )
df_epsi = df_epsi.resample('h').interpolate()
# df_epsi = pd.read_csv('Germany_Wind_PowerGeneration_Reanalysis_ECMWF-ERA5_Total_Hourly_202106052300.csv', sep=';', header=5)
# df_epsi.set_index('From yyyy-mm-dd hh:mm', inplace=True)
# df_epsi.index = pd.to_datetime(df_epsi.index)
df_epsi.columns = df_epsi.columns.str.replace('1Base_Wx-Years.Germany.Demand.','',regex=True)

# df_epsi.index = df_epsi.index.tz_localize('Europe/Berlin', nonexistent='shift_forward', ambiguous='True')


# %%

# Initialize client
eq = EnergyQuantified(api_key='0168d6-b068b6-f41124-7108d3')

# Free-text search (filtering on attributes is also supported)
# curves = eq.metadata.curves(q='de wind production actual')
# curves = eq.metadata.curves(q='de wind production')
curves = eq.metadata.curves(q='DE Consumption MWh/h 15min Climate')
# capacity_curves = eq.metadata.curves(q='DE Wind Power Installed MW Capacity')
# curves = eq.metadata.curves(q=['DE Wind Power Production MWh/h 15min Climate','DE Wind Power Installed MW Capacity'])
# curves = eq.metadata.curves(
#    area='DE',
#    data_type='FORECAST',
#    category=['nuclear', 'production']
# )

# Load time series data
curve = curves[0]
timeseries = eq.timeseries.load(
    curve,
    begin=date(2022,1,1),
    end=date(2023,1,2)
)

# capacity_eq = eq.periods.load(
#     capacity_curves[0],
#     begin=date(2021,1,1),
#     end=date(2022,1,2)
# )

# Convert to Pandas data frame
df_eq = timeseries.to_dataframe()
# df_cap_eq_year = capacity_eq.to_timeseries(time.Frequency.P1Y).to_dataframe()
# df_cap_eq_month = capacity_eq.to_timeseries(time.Frequency.P1M).to_dataframe()
#%%
df_eqh = df_eq.resample('h').mean()
# df_eqh.index = df_eqh.index.tz_convert('CET')
# df_eqh.index = df_eqh.index.tz_localize(None)
df_eqh.index = df_eqh.index.tz_convert(None)
df_eqh = df_eqh.loc['2022']

#%%
df_eqh.columns = df_eqh.columns.get_level_values(2)
#%%
df_eqh.columns = df_eqh.columns.str.replace('y','',regex=True)

# %%
# df_epsi_year = df_epsi.loc['1990':'2019','perc50'].resample('YS').sum().to_frame().rename(columns={'perc50':'epsi'})/1e6
df_epsi_year = df_epsi.sum().to_frame().rename(columns={0:'epsi'})/1e3
df_epsi_year.sort_index(inplace=True)
# df_epsi_year.index = df_epsi_year.index.year.astype(str)

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
    'value':'EPSI consumption / EQ consumption'})
fig.show()
fig.write_html('yearly_consumption.html')    
# %%
# # df_epsi2 = df_epsi.resample('MS').mean().stack().reset_index()
# # df_eqh2 = df_eqh.resample('MS').mean().stack().reset_index()
df_epsi2 = df_epsi.stack().reset_index()
df_epsi2[0] *= 1000
# df_epsi2 = df_epsi.loc['1990':'2019','perc50'].to_frame().rename(columns={'perc50':'wind_epsi'})
df_eqh2 = df_eqh.stack().reset_index()

df_epsi2['Month']=df_epsi2.date.dt.month
df_epsi2['Day']=df_epsi2.date.dt.day
df_epsi2['Hour']=df_epsi2.date.dt.hour
# df_epsi2['Year']=df_epsi2.index.year
# df_epsi2['Month']=df_epsi2.index.month
# df_epsi2['Day']=df_epsi2.index.day
# df_epsi2['Hour']=df_epsi2.index.hour
df_eqh2['Month']=df_eqh2.date.dt.month
df_eqh2['Day']=df_eqh2.date.dt.day
df_eqh2['Hour']=df_eqh2.date.dt.hour

df_epsi2.rename(columns={'level_1':'Year',0:'consumption_epsi'}, inplace=True)
df_eqh2.rename(columns={'level_1':'Year',0:'consumption_eq'}, inplace=True)
# %%
df_epsi2['Weather_years_2022'] = pd.to_datetime(df_epsi2[['Year','Month','Day','Hour']])
df_eqh2['Weather_years_2022'] = pd.to_datetime(df_eqh2[['Year','Month','Day','Hour']])
# %%
df_ws = pd.merge(left=df_epsi2[['Weather_years_2022','consumption_epsi']],
        right=df_eqh2[['Weather_years_2022','consumption_eq']],
        on='Weather_years_2022',
        how='inner').set_index('Weather_years_2022')

df_ws['epsi_over_eq'] = (df_ws['consumption_epsi']/df_ws['consumption_eq'])
# %%
df_ws['epsi_over_eq'].resample('d').mean().plot() #there is an issue here as the daylight savings time shifts one series over summer
#%%
df_ws.resample('d').mean().plot()# %%

# %%
df_wsm = df_ws[['consumption_epsi','consumption_eq']].resample('MS').mean()
df_wsm['epsi_over_eq'] = (df_wsm['consumption_epsi']/df_wsm['consumption_eq'])
# %%
df_wsm.plot()
# %%
# df_epsi_pivot = df_epsi.loc['1990':'2019','perc50'].groupby([
#     df_epsi.loc['1990':'2019','perc50'].index.year,
#     df_epsi.loc['1990':'2019','perc50'].index.month
#     ]).mean().unstack(0)
# df_epsi_pivot.columns = df_epsi_pivot.columns.astype(str)
#%%
# df_heatmap = (df_epsi.resample('MS').mean()/df_eqh.resample('MS').mean()).reset_index(drop=True)
# df_heatmap.index+=1
df_heatmap = ((df_epsi.groupby(df_epsi.index.month).mean()*1000)/df_eqh.groupby(df_eqh.index.month).mean())
df_heatmap = df_heatmap.sort_index(axis=1)
sns.set(rc={'figure.figsize':(11.7,8.27)})
ax = sns.heatmap(df_heatmap)
ax.set(xlabel='weather years', ylabel='month', title='EPSI consumption/ EQ consumption')
# %%
df_ws['epsi_over_eq'] = df_ws['consumption_epsi']/df_ws['consumption_eq']
df_ws['epsi_less_eq'] = df_ws['consumption_epsi']-df_ws['consumption_eq']

# %%
df_ws[['consumption_epsi','consumption_eq']].groupby([df_ws.index.year,df_ws.index.month]).mean().unstack(0).to_csv('monthly_cf.csv')
df_ws[['consumption_epsi','consumption_eq']].groupby(df_ws.index.year).mean().to_csv('yearly_cf.csv')

#%%
hourly_data = df_ws.copy()
hourly_data.index.name = 'Weather years'
hourly_data.columns = [
    'load factor (%) - EPSI',
     'load factor (%) - EQ',
      '2021 consumption (MW) - EPSI',
       '2021 consumption (MW) - EQ'
       ]
hourly_data.to_csv('consumption_hourly_data.csv')
# %%
# df_ws.plot(kind='scatter',x='epsi_less_eq',y='epsi_over_eq', opacity=0.2)
#%%
df_ws['epsi_over_eq'].hist(title='epsi over eq')
#%%
df_ws['epsi_less_eq'].hist(title='epsi less eq')
#%%
# fig, axes = plt.subplots()
# sns.violinplot( data=df_ws[['epsi_less_eq','epsi_over_eq']], ax = axes)
# %%
normal_dist_relative = stats.norm.rvs(
    loc=df_ws['epsi_over_eq'].mean(),
    scale=df_ws['epsi_over_eq'].std(),
    size=len(df_ws['epsi_over_eq'])
    )

normal_dist_absolute = stats.norm.rvs(
    loc=df_ws['epsi_less_eq'].mean(),
    scale=df_ws['epsi_less_eq'].std(),
    size=len(df_ws['epsi_less_eq'])
    )
# %%
act_vs_norm_rel_df = pd.concat(
    [
        df_ws['epsi_over_eq'].reset_index(drop=True),
        pd.DataFrame(data=normal_dist_relative)
    ],
    axis=1)

act_vs_norm_abs_df = pd.concat(
    [
        df_ws['epsi_less_eq'].reset_index(drop=True),
        pd.DataFrame(data=normal_dist_absolute)
    ],
    axis=1)

qq_df_rel=act_vs_norm_rel_df.quantile(np.linspace(1, 0, 100, 0))
qq_df_abs=act_vs_norm_abs_df.quantile(np.linspace(1, 0, 100, 0))

#%%
range_rel=[qq_df_rel.min().min()*.9,qq_df_rel.max().max()*1.1]
figqq_rel = qq_df_rel.plot(kind='scatter',x=0,y='epsi_over_eq',labels={
    '0':'theoretical normal distribution',
    'epsi_over_eq':'actual values',}, title='epsi over eq')
# figqq_rel.add_scatter(x=qq_df_abs['epsi_less_eq'], y=qq_df_abs[0], mode='markers')
figqq_rel.add_scatter(x=range_rel, y=range_rel,mode='lines')
figqq_rel.update_layout(xaxis_range=range_rel, yaxis_range=range_rel)
#%%
range_abs=[qq_df_abs.min().min()*.9,qq_df_abs.max().max()*1.1]
figqq_abs = qq_df_abs.plot(kind='scatter',x=0,y='epsi_less_eq',labels={
    '0':'theoretical normal distribution',
    'epsi_less_eq':'actual values',}, title='epsi less eq')
figqq_abs.add_scatter(x=range_abs, y=range_abs,mode='lines')
figqq_abs.update_layout(xaxis_range=range_abs, yaxis_range=range_abs)
# %%
# df_ws_ext = df_ws.loc[
#     (df_ws['epsi_over_eq'].abs() > df_ws['epsi_over_eq'].std()) & \
#     (df_ws['epsi_less_eq'].abs() > df_ws['epsi_less_eq'].std())]
df_ws_ext = df_ws.resample('MS').mean()
# fig_month = df_ws_ext.loc[df_ws_ext.index.month==1,:].plot(kind='scatter',x='epsi_less_eq',y='epsi_over_eq', opacity=0.2)
fig_month = make_subplots(rows=1, cols=1)
fig_month.update_layout(
    title="Differences in consumption",
    xaxis_title="EPSI less EQ",
    yaxis_title="EPSI over EQ",
    legend_title="Month",
)
# for mon in df_ws_ext.index.month.unique()[df_ws_ext.index.month.unique()>1]:
for mon in df_ws_ext.index.month.unique():
    if mon<3:
        series_color = 'red'
    else:
        series_color='blue'
    fig_month.add_scatter(
        x=df_ws_ext.loc[df_ws_ext.index.month==mon,'epsi_less_eq'],
        y=df_ws_ext.loc[df_ws_ext.index.month==mon,'epsi_over_eq'],
        mode='markers',
        marker={'color':series_color},
        # opacity=0.5,
        name=str(mon))
fig_month.show()
# %%
ax2 = sns.heatmap(df_ws.loc[:,'epsi_over_eq'].groupby([df_ws.index.year, df_ws.index.weekday]).mean().unstack(0))
ax2.set(xlabel='weather years', ylabel='weekday (Mon=0, Sun=6)', title='EPSI consumption/ EQ consumption')
# %%
