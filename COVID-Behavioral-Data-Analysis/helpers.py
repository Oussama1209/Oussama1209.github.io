import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
from datetime import datetime
import statsmodels.formula.api as smf
import re


def process_interventions(interventions):
    for col in interventions.columns:
        if col != 'lang':
            interventions.loc[:, col] = pd.to_datetime(interventions.loc[:, col],errors='coerce')
        else :
            interventions[col]=interventions[col].astype('string')
    return interventions.copy()

def process_topics_linked(topics_linked):
    tl=topics_linked.copy()
    idx=np.where(tl.dtypes==bool)[0]
    tl[tl.columns[idx]]=tl[tl.columns[idx]].astype(int)
    return tl

def process_agg_time_series(agg_ts, col_label='sum'):
    res=pd.DataFrame()
    if len(agg_ts.values())==3:
        res=pd.DataFrame(agg_ts)
    else:
        res=pd.DataFrame(agg_ts.values())
        res.index=agg_ts.keys()
        res=res.rename(columns={0 : col_label})
    res.index=pd.to_datetime(res.index)
    res['date']=res.index
    res['year']=res['date'].dt.year
    res['month']=res['date'].dt.month
    res['day']=res['date'].dt.day
    res.index=range(len(res))
    return res

def process_eu_unemployment(dfeu,countries,params):
    dfeuc = dfeu.copy()
    dfeuc['C2'] = dfeu['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[-1])
    dfeuc['age'] = dfeu['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[1])
    dfeuc['unit'] = dfeu['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[2])
    dfeuc['sex'] = dfeu['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[3])
    dfeuc['s_adj'] = dfeu['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[0])
    #remove unwanted years
    if (params==0):
        selected_cols = ['C2','age','unit','sex', 's_adj', 
                        '2020M07 ', '2020M06 ', '2020M05 ', '2020M04 ','2020M03 ','2020M02 ','2020M01 ',
                        '2019M12 ','2019M11 ']
    elif (params==1):
        selected_cols = ['C2','age','unit','sex', 's_adj', 
                        '2020M07 ', '2020M06 ', '2020M05 ', '2020M04 ','2020M03 ','2020M02 ','2020M01 ',
                        '2019M12 ','2019M11 ', '2019M10 ', '2019M09 ', '2019M08 ', '2019M07 ', '2019M06 ', '2019M05 ',
                        '2019M04 ', '2019M03 ', '2019M02 ', '2019M01 ']
    dfeuc = dfeuc[selected_cols]
    #remove NSA rows
    dfeuc.drop(dfeuc.loc[dfeuc['s_adj'] !='SA'].index, inplace=True)
    #we remove the adj column as it is no longer useful
    dfeuc.drop('s_adj', axis=1,inplace=True)
    dfeuc = dfeuc.rename(columns={'C2': "Country_code"})
    dfeuc_merged = dfeuc.merge(countries, on="Country_code")
    #we use the melt function to pivot the dataset
    dfts = dfeuc_merged.melt(id_vars=["Country", "age", "unit", "sex", "Country_code"], 
            var_name="Date", 
            value_name="Value")
    
    #removing non digit data and space
    dfts['Value'] = dfts['Value'].apply(lambda x: re.sub(r"[a-zA-Z: ]", "", x))
    dfts['Value'] = dfts['Value'].apply(lambda x: x.replace(" ",""))
    
    dfts = dfts.loc[~(dfts.Value=="")]
    
    dfts['Value'] = dfts['Value'].apply(lambda x: float(x))
    dfts['Date']=pd.to_datetime(dfts['Date'].apply(lambda x : x.replace('M','/')))
    dfts['Year']=dfts['Date'].dt.year
    dfts['Month']=dfts['Date'].dt.month
    
    return dfts


def process_group_df(df,test_group):
    df=df.reset_index().copy()
    df[test_group]=df[test_group].astype('category')
    df['lockdown']=np.where(df['Date']>=datetime(2020,3,1),1,0)
    return df

## General pipeline for processing the aforementionned data
def unemployment_across_groups(dfts,test_group,size=(12,18)):
    #devide dataset between different metrics
    df_percentage=dfts[dfts['unit']=='PC_ACT'].drop(columns='unit').copy()
    df_numeric=dfts[~(dfts['unit']=='PC_ACT')].drop(columns='unit').copy()
    #compile data needed for graph generation
    generals=[df_percentage,df_numeric]
    ylabels=['Unemployment Percentage','Unemployed People (x1000)']
    #Iterate over each metric dataset
    results=[]
    for i in range(len(generals)):
        #Select metric
        df=generals[i]
        #Aggregate by testing group and Date to get monthly mean 
        df_agg=df.groupby([test_group,'Date']).mean(numeric_only=True)['Value'].copy()
        #'Flatten' the indeces to get only one index column corresponding to the testing group
        df_agg=df_agg.reset_index(level='Date')
        #Plot the different lines for each of the testing group elements
        if(i==0):
            plt.figure(figsize=size)
            plt.axvline(x=datetime(2020,3,1),color='red', linestyle='--', label='Lockdown')
            for idx in df_agg.index.unique(): 
                plt.plot(df_agg.loc[idx]['Date'],df_agg.loc[idx]['Value'],label=idx)
        #Add details to the plot
                plt.legend(loc='upper left')
                plt.xlabel('Date')
                plt.ylabel(ylabels[i])
                plt.title(f'{ylabels[i]} per {test_group} across Europe')
        df_agg=process_group_df(df_agg,test_group)
        results.append(df_agg)
    plt.show()
    return results

def split_group(df, startct, endct, starttr, endtr, datetrct, datetrtr):
    #split control and treatment
    df_ct = df[df['Date'] <= endct]
    df_ct = df_ct[df_ct['Date'] >= startct]
    df_t = df[df['Date'] <= endtr]
    df_t = df_t[df_t['Date'] >= starttr]
    #add group and "reatment date" columns
    df_ct['g'] = 0
    df_t['g'] = 1
    df_ct['t'] = (df['Date'] >= datetrct).astype(int)
    df_t['t'] = (df['Date'] >= datetrtr).astype(int)
    dff = pd.concat([df_t, df_ct])
    #interaction between group and treatment
    dff['gt'] = dff['g']*dff['t']

    #keep only percentage values and work on total age and sex
    
    dff = dff[dff['unit']=='PC_ACT']
    dff = dff[dff['age']=='TOTAL']
    dff = dff[dff['sex']=='T']

    #keep only relevant columns
    dff = dff[['Date','Value','g','t','gt']]
    return dff

def split_group2(df, startct, endct, starttr, endtr, datetrct, datetrtr):
    #split control and treatment
    df_ct = df[df['Date'] <= endct]
    df_ct = df_ct[df_ct['Date'] >= startct]
    df_t = df[df['Date'] <= endtr]
    df_t = df_t[df_t['Date'] >= starttr]
    #add group and "reatment date" columns
    df_ct['g'] = 0
    df_t['g'] = 1
    df_ct['t'] = (df['Date'] >= datetrct).astype(int)
    df_t['t'] = (df['Date'] >= datetrtr).astype(int)
    dff = pd.concat([df_ct, df_t])
    #interaction between group and treatment
    dff['gt'] = dff['g']*dff['t']


    #keep only relevant columns
    dff = dff[['Date','Value','g','t','gt']]
    return dff

def regression_helper(df,group):
    model=smf.ols(data=df,formula=f'Value~{group}:lockdown')
    res=model.fit()
    return res

def plot_pies(df,group,total):
    dfs=[df[(df['lockdown']==0)],df[(df['lockdown']==1)]]
    fig, axs=plt.subplots(1,2)
    titles=['Inactive Population Before Lockdown','Inactive Population After Lockdown']
    for i in range(len(dfs)):
        df_pie=dfs[i].copy()
        df_pie=df_pie[df_pie[group]!='TOTAL']
        df_pie=df_pie.groupby(group).sum().drop(total)
        labels=list((df_pie.index))
        axs[i].pie(df_pie['Value'].astype(float).values,labels=labels)
        axs[i].set_title(titles[i])
    fig.tight_layout()
    plt.show()

def plot_df(df, y_label, title) :

    """
    Plots multiple lineplots in one figure using the data withing the 'df' dataframe

    """

    itLockdown = '2020-03-01'
    itNormalcy = '2020-07-01'
    it1stCase = '2020-02-01'

    dates = df['Timestamp'].values.copy()
    dates = pd.to_datetime(dates)

    topics = df.columns[1:]

    # Format to display only year-month-day
    dates = dates.strftime('%Y-%m-%d')


    plt.figure(figsize=(20, 10))
    for topic in topics :
        sns.lineplot(x=dates, y=df[topic].values, errorbar=None)
        
    plt.xticks(rotation=90)
    plt.xlabel('Dates (From 2019-01-01 to 2020-08-01)', fontsize = 14)

    # Creating 3 vertical lines, indicating the when the interventions (Lockdown, Normalcy and 1st Case)
    plt.axvline(x=itLockdown,color='red', linestyle='--', label='Lockdown')
    plt.axvline(x=itNormalcy,color='green', linestyle='--', label='Normalcy')
    plt.axvline(x=it1stCase,color='black', linestyle='--', label='1st Death')

    plt.ylabel(y_label, fontsize = 14)
    plt.title(title, fontsize = 18)
    plt.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.legend()
    plt.show()    

def rel_change_from_baseline (df) :
    """
    Creates and returns a new dataframe containing the relative change from baseline rather
    than the number of pageviews
    Adds a column 'avg' containing the average over the different columns
    
    """
    topics = df.columns[1:]
    baselines = df.iloc[0][1:].values
    
    dfCopy = df.copy()

    for index in range(len(baselines)) :
        dfCopy[topics[index]] = 100*(dfCopy[topics[index]] - baselines[index])/baselines[index]


    row_avg = dfCopy.iloc[:, 1:].mean(axis=1)
    dfCopy['avg'] = row_avg
    
    return dfCopy

def sum_pageviews(df) :
    """
    Creates and returns a new dataframe containing the sum of all the pageviews
    Adds a column 'avg' containing the average over the different columns
    
    """
    topics = df.columns[1:]
    dfCopy = pd.DataFrame()
    dfCopy['Timestamp'] = df['Timestamp']
    dfCopy['sum'] = df[topics].sum(axis=1)
    
    return dfCopy
    

def filter_outlier (df, thresh) :
    """
    Function that filters out outliers 

    """
    
    dfCopy = df.copy()
    timestamps = dfCopy['Timestamp']

    # Identify columns where the maximum value is under 10,000
    columns_to_keep = dfCopy.drop('Timestamp', axis=1).columns[dfCopy.drop('Timestamp', axis=1).max() < thresh]

    # Filter the DataFrame based on identified columns
    filtered_columns = dfCopy.loc[:, list(columns_to_keep)]

    # Display the result
    dfFinal = pd.concat([timestamps, filtered_columns], axis=1)

    if (dfCopy.columns[-1] == 'avg') :
        dfFinal['avg'] = dfFinal.iloc[:, 1:-1].mean(axis = 1)

    return (dfFinal)

def get_dfs(maslow_level):
    """
    Returns a structure variable 'dfs' containing a pageview dataframe in each field,
    corresponding to a specific country.

    Input : String indicating the Maslow level folder name that's being accessed in order to retrieve the 
            CSV files from which we will generate the datafarames 
    
    """
    dfs = {
        "it": [], 
        "cs": [],
        "ro": [],
        "sv": [],
        "fi": [],
        "da": [],
        }
    lan = ['it', 'cs', 'ro', 'sv', 'fi', 'da']
    for i in lan:
        df = pd.read_csv(f'Wiki-pageviews/{maslow_level}/{maslow_level}_pageviews_{i}.csv')
        dfs[i] = df
    
    return dfs

def plot_df_countries (dfs, yLabel, title) :
    """
    Plots lineplots for each column of the given dataframe

    """

    itLockdown = '2020-03-01'
    itNormalcy = '2020-07-01'
    it1stCase = '2020-02-01'

    # Region code of the different countries
    lan = ['it', 'cs', 'ro', 'sv', 'fi', 'da']

    # Name of the countries
    countries = ['Italy', 'Serbia', 'Romania', 'Slovenia', 'Finland', 'Denmark']
    
    # Get timestamps
    dates = dfs['it']['Timestamp'].values

    # Figure contains 2x3 (12) subplots, one for each european country that we will be analyzing 
    fig, axs= plt.subplots(2, 3, sharey=True, sharex=True, figsize=(14,10))

    # Iterate over every country in the 'euroCountries' array
    for i, country in enumerate(countries) :

        # Retrieve Dataframe Corresponding to the country
        dfCurrent = dfs[lan[i]].copy()

        # Proper index formating to be at the right plot
        ax = axs[i//3, i%3]

        topics = dfCurrent.columns[1:]

        # Format to display only year-month-day
        # dates = dates.strftime('%Y-%m-%d')

        for topic in topics :
            sns.lineplot(x=dates, y=dfCurrent[topic].values, errorbar=None, ax=ax)

        ax.set_xticklabels(labels= dates,rotation=90) 
            
        ax.axvline(x=itLockdown,color='red', linestyle='--', label='Lockdown')
        ax.axvline(x=itNormalcy,color='green', linestyle='--', label='Normalcy')
        ax.axvline(x=it1stCase,color='black', linestyle='--', label='1st Death')
           
        ax.set_title(f'Evolution in {country}')
        ax.grid(True)

    # Adding general titles to the x and y-axis
    fig.text(0.5, 0.001, 'Dates (From 2019-01-01 to 2020-08-01)', ha='center',
            va='center', fontsize=10)
    fig.text(0.05, 0.5, yLabel, ha='center',
            va='center', rotation='vertical', fontsize=10)

    # Addjusting figure formatting
    # plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Adding a general title
    plt.suptitle(title, fontsize=14)
    plt.legend()
    plt.show()

def plot_df_countries_avg (dfs, yLabel, title) :

    itLockdown = '2020-03-01'
    itNormalcy = '2020-07-01'
    it1stCase = '2020-02-01'

    # Region code of the different countries
    lan = ['it', 'cs', 'ro', 'sv', 'fi', 'da']

    # Name of the countries
    countries = ['Italy', 'Serbia', 'Romania', 'Slovenia', 'Finland', 'Denmark']
    
    # Get timestamps
    dates = dfs['it']['Timestamp'].values

    # Figure contains 2x3 (12) subplots, one for each european country that we will be analyzing 
    fig, axs= plt.subplots(2, 3, sharey=True, sharex=True, figsize=(14,10))

    # Iterate over every country in the 'euroCountries' array
    for i, country in enumerate(countries) :

        # Retrieve Dataframe Corresponding to the country
        dfCurrent = dfs[lan[i]].copy()

        # Proper index formating to be at the right plot
        ax = axs[i//3, i%3]

        topics = dfCurrent.columns[1:]

        # Format to display only year-month-day
        # dates = dates.strftime('%Y-%m-%d')

        for topic in topics :
            sns.lineplot(x=dates, y=dfCurrent['avg'].values, errorbar=None, ax=ax)
            
        ax.set_xticklabels(labels= dates,rotation=90)

        ax.axvline(x=itLockdown,color='red', linestyle='--', label='Lockdown')
        ax.axvline(x=itNormalcy,color='green', linestyle='--', label='Normalcy')
        ax.axvline(x=it1stCase,color='black', linestyle='--', label='1st Death')
            
        ax.set_title(f'Evolution for {country}')
        ax.grid(True)

    # Adding general titles to the x and y-axis
    fig.text(0.5, 0.001, 'Dates (From 2019-01-01 to 2020-08-01)', ha='center',
            va='center', fontsize=10)
    fig.text(0.05, 0.5, yLabel, ha='center',
            va='center', rotation='vertical', fontsize=10)

    # Addjusting figure formatting
    # plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Adding a general title
    plt.suptitle(title, fontsize=14)
    plt.legend()
    plt.show()

def one_plot_avg (dfs, title) :
    """
    Plots multiple lineplots in one figure using the 'avg' column of each dataframe within the
    structure dfs (with specific labeling).

    """

    # Region code of the different countries
    lan = ['it', 'cs', 'ro', 'sv', 'fi', 'da']

    # Name of the countries
    countries = ['Italy', 'Serbia', 'Romania', 'Slovenia', 'Finland', 'Denmark']

    itLockdown = '2020-03-01'
    itNormalcy = '2020-07-01'
    it1stCase = '2020-02-01'

    dates = dfs['it']['Timestamp'].values.copy()
    dates = pd.to_datetime(dates)


    # Format to display only year-month-day
    dates = dates.strftime('%Y-%m-%d')


    plt.figure(figsize=(20, 10))

    for i, country in enumerate(countries) :
        sns.lineplot(x=dates, y=dfs[lan[i]]['avg'].values, errorbar=None, label=country)
    
    # Creating 3 vertical lines, indicating the when the interventions (Lockdown, Normalcy and 1st Case)
    plt.axvline(x=itLockdown,color='red', linestyle='--', label='Lockdown')
    plt.axvline(x=itNormalcy,color='green', linestyle='--', label='Normalcy')
    plt.axvline(x=it1stCase,color='black', linestyle='--', label='1st Death')

    plt.grid(color='black', linestyle='dotted', linewidth=0.75)
    
    plt.xticks(rotation=90)
    plt.xlabel('Dates (From 2019-01-01 to 2020-08-01)', fontsize = 14)
    plt.ylabel('Average Relative Percentage Change from Baseline', fontsize = 14)
    plt.title(title, fontsize = 18)
    
    plt.legend()
    plt.show()    

    
def plot_df_countries_px (dfs, yLabel, title):
    """
    Plots lineplots for each column of the given dataframe using Plotly Express
    """

    itLockdown = '2020-03-01'
    itNormalcy = '2020-07-01'
    it1stCase = '2020-02-01'

    # Region code of the different countries
    lan = ['it', 'cs', 'ro', 'sv', 'fi', 'da']

    # Name of the countries
    countries = ['Italy', 'Serbia', 'Romania', 'Slovenia', 'Finland', 'Denmark']

    # Get timestamps
    dates = dfs['it']['Timestamp'].values

    # Create a DataFrame for Plotly Express
    plotly_df = pd.DataFrame()

    for i, country in enumerate(countries):
        dfCurrent = dfs[lan[i]].copy()

        for topic in dfCurrent.columns[1:]:
            plotly_df = pd.concat([plotly_df, pd.DataFrame({'Dates': dates, 'Values': dfCurrent[topic].values, 'Country': country, 'Topic': topic})])

    # Plot using Plotly Express
    fig = px.line(plotly_df, x='Dates', y='Values', color='Country', line_group='Topic',
                  labels={'Values': yLabel, 'Dates': 'Dates (From 2019-01-01 to 2020-08-01)'}, 
                  title=title, facet_col='Country', facet_col_wrap=3)

    # Highlight important dates
    fig.add_shape(dict(type="line", x0=itLockdown, x1=itLockdown, y0=0, y1=1, line=dict(color="red", dash="dash")))
    fig.add_shape(dict(type="line", x0=itNormalcy, x1=itNormalcy, y0=0, y1=1, line=dict(color="green", dash="dash")))
    fig.add_shape(dict(type="line", x0=it1stCase, x1=it1stCase, y0=0, y1=1, line=dict(color="black", dash="dash")))

    # Update layout
    fig.update_layout(xaxis=dict(tickangle=90), legend=dict(title=dict(text='Country')), showlegend=True)

    # Show the plot
    fig.show()

def one_plot_avg_px(dfs, title):
    """
    Plots multiple lineplots in one figure using the 'avg' column of each dataframe within the
    structure dfs (with specific labeling).

    """

    # Region code of the different countries
    lan = ['it', 'cs', 'ro', 'sv', 'fi', 'da']

    # Name of the countries
    countries = ['Italy', 'Serbia', 'Romania', 'Slovenia', 'Finland', 'Denmark']

    itLockdown = '2020-03-01'
    itNormalcy = '2020-07-01'
    it1stCase = '2020-02-01'

    dates = dfs['it']['Timestamp'].values.copy()
    dates = pd.to_datetime(dates)

    # Format to display only year-month-day
    dates = dates.strftime('%Y-%m-%d')

    fig = px.line()
    for i, country in enumerate(countries):
        fig.add_scatter(x=dates, y=dfs[lan[i]]['avg'].values, mode='lines', name=country)

    # Adding vertical lines for interventions
    # Adding vertical lines for interventions with labels
    fig.add_shape(dict(type='line', x0=itLockdown, x1=itLockdown, y0=0, y1=1, xref='x', yref='paper',
                       line=dict(color='red', dash='dash'), name='Lockdown'))
    fig.add_annotation(x=itLockdown, y=1.1, xref='x', yref='paper',
                       text='Lockdown', showarrow=False, xshift=10)

    fig.add_shape(dict(type='line', x0=itNormalcy, x1=itNormalcy, y0=0, y1=1, xref='x', yref='paper',
                       line=dict(color='green', dash='dash'), name='Normalcy'))
    fig.add_annotation(x=itNormalcy, y=1.1, xref='x', yref='paper',
                       text='Normalcy', showarrow=False)

    fig.add_shape(dict(type='line', x0=it1stCase, x1=it1stCase, y0=0, y1=1, xref='x', yref='paper',
                       line=dict(color='black', dash='dash'), name='1st Death'))
    fig.add_annotation(x=it1stCase, y=1.1, xref='x', yref='paper',
                       text='1st Death', showarrow=False, xshift=-20)

    fig.update_layout(
        xaxis=dict(tickangle=90),
        xaxis_title='Dates (From 2019-01-01 to 2020-08-01)',
        yaxis_title='Average Relative Percentage Change from Baseline',
        title=title,
        legend=dict(title='Country'),
        showlegend=True
    )

    fig.show()
