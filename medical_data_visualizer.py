import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')
#print(df.head())

# Add 'overweight' column
#IMC
#Calculate IMC by dividing their weight in kilograms by the square of their height in meters
imc = (df['weight'] / (pow((df['height'] / 100), 2)))
#If that value is > 25 then the person is overweight. Use the value 0 for NOT overweight and the value 1 for overweight
#astype if true returns 1 otherwise 0
overweight = (imc > 25).astype(int)
#print(overweight)
df['overweight'] = overweight

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1

df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
#print(df['cholesterol'])
df['gluc'] = (df['gluc'] > 1).astype(int)


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df,
                     id_vars=['cardio'],
                     value_vars=sorted([
                         'cholesterol', 'gluc', 'smoke', 'alco', 'active',
                         'overweight'
                     ]))
    #print(df_cat)

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat_group = df_cat.groupby(['cardio', 'variable'
                                   ]).agg({'value': ['value_counts']})

    df_cat_group.columns = ['total']
    df_cat_group = df_cat_group.reset_index(
        level=['cardio', 'variable', 'value'])

    # Draw the catplot with 'sns.catplot()'
    graph = sns.catplot(data=df_cat_group,
                        x='variable',
                        y='total',
                        col='cardio',
                        kind='bar',
                        hue='value')

    #graph.set_ylabels('total')
    graph.set_axis_labels("variable", "total")

    # Get the figure for the output
    fig = graph.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.copy()

    df_heat.drop(df_heat.loc[df_heat["ap_lo"] > df_heat["ap_hi"]].index,
                 inplace=True)

    for col_name in ["height", "weight"]:
        a = df_heat.loc[df[col_name] < df[col_name].quantile(0.025)]
        b = df_heat.loc[df[col_name] > df[col_name].quantile(0.975)]
        df_heat.drop(a.index.union(b.index), inplace=True)

    # Calculate the correlation matrix
    corr = df_heat.corr().round(1)
    #print(corr)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    #or
    #mask = np.zeros_like(corr)
    #mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(11, 9))

    # Draw the heatmap with 'sns.heatmap()'
    ax = sns.heatmap(corr,
                     vmin=0.08,
                     vmax=0.24,
                     center=0,
                     square=True,
                     annot=True,
                     linewidths=0.5,
                     fmt=".1f",
                     mask=mask,
                     cbar_kws={
                         'shrink': .5
                     }).figure

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
