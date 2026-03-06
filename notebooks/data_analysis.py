# Data Breach Analysis
# Exploring patterns in data breaches (2000-2024) - what types of attacks
# happen most, which industries get hit hardest, and how human error plays into it.

# %% Setup

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CHARTS_DIR = os.path.join(PROJECT_DIR, 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(PROJECT_DIR, 'dataset', 'raw', 'DataBreach_dataset.csv'))
print(f'Loaded {df.shape[0]} rows, {df.shape[1]} columns')
print(df.head(10))


# %% Quick look at the data

df.info()
print(df.describe())
print(df.describe(include='object'))

print('Missing values per column:')
print(df.isnull().sum())
print(f'\nDuplicate rows: {df.duplicated().sum()}')
# nice - no nulls, no dupes

for col in df.columns:
    print(f'{col}: {df[col].nunique()} unique')


# %% Cleaning

# some of the string columns have trailing whitespace
str_cols = df.select_dtypes(include='object').columns
for col in str_cols:
    df[col] = df[col].str.strip()

print('Human Error Factors:', df['Human Error Factor'].unique())
print('Impact Levels:', df['Impact Level'].unique())

impact_order = ['Low', 'Medium', 'High', 'Critical']
df['Impact Level'] = pd.Categorical(df['Impact Level'], categories=impact_order, ordered=True)

# bin financial loss and records into categories for easier analysis later
df['Loss Category'] = pd.cut(df['Financial Loss'],
                              bins=[0, 500000, 1000000, 1500000, 2000000, float('inf')],
                              labels=['<500K', '500K-1M', '1M-1.5M', '1.5M-2M', '>2M'])

df['Records Category'] = pd.cut(df['Records Compromised'],
                                 bins=[0, 200000, 400000, 600000, 800000, float('inf')],
                                 labels=['<200K', '200K-400K', '400K-600K', '600K-800K', '>800K'])

print(f'\nLoss Category distribution:\n{df["Loss Category"].value_counts().sort_index()}')
print(f'\nRecords Category distribution:\n{df["Records Category"].value_counts().sort_index()}')

df.to_csv(os.path.join(PROJECT_DIR, 'dataset', 'cleaned', 'data_breaches_cleaned.csv'), index=False)
print('Saved cleaned data')


# %% Breach type analysis

breach_counts = df['Type of Breach'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = sns.color_palette('Set2', len(breach_counts))
breach_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
axes[0].set_title('Number of Incidents by Breach Type', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Breach Type')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

axes[1].pie(breach_counts.values, labels=breach_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=140, pctdistance=0.85)
axes[1].set_title('Breach Type Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '01_breach_type_frequency.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 01_breach_type_frequency.png')

# quick check on how different breach types compare in terms of damage
breach_impact = df.groupby('Type of Breach').agg(
    avg_loss=('Financial Loss', 'mean'),
    avg_records=('Records Compromised', 'mean'),
    total_loss=('Financial Loss', 'sum'),
    total_records=('Records Compromised', 'sum'),
    count=('Year', 'count')
).sort_values('avg_loss', ascending=False)

breach_impact['avg_loss'] = breach_impact['avg_loss'].round(2)
breach_impact['avg_records'] = breach_impact['avg_records'].round(0).astype(int)
breach_impact['total_loss'] = breach_impact['total_loss'].round(2)

print('Breach Type Impact Summary:')
print(breach_impact)
# phishing has the highest avg loss - interesting

fig = px.box(df, x='Type of Breach', y='Financial Loss', color='Type of Breach',
             title='Financial Loss Distribution by Breach Type',
             labels={'Financial Loss': 'Financial Loss ($)', 'Type of Breach': 'Breach Type'})
fig.update_layout(showlegend=False, xaxis_tickangle=-45)
fig.write_image(os.path.join(CHARTS_DIR, '02_financial_loss_by_breach_type_box.png'), width=1200, height=600, scale=2)
print('Saved: 02_financial_loss_by_breach_type_box.png')


# %% Industry analysis

industry_counts = df['Industry'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors_ind = sns.color_palette('coolwarm', len(industry_counts))
industry_counts.plot(kind='barh', ax=axes[0], color=colors_ind, edgecolor='black')
axes[0].set_title('Total Breaches by Industry', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Breaches')

industry_loss = df.groupby('Industry')['Financial Loss'].mean().sort_values(ascending=True)
industry_loss.plot(kind='barh', ax=axes[1], color=sns.color_palette('YlOrRd', len(industry_loss)), edgecolor='black')
axes[1].set_title('Average Financial Loss by Industry', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Average Financial Loss ($)')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '03_industry_breaches_and_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 03_industry_breaches_and_loss.png')

# heatmap to see if certain industries attract specific attack types
cross_tab = pd.crosstab(df['Industry'], df['Type of Breach'])

plt.figure(figsize=(12, 6))
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5)
plt.title('Breach Type Distribution Across Industries', fontsize=14, fontweight='bold')
plt.xlabel('Type of Breach')
plt.ylabel('Industry')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '04_breach_type_industry_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 04_breach_type_industry_heatmap.png')

# looks like breaches are spread pretty evenly across industries,
# no single industry is immune to any particular type

impact_cross = pd.crosstab(df['Industry'], df['Impact Level'])

fig = px.bar(impact_cross, barmode='group',
             title='Impact Level Distribution Across Industries',
             labels={'value': 'Count', 'Industry': 'Industry'})
fig.update_layout(xaxis_tickangle=-45, legend_title='Impact Level')
fig.write_image(os.path.join(CHARTS_DIR, '05_impact_level_by_industry.png'), width=1200, height=600, scale=2)
print('Saved: 05_impact_level_by_industry.png')

industry_summary = df.groupby('Industry').agg(
    total_records=('Records Compromised', 'sum'),
    total_loss=('Financial Loss', 'sum'),
    avg_records=('Records Compromised', 'mean'),
    avg_loss=('Financial Loss', 'mean'),
    breach_count=('Year', 'count')
).round(2).sort_values('total_loss', ascending=False)

print('\nIndustry Summary:')
print(industry_summary)


# %% Temporal trends

yearly_counts = df.groupby('Year').size().reset_index(name='Count')

fig = px.line(yearly_counts, x='Year', y='Count', markers=True,
              title='Number of Data Breaches Over Time (2000-2024)',
              labels={'Count': 'Number of Breaches'})
fig.update_traces(line=dict(width=3))
fig.update_layout(xaxis=dict(dtick=1))
fig.write_image(os.path.join(CHARTS_DIR, '06_breaches_over_time.png'), width=1400, height=600, scale=2)
print('Saved: 06_breaches_over_time.png')

yearly_loss = df.groupby('Year').agg(
    total_loss=('Financial Loss', 'sum'),
    avg_loss=('Financial Loss', 'mean'),
    total_records=('Records Compromised', 'sum')
).reset_index()

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=('Total Financial Loss by Year', 'Total Records Compromised by Year'))

fig.add_trace(go.Bar(x=yearly_loss['Year'], y=yearly_loss['total_loss'],
                     name='Total Loss ($)', marker_color='indianred'), row=1, col=1)
fig.add_trace(go.Bar(x=yearly_loss['Year'], y=yearly_loss['total_records'],
                     name='Total Records', marker_color='steelblue'), row=2, col=1)

fig.update_layout(height=700, title_text='Financial Loss & Records Compromised Over Time', showlegend=True)
fig.write_image(os.path.join(CHARTS_DIR, '07_yearly_loss_and_records.png'), width=1400, height=700, scale=2)
print('Saved: 07_yearly_loss_and_records.png')

# breakdown by breach type over the years
breach_yearly = df.groupby(['Year', 'Type of Breach']).size().reset_index(name='Count')

fig = px.area(breach_yearly, x='Year', y='Count', color='Type of Breach',
              title='Breach Type Trends Over Time',
              labels={'Count': 'Number of Breaches'})
fig.update_layout(xaxis=dict(dtick=2))
fig.write_image(os.path.join(CHARTS_DIR, '08_breach_type_trends.png'), width=1400, height=600, scale=2)
print('Saved: 08_breach_type_trends.png')

# same but by industry
industry_yearly = df.groupby(['Year', 'Industry']).size().reset_index(name='Count')

fig = px.line(industry_yearly, x='Year', y='Count', color='Industry', markers=True,
              title='Breach Trends by Industry Over Time',
              labels={'Count': 'Number of Breaches'})
fig.update_layout(xaxis=dict(dtick=2))
fig.write_image(os.path.join(CHARTS_DIR, '09_industry_trends_over_time.png'), width=1400, height=600, scale=2)
print('Saved: 09_industry_trends_over_time.png')


# %% Human error factors

error_counts = df['Human Error Factor'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors_err = sns.color_palette('Paired', len(error_counts))
error_counts.plot(kind='bar', ax=axes[0], color=colors_err, edgecolor='black')
axes[0].set_title('Frequency of Human Error Factors', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Human Error Factor')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

error_loss = df.groupby('Human Error Factor')['Financial Loss'].mean().sort_values(ascending=True)
error_loss.plot(kind='barh', ax=axes[1], color=sns.color_palette('RdYlGn_r', len(error_loss)), edgecolor='black')
axes[1].set_title('Avg Financial Loss by Human Error Factor', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Average Financial Loss ($)')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '10_human_error_frequency_and_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 10_human_error_frequency_and_loss.png')

# weak password management leads the pack - not surprising honestly

error_impact = pd.crosstab(df['Human Error Factor'], df['Impact Level'])

error_impact.plot(kind='bar', stacked=True, colormap='RdYlGn_r', edgecolor='black', figsize=(12, 6))
plt.title('Human Error Factor vs Impact Level', fontsize=14, fontweight='bold')
plt.xlabel('Human Error Factor')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Impact Level')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '11_human_error_vs_impact_level.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 11_human_error_vs_impact_level.png')

# how does human error differ by industry?
error_industry = pd.crosstab(df['Industry'], df['Human Error Factor'])

plt.figure(figsize=(12, 6))
sns.heatmap(error_industry, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
plt.title('Human Error Factors Across Industries', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '12_human_error_by_industry_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 12_human_error_by_industry_heatmap.png')


# %% Financial impact deep dive

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(df['Financial Loss'], bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[0].set_title('Distribution of Financial Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Financial Loss ($)')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['Financial Loss'].mean(), color='red', linestyle='--',
                label=f"Mean: ${df['Financial Loss'].mean():,.0f}")
axes[0].axvline(df['Financial Loss'].median(), color='blue', linestyle='--',
                label=f"Median: ${df['Financial Loss'].median():,.0f}")
axes[0].legend()

axes[1].hist(df['Records Compromised'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].set_title('Distribution of Records Compromised', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Records Compromised')
axes[1].set_ylabel('Frequency')
axes[1].axvline(df['Records Compromised'].mean(), color='red', linestyle='--',
                label=f"Mean: {df['Records Compromised'].mean():,.0f}")
axes[1].axvline(df['Records Compromised'].median(), color='blue', linestyle='--',
                label=f"Median: {df['Records Compromised'].median():,.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '13_financial_loss_records_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 13_financial_loss_records_distribution.png')

# mean and median are very close - distribution looks roughly uniform
# (this dataset might be synthetically generated, but still useful to practice with)

# do more records compromised = more financial loss?
fig = px.scatter(df, x='Records Compromised', y='Financial Loss', color='Industry',
                 size='Records Compromised', hover_data=['Company Name', 'Type of Breach', 'Year'],
                 title='Records Compromised vs Financial Loss',
                 labels={'Financial Loss': 'Financial Loss ($)', 'Records Compromised': 'Records Compromised'},
                 opacity=0.6)
fig.write_image(os.path.join(CHARTS_DIR, '14_records_vs_financial_loss_scatter.png'), width=1200, height=600, scale=2)
print('Saved: 14_records_vs_financial_loss_scatter.png')

# violin works better than box here since it shows the full distribution shape
fig = px.violin(df, x='Impact Level', y='Financial Loss', color='Impact Level',
                box=True, points='outliers',
                title='Financial Loss Distribution by Impact Level',
                labels={'Financial Loss': 'Financial Loss ($)'},
                category_orders={'Impact Level': ['Low', 'Medium', 'High', 'Critical']})
fig.write_image(os.path.join(CHARTS_DIR, '15_financial_loss_by_impact_violin.png'), width=1200, height=600, scale=2)
print('Saved: 15_financial_loss_by_impact_violin.png')

numeric_cols = df.select_dtypes(include=[np.number])
correlation = numeric_cols.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.3f', linewidths=1)
plt.title('Correlation Matrix (Numeric Columns)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '16_correlation_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 16_correlation_matrix.png')


# %% Company-level analysis

top_companies = df['Company Name'].value_counts().head(15)

plt.figure(figsize=(14, 6))
colors_comp = sns.color_palette('viridis', len(top_companies))
top_companies.plot(kind='bar', color=colors_comp, edgecolor='black')
plt.title('Top 15 Most Frequently Breached Companies', fontsize=14, fontweight='bold')
plt.xlabel('Company')
plt.ylabel('Number of Breaches')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '17_top15_most_breached_companies.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 17_top15_most_breached_companies.png')

company_loss = df.groupby('Company Name').agg(
    total_loss=('Financial Loss', 'sum'),
    avg_loss=('Financial Loss', 'mean'),
    total_records=('Records Compromised', 'sum'),
    breach_count=('Year', 'count')
).sort_values('total_loss', ascending=False).head(15)

fig = px.bar(company_loss.reset_index(), x='Company Name', y='total_loss',
             color='breach_count', color_continuous_scale='Reds',
             hover_data=['avg_loss', 'total_records', 'breach_count'],
             title='Top 15 Companies by Total Financial Loss',
             labels={'total_loss': 'Total Financial Loss ($)', 'breach_count': 'Breach Count'})
fig.update_layout(xaxis_tickangle=-45)
fig.write_image(os.path.join(CHARTS_DIR, '18_top15_companies_by_loss.png'), width=1200, height=600, scale=2)
print('Saved: 18_top15_companies_by_loss.png')

# stacked bar to see what kinds of attacks hit the top 10 most
top10 = df['Company Name'].value_counts().head(10).index
company_breach = pd.crosstab(df[df['Company Name'].isin(top10)]['Company Name'],
                              df[df['Company Name'].isin(top10)]['Type of Breach'])

company_breach.plot(kind='bar', stacked=True, colormap='Set3', edgecolor='black', figsize=(14, 6))
plt.title('Breach Type Profile of Top 10 Companies', fontsize=14, fontweight='bold')
plt.xlabel('Company')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Breach Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '19_top10_company_breach_profile.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 19_top10_company_breach_profile.png')


# %% Mitigation measures

mitigation_counts = df['Mitigation Measures'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors_mit = sns.color_palette('Set1', len(mitigation_counts))
mitigation_counts.plot(kind='bar', ax=axes[0], color=colors_mit, edgecolor='black')
axes[0].set_title('Frequency of Mitigation Measures', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Mitigation Measure')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

mit_loss = df.groupby('Mitigation Measures')['Financial Loss'].mean().sort_values(ascending=True)
mit_loss.plot(kind='barh', ax=axes[1], color=sns.color_palette('RdYlGn', len(mit_loss)), edgecolor='black')
axes[1].set_title('Average Financial Loss by Mitigation Measure', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Average Financial Loss ($)')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '20_mitigation_frequency_and_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 20_mitigation_frequency_and_loss.png')

mit_breach = pd.crosstab(df['Mitigation Measures'], df['Type of Breach'])

plt.figure(figsize=(12, 6))
sns.heatmap(mit_breach, annot=True, fmt='d', cmap='Greens', linewidths=0.5)
plt.title('Mitigation Measures vs Breach Types', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '21_mitigation_vs_breach_type_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: 21_mitigation_vs_breach_type_heatmap.png')

# sunburst gives a nice hierarchical view of mitigation vs impact
mit_impact = df.groupby(['Mitigation Measures', 'Impact Level']).agg(
    avg_records=('Records Compromised', 'mean'),
    avg_loss=('Financial Loss', 'mean'),
    count=('Year', 'count')
).round(2).reset_index()

fig = px.sunburst(mit_impact, path=['Mitigation Measures', 'Impact Level'], values='count',
                  color='avg_loss', color_continuous_scale='RdYlGn_r',
                  title='Mitigation Measures: Impact Level Breakdown (colored by Avg Financial Loss)',
                  labels={'avg_loss': 'Avg Loss ($)'})
fig.write_image(os.path.join(CHARTS_DIR, '22_mitigation_sunburst.png'), width=1000, height=800, scale=2)
print('Saved: 22_mitigation_sunburst.png')


# %% Summary

print('=' * 60)
print('KEY FINDINGS')
print('=' * 60)

print(f'''
DATASET: {len(df):,} records | {df['Year'].min()}-{df['Year'].max()} | {df['Company Name'].nunique()} companies | {df['Industry'].nunique()} industries

FINANCIAL IMPACT:
  Total loss across all breaches: ${df['Financial Loss'].sum():,.2f}
  Avg per breach: ${df['Financial Loss'].mean():,.2f}
  Median: ${df['Financial Loss'].median():,.2f}
  Worst single breach: ${df['Financial Loss'].max():,.2f}

RECORDS: {df['Records Compromised'].sum():,} total | avg {df['Records Compromised'].mean():,.0f} per breach

MOST COMMON:
  Breach type: {df['Type of Breach'].value_counts().index[0]} ({df['Type of Breach'].value_counts().iloc[0]:,})
  Targeted industry: {df['Industry'].value_counts().index[0]} ({df['Industry'].value_counts().iloc[0]:,})
  Human error: {df['Human Error Factor'].value_counts().index[0]} ({df['Human Error Factor'].value_counts().iloc[0]:,})

COSTLIEST:
  Industry: {df.groupby('Industry')['Financial Loss'].mean().idxmax()} (avg ${df.groupby('Industry')['Financial Loss'].mean().max():,.2f})
  Attack type: {df.groupby('Type of Breach')['Financial Loss'].mean().idxmax()} (avg ${df.groupby('Type of Breach')['Financial Loss'].mean().max():,.2f})
  Company (total): {df.groupby('Company Name')['Financial Loss'].sum().idxmax()} (${df.groupby('Company Name')['Financial Loss'].sum().max():,.2f})

MOST TARGETED: {df['Company Name'].value_counts().index[0]} ({df['Company Name'].value_counts().iloc[0]:,} breaches)
''')

print(f'All 22 charts saved to: {CHARTS_DIR}')
