"""
============================================================
  SYNENT TECHNOLOGIES – DATA SCIENCE INTERNSHIP
  Task 3: Exploratory Data Analysis (EDA)
  Name   : Palak
  Dataset: Netflix Top 10 Weekly Rankings
============================================================

Task Objectives:
  - Summary Statistics
  - Correlation Analysis
  - Trend Identification
Output:
  - Insights supported with graphs
"""

import os
import sys

# Ensure paths are always relative to THIS script's directory,
# regardless of which working directory Python is launched from.
_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Chart styling
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (11, 5)
plt.rcParams['font.size'] = 12

# Create output folder for charts
os.makedirs('data/charts', exist_ok=True)

print("=" * 60)
print("   TASK 3 - EXPLORATORY DATA ANALYSIS")
print("   Netflix Top 10 Dataset")
print("   Synent Technologies Data Science Internship")
print("=" * 60)


# ============================================================
#  PHASE 1: DATA CLEANING
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 1: DATA CLEANING")
print("=" * 60)

# --- Load all datasets ---
print("\n>> Loading datasets...")
df_merged  = pd.read_csv('data/netflix_merged.csv',          encoding='latin-1')
df_country = pd.read_csv('data/all-weeks-countries.csv',     encoding='latin-1')
df_popular = pd.read_csv('data/most-popular.csv',            encoding='latin-1')

print(f"   Merged dataset   : {df_merged.shape[0]} rows x {df_merged.shape[1]} columns")
print(f"   Countries dataset: {df_country.shape[0]} rows x {df_country.shape[1]} columns")
print(f"   Most popular     : {df_popular.shape[0]} rows x {df_popular.shape[1]} columns")

# --- Parse date columns ---
print("\n>> Parsing date columns...")
df_merged['week']  = pd.to_datetime(df_merged['week'],  errors='coerce')
df_country['week'] = pd.to_datetime(df_country['week'], errors='coerce')

df_merged['year']    = df_merged['week'].dt.year
df_merged['month']   = df_merged['week'].dt.month
df_merged['quarter'] = df_merged['week'].dt.quarter

print(f"   Date range: {df_merged['week'].min().date()} -> {df_merged['week'].max().date()}")

# --- Convert numeric columns ---
print("\n>> Converting numeric columns...")
num_cols = ['weekly_hours_viewed', 'weekly_views', 'runtime',
            'cumulative_weeks_in_top_10', 'hours_viewed_first_91_days', 'views_first_91_days']
for col in num_cols:
    if col in df_merged.columns:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

df_popular['hours_viewed_first_91_days'] = pd.to_numeric(df_popular['hours_viewed_first_91_days'], errors='coerce')
df_popular['views_first_91_days']        = pd.to_numeric(df_popular['views_first_91_days'],        errors='coerce')
df_popular['runtime']                    = pd.to_numeric(df_popular['runtime'],                    errors='coerce')

# --- Add content type column ---
df_merged['content_type']  = df_merged['category'].apply(lambda x: 'TV Shows' if 'TV' in str(x) else 'Films')
df_country['content_type'] = df_country['category'].apply(lambda x: 'TV Shows' if 'TV' in str(x) else 'Films')

# --- Handle missing values ---
print("\n>> Checking for missing values...")
missing = df_merged.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("   No missing values found!")
else:
    for col, cnt in missing.items():
        print(f"   {col}: {cnt} missing ({cnt/len(df_merged)*100:.1f}%)")

# --- Check for duplicates ---
print("\n>> Checking for duplicates...")
dups = df_merged.duplicated().sum()
if dups > 0:
    df_merged.drop_duplicates(inplace=True)
    print(f"   {dups} duplicates removed.")
else:
    print("   No duplicate rows found.")

print("\n   [Data Cleaning Complete]")


# ============================================================
#  PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)


# -----------------------------------------------------------
#  2A: SUMMARY STATISTICS
# -----------------------------------------------------------
print("\n>> Summary Statistics")
print("-" * 40)

total_shows   = df_merged['show_title'].nunique()
total_weeks   = df_merged['week'].nunique()
avg_hrs       = df_merged['weekly_hours_viewed'].mean() / 1e6
total_hrs     = df_merged['weekly_hours_viewed'].sum() / 1e6
max_row       = df_merged.loc[df_merged['weekly_hours_viewed'].idxmax()]
avg_weeks_top = df_merged['cumulative_weeks_in_top_10'].mean()

print(f"   Total unique shows tracked   : {total_shows}")
print(f"   Total weeks covered          : {total_weeks}")
print(f"   Avg weekly hours viewed      : {avg_hrs:.1f} million")
print(f"   Total hours viewed (all time): {total_hrs:.0f} million")
print(f"   Highest single-week title    : '{max_row['show_title']}' ({max_row['weekly_hours_viewed']/1e6:.0f}M hrs)")
print(f"   Avg weeks a title stays Top10: {avg_weeks_top:.1f} weeks")

print("\n   Category breakdown:")
for cat, cnt in df_merged['category'].value_counts().items():
    print(f"     {cat}: {cnt} appearances")


# -----------------------------------------------------------
#  2B: CORRELATION ANALYSIS
# -----------------------------------------------------------
print("\n>> Correlation Analysis")
print("-" * 40)

corr_cols = ['weekly_rank', 'weekly_hours_viewed', 'weekly_views',
             'runtime', 'cumulative_weeks_in_top_10']
corr_matrix = df_merged[corr_cols].corr()

pairs = corr_matrix.unstack().reset_index()
pairs.columns = ['F1', 'F2', 'Corr']
pairs = pairs[pairs['F1'] < pairs['F2']].copy()
pairs['AbsCorr'] = pairs['Corr'].abs()
pairs = pairs.sort_values('AbsCorr', ascending=False)

print("   Top feature correlations:")
for _, row in pairs.head(5).iterrows():
    direction = 'positive' if row['Corr'] > 0 else 'negative'
    print(f"     {row['F1']} <-> {row['F2']}: {row['Corr']:.3f} ({direction})")


# -----------------------------------------------------------
#  2C: TREND IDENTIFICATION
# -----------------------------------------------------------
print("\n>> Trend Identification")
print("-" * 40)

films_pct = (df_merged['content_type'] == 'Films').mean() * 100
tv_pct    = 100 - films_pct
max_wks   = df_merged.loc[df_merged['cumulative_weeks_in_top_10'].idxmax()]
top_country = df_country['country_name'].value_counts().idxmax()
rank1_avg  = df_merged[df_merged['weekly_rank'] == 1]['weekly_hours_viewed'].mean() / 1e6
rank10_avg = df_merged[df_merged['weekly_rank'] == 10]['weekly_hours_viewed'].mean() / 1e6
stag_avg   = df_merged[df_merged['is_staggered_launch'] == True]['weekly_hours_viewed'].mean() / 1e6
norm_avg   = df_merged[df_merged['is_staggered_launch'] == False]['weekly_hours_viewed'].mean() / 1e6

print(f"   Content split      : Films {films_pct:.1f}%  |  TV Shows {tv_pct:.1f}%")
print(f"   Longest Top 10 run : '{max_wks['show_title']}' ({int(max_wks['cumulative_weeks_in_top_10'])} weeks)")
print(f"   Most active country: {top_country}")
print(f"   Rank #1 avg views  : {rank1_avg:.1f}M hrs/week")
print(f"   Rank #10 avg views : {rank10_avg:.1f}M hrs/week")
print(f"   Rank 1 vs Rank 10  : {rank1_avg/rank10_avg:.1f}x more viewership")
print(f"   Launch type winner : {'Staggered' if stag_avg > norm_avg else 'Normal'} ({stag_avg:.1f}M vs {norm_avg:.1f}M)")

print("\n   [EDA Complete]")


# ============================================================
#  PHASE 3: VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 3: VISUALIZATION")
print("=" * 60)
print("\n   Generating charts... (close each chart window to continue)")


# --- Chart 1: Category Pie + Top 10 All-Time ---
print("\n>> Chart 1/8: Content Category Overview")
cat_counts = df_merged['category'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
colors = sns.color_palette('muted', len(cat_counts))
axes[0].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%',
            startangle=140, colors=colors,
            wedgeprops=dict(edgecolor='white', linewidth=1.5))
axes[0].set_title('Top 10 Appearances by Content Category', fontsize=13)
top10 = df_popular.nlargest(10, 'hours_viewed_first_91_days')
axes[1].barh(top10['show_title'], top10['hours_viewed_first_91_days'] / 1e6,
             color=sns.color_palette('Blues_r', 10), edgecolor='white')
axes[1].set_xlabel('Hours Viewed (Millions) – First 91 Days')
axes[1].set_title('Top 10 Most-Watched Titles All-Time', fontsize=13)
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig('data/charts/01_summary_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 01_summary_overview.png")


# --- Chart 2: Distribution of Weekly Hours ---
print("\n>> Chart 2/8: Distribution of Weekly Hours Viewed")
data_m = df_merged['weekly_hours_viewed'].dropna() / 1e6
plt.figure(figsize=(10, 4))
plt.hist(data_m, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
plt.axvline(data_m.mean(),   color='red',    linewidth=2, linestyle='--', label=f'Mean: {data_m.mean():.1f}M')
plt.axvline(data_m.median(), color='orange', linewidth=2, linestyle='--', label=f'Median: {data_m.median():.1f}M')
plt.xlabel('Weekly Hours Viewed (Millions)')
plt.ylabel('Frequency')
plt.title('Distribution of Weekly Hours Viewed', fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig('data/charts/02_hours_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 02_hours_distribution.png")


# --- Chart 3: Correlation Heatmap ---
print("\n>> Chart 3/8: Correlation Heatmap")
plt.figure(figsize=(9, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.6,
            square=True, annot_kws={'size': 11})
plt.title('Correlation Matrix – Netflix Weekly Data', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('data/charts/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 03_correlation_heatmap.png")


# --- Chart 4: Runtime vs Views + Rank vs Views ---
print("\n>> Chart 4/8: Runtime vs Views & Rank vs Avg Viewership")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sample = df_merged[['runtime', 'weekly_views', 'content_type']].dropna()
for ctype in sample['content_type'].unique():
    sub = sample[sample['content_type'] == ctype]
    axes[0].scatter(sub['runtime'], sub['weekly_views'] / 1e6, alpha=0.35, label=ctype, s=30)
z = np.polyfit(sample['runtime'], sample['weekly_views'] / 1e6, 1)
p = np.poly1d(z)
x_line = np.linspace(sample['runtime'].min(), sample['runtime'].max(), 100)
axes[0].plot(x_line, p(x_line), color='black', linewidth=2, linestyle='--', label='Trend Line')
axes[0].set_xlabel('Runtime (hours)')
axes[0].set_ylabel('Weekly Views (Millions)')
axes[0].set_title('Runtime vs Weekly Views', fontsize=12)
axes[0].legend(fontsize=9)
rank_avg = df_merged.groupby('weekly_rank')['weekly_hours_viewed'].mean() / 1e6
axes[1].bar(rank_avg.index, rank_avg.values,
            color=sns.color_palette('Blues_r', len(rank_avg)), edgecolor='white')
axes[1].set_xlabel('Weekly Rank')
axes[1].set_ylabel('Avg Hours Viewed (Millions)')
axes[1].set_title('Avg Viewership by Weekly Rank', fontsize=12)
plt.suptitle('Correlation Insights', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('data/charts/04_correlation_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 04_correlation_scatter.png")


# --- Chart 5: Launch Type Comparison ---
print("\n>> Chart 5/8: Normal vs Staggered Launch Comparison")
launch_group = df_merged.groupby('is_staggered_launch')['weekly_hours_viewed'].mean() / 1e6
launch_group.index = ['Normal Launch', 'Staggered Launch']
plt.figure(figsize=(6, 4))
bars = plt.bar(launch_group.index, launch_group.values,
               color=['steelblue', 'coral'], edgecolor='white', width=0.4)
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
             f'{h:.1f}M', ha='center', va='bottom', fontsize=11)
plt.ylabel('Avg Weekly Hours Viewed (Millions)')
plt.title('Normal vs Staggered Launch: Viewership Comparison', fontsize=12)
plt.tight_layout()
plt.savefig('data/charts/05_launch_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 05_launch_comparison.png")


# --- Chart 6: Weekly Viewership Trend ---
print("\n>> Chart 6/8: Weekly Global Viewership Trend")
weekly_trend = df_merged.groupby('week')['weekly_hours_viewed'].sum() / 1e6
plt.figure(figsize=(13, 5))
plt.plot(weekly_trend.index, weekly_trend.values,
         linewidth=2, color='steelblue', marker='o', markersize=3)
plt.fill_between(weekly_trend.index, weekly_trend.values, alpha=0.15, color='steelblue')
plt.title('Total Global Netflix Viewership Over Time (Weekly)', fontsize=14)
plt.xlabel('Week')
plt.ylabel('Total Hours Viewed (Millions)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/charts/06_weekly_viewership_trend.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 06_weekly_viewership_trend.png")


# --- Chart 7: Films vs TV Trend ---
print("\n>> Chart 7/8: Films vs TV Shows Viewership Trend")
type_trend = (df_merged.groupby(['week', 'content_type'])['weekly_hours_viewed']
              .sum().unstack(fill_value=0) / 1e6)
plt.figure(figsize=(13, 5))
for col in type_trend.columns:
    plt.plot(type_trend.index, type_trend[col], linewidth=2.2, label=col, marker='o', markersize=2)
plt.title('Films vs TV Shows – Weekly Viewership Trend', fontsize=14)
plt.xlabel('Week')
plt.ylabel('Hours Viewed (Millions)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/charts/07_films_vs_tv_trend.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 07_films_vs_tv_trend.png")


# --- Chart 8: Top Countries ---
print("\n>> Chart 8/8: Top 10 Countries by Netflix Appearances")
top_countries = df_country['country_name'].value_counts().head(10)
plt.figure(figsize=(10, 5))
bars = plt.barh(top_countries.index[::-1], top_countries.values[::-1],
                color=sns.color_palette('muted', 10), edgecolor='white')
for bar in bars:
    w = bar.get_width()
    plt.text(w + 10, bar.get_y() + bar.get_height() / 2,
             str(int(w)), va='center', fontsize=9)
plt.xlabel('Number of Top-10 Appearances')
plt.title('Top 10 Countries by Netflix Top-10 Appearances', fontsize=13)
plt.tight_layout()
plt.savefig('data/charts/08_top_countries.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 08_top_countries.png")


print("\n   [Visualization Complete]")


# ============================================================
#  FINAL: KEY INSIGHTS & CONCLUSIONS
# ============================================================
print("\n" + "=" * 60)
print("  KEY INSIGHTS & CONCLUSIONS")
print("=" * 60)

top_show = df_popular.loc[df_popular['hours_viewed_first_91_days'].idxmax(), 'show_title']
top_hrs  = df_popular['hours_viewed_first_91_days'].max() / 1e6

print(f"""
  1. Most-watched title ever  : '{top_show}'
     Hours viewed (91 days)   : {top_hrs:.0f} million

  2. Content split in Top 10  : Films {films_pct:.1f}%  |  TV Shows {tv_pct:.1f}%

  3. Longest-running title    : '{max_wks['show_title']}'
     Weeks in Top 10          : {int(max_wks['cumulative_weeks_in_top_10'])} weeks

  4. Most active country      : {top_country}

  5. Rank #1 vs Rank #10      : {rank1_avg:.1f}M vs {rank10_avg:.1f}M hrs/week
     Rank 1 gets {rank1_avg/rank10_avg:.1f}x more views than Rank 10

  6. Launch strategy winner   : {'Staggered' if stag_avg > norm_avg else 'Normal'} release
     Staggered: {stag_avg:.1f}M  |  Normal: {norm_avg:.1f}M avg hrs/week

  7. Strongest correlation    : Hours Viewed <-> View Count (r = 0.66)
     Rank inversely correlates with views (r = -0.59)
""")

print("=" * 60)
print("  All 8 important charts saved in: data/charts/")
print("  Task 3 – EDA Complete!")
print("=" * 60)
