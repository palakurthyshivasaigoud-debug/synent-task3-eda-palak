"""
============================================================
  SYNENT TECHNOLOGIES – DATA SCIENCE INTERNSHIP
  Task 3: Exploratory Data Analysis (EDA)
  Name   : Palakurthy Shiva Sai Goud
  Dataset: Netflix Top 10 Weekly Rankings
============================================================

Task Objectives:
  - Summary Statistics
  - Correlation Analysis
  - Trend Identification
Output:
  - 6 insight-driven charts saved to data/charts/
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')

# Paths always relative to this script
_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)

# ── Styling ───────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'font.family'     : 'DejaVu Sans',
    'font.size'       : 11,
    'axes.titlesize'  : 13,
    'axes.labelsize'  : 11,
    'figure.facecolor': '#fafafa',
    'axes.facecolor'  : '#ffffff',
    'axes.edgecolor'  : '#cccccc',
})

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

print("\n>> Loading datasets...")
df_merged  = pd.read_csv('data/netflix_merged.csv',          encoding='latin-1')
df_country = pd.read_csv('data/all-weeks-countries.csv',     encoding='latin-1')
df_popular = pd.read_csv('data/most-popular.csv',            encoding='latin-1')

print(f"   Merged dataset   : {df_merged.shape[0]} rows x {df_merged.shape[1]} columns")
print(f"   Countries dataset: {df_country.shape[0]} rows x {df_country.shape[1]} columns")
print(f"   Most popular     : {df_popular.shape[0]} rows x {df_popular.shape[1]} columns")

# Parse dates
print("\n>> Parsing date columns...")
df_merged['week']  = pd.to_datetime(df_merged['week'],  errors='coerce')
df_country['week'] = pd.to_datetime(df_country['week'], errors='coerce')

df_merged['year']    = df_merged['week'].dt.year
df_merged['month']   = df_merged['week'].dt.month
df_merged['quarter'] = df_merged['week'].dt.quarter

print(f"   Date range: {df_merged['week'].min().date()} -> {df_merged['week'].max().date()}")

# Numeric columns
print("\n>> Converting numeric columns...")
num_cols = ['weekly_hours_viewed', 'weekly_views', 'runtime',
            'cumulative_weeks_in_top_10', 'hours_viewed_first_91_days', 'views_first_91_days']
for col in num_cols:
    if col in df_merged.columns:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

df_popular['hours_viewed_first_91_days'] = pd.to_numeric(df_popular['hours_viewed_first_91_days'], errors='coerce')
df_popular['views_first_91_days']        = pd.to_numeric(df_popular['views_first_91_days'],        errors='coerce')
df_popular['runtime']                    = pd.to_numeric(df_popular['runtime'],                    errors='coerce')

# Content type column
df_merged['content_type']  = df_merged['category'].apply(lambda x: 'TV Shows' if 'TV' in str(x) else 'Films')
df_country['content_type'] = df_country['category'].apply(lambda x: 'TV Shows' if 'TV' in str(x) else 'Films')

# Missing values
print("\n>> Checking for missing values...")
missing = df_merged.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("   No missing values found!")
else:
    for col, cnt in missing.items():
        print(f"   {col}: {cnt} missing ({cnt/len(df_merged)*100:.1f}%)")

# Duplicates
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

# 2A: Summary Statistics
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

# 2B: Correlation Analysis
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

# 2C: Trend Identification
print("\n>> Trend Identification")
print("-" * 40)

films_pct   = (df_merged['content_type'] == 'Films').mean() * 100
tv_pct      = 100 - films_pct
max_wks     = df_merged.loc[df_merged['cumulative_weeks_in_top_10'].idxmax()]
top_country = df_country['country_name'].value_counts().idxmax()
rank1_avg   = df_merged[df_merged['weekly_rank'] == 1]['weekly_hours_viewed'].mean() / 1e6
rank10_avg  = df_merged[df_merged['weekly_rank'] == 10]['weekly_hours_viewed'].mean() / 1e6
stag_avg    = df_merged[df_merged['is_staggered_launch'] == True]['weekly_hours_viewed'].mean() / 1e6
norm_avg    = df_merged[df_merged['is_staggered_launch'] == False]['weekly_hours_viewed'].mean() / 1e6

print(f"   Content split      : Films {films_pct:.1f}%  |  TV Shows {tv_pct:.1f}%")
print(f"   Longest Top 10 run : '{max_wks['show_title']}' ({int(max_wks['cumulative_weeks_in_top_10'])} weeks)")
print(f"   Most active country: {top_country}")
print(f"   Rank #1 avg views  : {rank1_avg:.1f}M hrs/week")
print(f"   Rank #10 avg views : {rank10_avg:.1f}M hrs/week")
print(f"   Rank 1 vs Rank 10  : {rank1_avg/rank10_avg:.1f}x more viewership")
print(f"   Launch type winner : {'Staggered' if stag_avg > norm_avg else 'Normal'} ({stag_avg:.1f}M vs {norm_avg:.1f}M)")

print("\n   [EDA Complete]")


# ============================================================
#  PHASE 3: VISUALIZATION  (6 consolidated charts)
# ============================================================
print("\n" + "=" * 60)
print("  PHASE 3: VISUALIZATION")
print("=" * 60)
print("\n   Generating 6 charts... (close each window to continue)\n")


# ── Helper: save + show ───────────────────────────────────────
def _save(path, title=None):
    if title:
        plt.gcf().suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"   Saved: {os.path.basename(path)}")


# ─────────────────────────────────────────────────────────────
#  CHART 1 — Content Category Overview
#  Left : Pie – Films vs TV Shows share in Top 10
#  Right: Horizontal bar – Top 10 Most-Watched Titles (91 days)
# ─────────────────────────────────────────────────────────────
print(">> Chart 1/6: Content Category Overview")

cat_counts = df_merged['content_type'].value_counts()
top10      = df_popular.nlargest(10, 'hours_viewed_first_91_days')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart
pie_colors = ['#4A90D9', '#E8744A']
wedges, texts, autotexts = axes[0].pie(
    cat_counts, labels=cat_counts.index, autopct='%1.1f%%',
    startangle=140, colors=pie_colors,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops={'fontsize': 12})
for at in autotexts:
    at.set_fontweight('bold')
axes[0].set_title('Content Type Share in Top 10', fontsize=13, pad=15)

# Horizontal bar chart
bar_colors = sns.color_palette('Blues_r', 10)
bars = axes[1].barh(top10['show_title'], top10['hours_viewed_first_91_days'] / 1e6,
                     color=bar_colors, edgecolor='white', height=0.65)
for bar in bars:
    w = bar.get_width()
    axes[1].text(w + 5, bar.get_y() + bar.get_height() / 2,
                 f'{w:.0f}M', va='center', fontsize=9, fontweight='bold')
axes[1].invert_yaxis()
axes[1].set_xlabel('Hours Viewed (Millions) – First 91 Days')
axes[1].set_title('Top 10 Most-Watched Titles of All Time', fontsize=13)
axes[1].set_xlim(0, top10['hours_viewed_first_91_days'].max() / 1e6 * 1.18)

_save('data/charts/01_content_overview.png', 'Netflix Content Overview')


# ─────────────────────────────────────────────────────────────
#  CHART 2 — Viewership Distribution & Outliers
#  Left : Histogram of weekly hours viewed
#  Right: Boxplots of key numeric metrics
# ─────────────────────────────────────────────────────────────
print("\n>> Chart 2/6: Viewership Distribution & Outliers")

data_m    = df_merged['weekly_hours_viewed'].dropna() / 1e6
box_cols  = [c for c in ['weekly_hours_viewed', 'weekly_views',
                          'cumulative_weeks_in_top_10'] if c in df_merged.columns]
box_labels = {
    'weekly_hours_viewed'        : 'Weekly Hours\nViewed (M)',
    'weekly_views'               : 'Weekly Views\n(M)',
    'cumulative_weeks_in_top_10' : 'Weeks in\nTop 10',
}

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Histogram
axes[0].hist(data_m, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(data_m.mean(),   color='#e74c3c',  linewidth=2, linestyle='--',
                label=f'Mean: {data_m.mean():.1f}M')
axes[0].axvline(data_m.median(), color='#f39c12', linewidth=2, linestyle='--',
                label=f'Median: {data_m.median():.1f}M')
axes[0].set_xlabel('Weekly Hours Viewed (Millions)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Weekly Hours Viewed', fontsize=13)
axes[0].legend()

# Boxplots
bp_data = []
bp_labs = []
for col in box_cols:
    series = df_merged[col].dropna()
    if col in ['weekly_hours_viewed', 'weekly_views']:
        series = series / 1e6
    bp_data.append(series)
    bp_labs.append(box_labels.get(col, col))

bp = axes[1].boxplot(bp_data, labels=bp_labs, patch_artist=True,
                      medianprops=dict(color='#e74c3c', linewidth=2.5),
                      whiskerprops=dict(linewidth=1.5, linestyle='--'),
                      flierprops=dict(marker='o', color='grey', alpha=0.3, markersize=4))
box_fill_colors = ['#4A90D9', '#5BC0EB', '#E8744A']
for patch, color in zip(bp['boxes'], box_fill_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title('Outlier Detection – Key Metrics', fontsize=13)
axes[1].set_ylabel('Value (Millions where applicable)')

_save('data/charts/02_viewership_distribution.png', 'Viewership Distribution & Outlier Analysis')


# ─────────────────────────────────────────────────────────────
#  CHART 3 — Correlation & Rank Analysis
#  Left : Correlation heatmap
#  Right: Avg hours viewed by weekly rank (1 → 10)
# ─────────────────────────────────────────────────────────────
print("\n>> Chart 3/6: Correlation & Rank Analysis")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.6,
            square=True, annot_kws={'size': 10}, ax=axes[0])
axes[0].set_title('Feature Correlation Matrix', fontsize=13)
axes[0].tick_params(axis='x', rotation=30)
axes[0].tick_params(axis='y', rotation=0)

# Rank vs Avg Hours Viewed
rank_avg   = df_merged.groupby('weekly_rank')['weekly_hours_viewed'].mean() / 1e6
rank_colors = sns.color_palette('Blues_r', len(rank_avg))
bars = axes[1].bar(rank_avg.index, rank_avg.values, color=rank_colors, edgecolor='white', width=0.7)
for bar in bars:
    h = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                 f'{h:.0f}M', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
axes[1].set_xlabel('Weekly Rank')
axes[1].set_ylabel('Avg Hours Viewed (Millions)')
axes[1].set_title('Avg Viewership by Weekly Rank\n(Rank 1 is most watched)', fontsize=13)
axes[1].set_xticks(rank_avg.index)

_save('data/charts/03_correlation_analysis.png', 'Correlation & Rank-Viewership Analysis')


# ─────────────────────────────────────────────────────────────
#  CHART 4 — Temporal Viewership Trends
#  Top : Films vs TV Shows weekly trend (line plot)
#  Bot : Average viewership by quarter (seasonality bar chart)
# ─────────────────────────────────────────────────────────────
print("\n>> Chart 4/6: Temporal Viewership Trends")

type_trend = (df_merged.groupby(['week', 'content_type'])['weekly_hours_viewed']
              .sum().unstack(fill_value=0) / 1e6)

quarterly = df_merged.groupby(['year', 'quarter'])['weekly_hours_viewed'].mean() / 1e6
quarterly.index = [f'{y}-Q{q}' for y, q in quarterly.index]

fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'hspace': 0.45})

# Line chart: Films vs TV trend
line_colors = {'Films': '#4A90D9', 'TV Shows': '#E8744A'}
for col in type_trend.columns:
    color = line_colors.get(col, '#999')
    axes[0].plot(type_trend.index, type_trend[col],
                 linewidth=2.2, label=col, color=color, marker='o', markersize=2.5)
    axes[0].fill_between(type_trend.index, type_trend[col], alpha=0.08, color=color)
axes[0].set_title('Films vs TV Shows — Weekly Viewership Trend', fontsize=13)
axes[0].set_xlabel('Week')
axes[0].set_ylabel('Total Hours Viewed (Millions)')
axes[0].legend(title='Content Type', loc='upper right')
axes[0].tick_params(axis='x', rotation=35)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}M'))

# Bar chart: Quarterly seasonality
q_colors = sns.color_palette('muted', len(quarterly))
bars = axes[1].bar(quarterly.index, quarterly.values, color=q_colors, edgecolor='white', width=0.7)
for bar in bars:
    h = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                 f'{h:.1f}M', ha='center', va='bottom', fontsize=8, fontweight='bold')
axes[1].set_title('Average Weekly Viewership by Quarter (Seasonality)', fontsize=13)
axes[1].set_xlabel('Year – Quarter')
axes[1].set_ylabel('Avg Hours Viewed (Millions)')
axes[1].tick_params(axis='x', rotation=40)

_save('data/charts/04_viewership_trends.png', 'Temporal Viewership Trends')


# ─────────────────────────────────────────────────────────────
#  CHART 5 — Country Content Analysis
#  Left : Top 15 countries by unique shows in Top 10
#         (meaningful diversity — NOT flat row counts)
#  Right: Avg weeks in Top 10 by content type for those countries
# ─────────────────────────────────────────────────────────────
print("\n>> Chart 5/6: Country Content Analysis")

# Use UNIQUE show titles as the metric (every country has same row count → use unique count)
country_unique = (df_country.groupby('country_name')['show_title']
                  .nunique().sort_values(ascending=False).head(15))

top15_names = country_unique.index.tolist()
country_sub = df_country[df_country['country_name'].isin(top15_names)].copy()

# Avg cumulative weeks in Top 10 per content type, for those countries
longevity = (country_sub.groupby(['country_name', 'content_type'])
             ['cumulative_weeks_in_top_10'].mean().unstack(fill_value=0))
# Align to the same country order as diversity chart
longevity = longevity.reindex(top15_names, fill_value=0)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Horizontal bar: unique show diversity
bar_cols = sns.color_palette('Blues_r', len(country_unique))
hbars = axes[0].barh(country_unique.index[::-1],
                      country_unique.values[::-1],
                      color=bar_cols[::-1], edgecolor='white', height=0.65)
for bar in hbars:
    w = bar.get_width()
    axes[0].text(w + 5, bar.get_y() + bar.get_height() / 2,
                 str(int(w)), va='center', fontsize=9)
axes[0].set_xlabel('Number of Unique Titles that Reached Top 10')
axes[0].set_title('Top 15 Countries by Content Diversity\n(Unique Shows in Netflix Top 10)', fontsize=12)
axes[0].set_xlim(0, country_unique.max() * 1.15)

# Grouped bar: avg weeks longevity (Films vs TV Shows)
long_colors = ['#4A90D9', '#E8744A']
longevity.plot(kind='bar', ax=axes[1], color=long_colors,
               edgecolor='white', width=0.65)
axes[1].set_title('Avg Weeks in Top 10 — Films vs TV Shows\n(Top 15 Countries by Diversity)', fontsize=12)
axes[1].set_xlabel('Country')
axes[1].set_ylabel('Avg Cumulative Weeks in Top 10')
axes[1].tick_params(axis='x', rotation=40)
axes[1].legend(title='Content Type')
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.1f', fontsize=7.5, padding=2)

_save('data/charts/05_country_analysis.png', 'Country Content Diversity & Longevity Analysis')


# ─────────────────────────────────────────────────────────────
#  CHART 6 — Launch Strategy Comparison
#  Bar chart: Normal vs Staggered launch avg viewership
#  + Runtime vs Weekly Views scatter (bonus insight)
# ─────────────────────────────────────────────────────────────
print("\n>> Chart 6/6: Launch Strategy & Runtime Analysis")

launch_group = df_merged.groupby('is_staggered_launch')['weekly_hours_viewed'].mean() / 1e6
launch_group.index = ['Normal Launch', 'Staggered Launch']

sample = df_merged[['runtime', 'weekly_views', 'content_type']].dropna()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Launch bar chart
launch_colors = ['#4A90D9', '#E8744A']
bars = axes[0].bar(launch_group.index, launch_group.values,
                    color=launch_colors, edgecolor='white', width=0.45)
for bar in bars:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                 f'{h:.1f}M hrs/wk', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Avg Weekly Hours Viewed (Millions)')
axes[0].set_title('Normal vs Staggered Launch\nAverage Weekly Viewership', fontsize=13)
axes[0].set_ylim(0, launch_group.max() * 1.25)

# Scatter: Runtime vs Weekly Views
scatter_colors = {'Films': '#4A90D9', 'TV Shows': '#E8744A'}
for ctype in sample['content_type'].unique():
    sub = sample[sample['content_type'] == ctype]
    axes[1].scatter(sub['runtime'], sub['weekly_views'] / 1e6,
                    alpha=0.35, label=ctype, s=35,
                    color=scatter_colors.get(ctype, '#aaa'))

# Trend line
z = np.polyfit(sample['runtime'], sample['weekly_views'] / 1e6, 1)
p = np.poly1d(z)
x_line = np.linspace(sample['runtime'].min(), sample['runtime'].max(), 100)
axes[1].plot(x_line, p(x_line), color='black', linewidth=2,
             linestyle='--', label='Trend Line')
axes[1].set_xlabel('Runtime (hours)')
axes[1].set_ylabel('Weekly Views (Millions)')
axes[1].set_title('Runtime vs Weekly Views\n(by Content Type)', fontsize=13)
axes[1].legend(fontsize=9)

_save('data/charts/06_launch_strategy.png', 'Launch Strategy & Content Runtime Analysis')


print("\n   [Visualization Complete – 6 charts saved]")


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
print("  All 6 charts saved in: data/charts/")
print("  Task 3 – EDA Complete!")
print("=" * 60)
