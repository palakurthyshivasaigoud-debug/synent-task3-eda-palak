import pandas as pd

# Load all three datasets
df_global  = pd.read_csv(r'c:\Internship\synent-task3-eda-palak\data\all-weeks-global.csv',  encoding='latin-1')
df_popular = pd.read_csv(r'c:\Internship\synent-task3-eda-palak\data\most-popular.csv',       encoding='latin-1')
df_country = pd.read_csv(r'c:\Internship\synent-task3-eda-palak\data\all-weeks-countries.csv', encoding='latin-1')

print("Files loaded:")
print(f"  all-weeks-global   -> {df_global.shape}")
print(f"  most-popular       -> {df_popular.shape}")
print(f"  all-weeks-countries-> {df_country.shape}")

# Fix numeric columns in global data
df_global['weekly_hours_viewed'] = pd.to_numeric(df_global['weekly_hours_viewed'], errors='coerce')
df_global['weekly_views']        = pd.to_numeric(df_global['weekly_views'], errors='coerce')
df_global['runtime']             = pd.to_numeric(df_global['runtime'], errors='coerce')

# Fix numeric columns in most-popular
df_popular['hours_viewed_first_91_days'] = pd.to_numeric(df_popular['hours_viewed_first_91_days'], errors='coerce')
df_popular['views_first_91_days']        = pd.to_numeric(df_popular['views_first_91_days'], errors='coerce')
df_popular['runtime']                    = pd.to_numeric(df_popular['runtime'], errors='coerce')

# -------------------------------------------------------------------
# FIX: Clean season_title before merging.
# The original merge on ['show_title', 'category'] introduced duplicate
# rows because some shows appear multiple seasons in most-popular.csv.
# Using season_title as an additional key gives an exact match and keeps
# the merged shape identical to all-weeks-global (5,160 rows).
# -------------------------------------------------------------------
df_global['season_title']  = df_global['season_title'].fillna('N/A').str.strip()
df_popular['season_title'] = df_popular['season_title'].fillna('N/A').str.strip()

# Merge: global weekly data + all-time popularity stats (no duplicates)
df_merged = df_global.merge(
    df_popular[['show_title', 'category', 'season_title',
                'hours_viewed_first_91_days', 'views_first_91_days']],
    on=['show_title', 'category', 'season_title'],
    how='left'
)

# Save merged dataset
output_path = r'c:\Internship\synent-task3-eda-palak\data\netflix_merged.csv'
df_merged.to_csv(output_path, index=False)

print("\nMerge complete!")
print(f"  Merged shape : {df_merged.shape[0]} rows x {df_merged.shape[1]} columns")
print(f"  Columns      : {list(df_merged.columns)}")
print(f"  Saved to     : {output_path}")
matched = df_merged['hours_viewed_first_91_days'].notna().sum()
print(f"  Rows matched with all-time popularity data: {matched}")
