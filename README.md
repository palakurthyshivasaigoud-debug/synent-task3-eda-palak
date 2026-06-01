# synent-task3-eda-palak

## Task 3: Netflix Exploratory Data Analysis

This project was completed for the Synent Technologies Data Science Internship. The aim is to study Netflix Top 10 viewing data and explain patterns in content popularity, category performance, country activity, and weekly viewing trends.

## Problem Statement

Netflix releases weekly ranking data for films and TV shows. I used this dataset to answer practical EDA questions such as which titles received the highest viewing hours, whether films and TV shows behave differently, how rank relates to viewership, and which countries appear most often in the Top 10 list.

## Dataset

Source: Netflix Top 10 dataset from Kaggle / Netflix public ranking data.

Files used:

| File | Purpose |
| --- | --- |
| `data/all-weeks-global.csv` | Weekly global Top 10 ranking records |
| `data/all-weeks-countries.csv` | Country-level weekly Top 10 records |
| `data/most-popular.csv` | All-time popular titles based on first 91 days |
| `data/netflix_merged.csv` | Prepared dataset used by the final EDA script |

## Approach

1. Loaded the global, country, and most-popular datasets.
2. Parsed date columns and converted numeric fields for analysis.
3. Checked missing values and duplicate records.
4. Created content type fields for Films and TV Shows.
5. Calculated summary statistics for titles, weeks, viewership, ranks, and categories.
6. Built correlation analysis for rank, hours viewed, weekly views, runtime, and weeks in Top 10.
7. Generated charts to support the final insights.

## Key Insights

- TV Shows and Films have a nearly balanced share in the Top 10 dataset.
- Rank 1 titles receive much higher average viewing hours than Rank 10 titles.
- `weekly_hours_viewed` and `weekly_views` have the strongest positive relationship among the selected numeric fields.
- Country-level data shows which regions repeatedly appear in Netflix Top 10 rankings.
- Staggered launch titles were compared against normal launches to understand release strategy impact.

## Visualizations

Charts are saved in `data/charts/`.

| Chart | Description |
| --- | --- |
| `01_summary_overview.png` | Category share and most-watched titles |
| `02_hours_distribution.png` | Distribution of weekly viewing hours |
| `03_correlation_heatmap.png` | Correlation matrix |
| `04_correlation_scatter.png` | Runtime/view and rank/view relationships |
| `05_launch_comparison.png` | Normal vs staggered launch comparison |
| `06_weekly_viewership_trend.png` | Weekly global viewing trend |
| `07_films_vs_tv_trend.png` | Films and TV Shows over time |
| `08_top_countries.png` | Countries with the most Top 10 appearances |

## How to Run

```bash
pip install -r requirements.txt
python merge_datasets.py
python eda_netflix.py
```

## Repository Structure

```text
synent-task3-eda-palak/
|-- eda_netflix.py
|-- merge_datasets.py
|-- README.md
|-- requirements.txt
|-- data/
|   |-- all-weeks-global.csv
|   |-- all-weeks-countries.csv
|   |-- most-popular.csv
|   |-- netflix_merged.csv
|   `-- charts/
`-- .gitignore
```

## Internship Requirement Mapping

| Requirement | Status |
| --- | --- |
| Summary statistics | Completed |
| Correlation analysis | Completed |
| Trend identification | Completed |
| Graph-supported insights | Completed |
| Dataset included | Completed |

## Author

Palak

Submitted for Synent Technologies Data Science Internship - Task 3.
