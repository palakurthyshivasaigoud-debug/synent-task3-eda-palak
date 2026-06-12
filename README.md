# synent-task3-eda-palak

## Task 3: Netflix Exploratory Data Analysis

This project was completed for the Synent Technologies Data Science Internship. The aim is to study Netflix Top 10 viewing data and explain patterns in content popularity, category performance, country activity, and weekly viewing trends.

## Problem Statement

Netflix releases weekly ranking data for films and TV shows. I used this dataset to answer practical EDA questions such as which titles received the highest viewing hours, whether films and TV shows behave differently, how rank relates to viewership, and which countries appear most often in the Top 10 list.

## Dataset

**Source:** [Official Netflix Viewership Database](https://www.kaggle.com/datasets/sujaykapadnis/official-netflix-streaming-data) - Kaggle

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
3. Fixed the data merge to use `season_title` as an additional join key, preventing duplicate rows from multi-season shows.
4. Checked missing values and duplicate records.
5. Created content type fields for Films and TV Shows.
6. Calculated summary statistics for titles, weeks, viewership, ranks, and categories.
7. Built correlation analysis for rank, hours viewed, weekly views, runtime, and weeks in Top 10.
8. Generated 6 consolidated, insight-driven charts.

## Key Insights

- **Squid Game** is the most-watched Netflix title ever with 2,205 million hours viewed in its first 91 days.
- Films and TV Shows have an equal 50% share in the global Top 10 dataset.
- Rank #1 titles receive **8.7x more** average viewing hours than Rank #10 titles.
- **Staggered launches** outperform normal releases (22.2M vs 18.7M avg hrs/week).
- `weekly_hours_viewed` and `weekly_views` have the strongest positive correlation (r = 0.66).
- Country-level diversity analysis shows English-speaking countries tend to have the widest spread of unique titles in their Top 10.

## Visualizations

Charts are saved in `data/charts/`.

| Chart | Description |
| --- | --- |
| `01_content_overview.png` | Content type share (pie) + Top 10 Most-Watched Titles all-time (bar) |
| `02_viewership_distribution.png` | Weekly hours viewed histogram + Boxplots of key metrics |
| `03_correlation_analysis.png` | Feature correlation heatmap + Avg viewership by weekly rank |
| `04_viewership_trends.png` | Films vs TV Shows weekly trend (line) + Quarterly seasonality (bar) |
| `05_country_analysis.png` | Top 15 countries by content diversity + Avg longevity by content type |
| `06_launch_strategy.png` | Normal vs Staggered launch comparison + Runtime vs Weekly Views scatter |

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

Palakurthy Shiva Sai Goud

Submitted for Synent Technologies Data Science Internship - Task 3.
