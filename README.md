# The Growth Paradox Rising GDP, Uneven Gains 1980–2024

## Project Overview
This project investigates whether economic growth reduces inequality using a global country-year panel dataset from **1980 to 2024**.  
The analysis focuses on the relationship between:

- **Inequality** (`gini_index`)
- **Economic output** (`gdp`, transformed as `log_gdp`)
- **Poverty** (`poverty_rate`)

The project is designed as a **portfolio-ready, end-to-end data science workflow** with data cleaning, feature engineering, visual analysis, econometric modeling, and business/policy recommendations.

---

## Business Question
**Does economic growth reduce inequality, or does inequality persist unless poverty is directly addressed?**

This question is relevant for:
- Governments designing inclusive-growth policy
- International development institutions
- Businesses entering emerging markets with uneven income distribution

---

## Dataset
Single source dataset (no external joins used):

`disuguaglianza-economica-globale-e-povert-1980-2024.csv`

Key fields used:
- `country`
- `year`
- `gdp`
- `gdp_per_capita` (exploratory context)
- `poverty_rate`
- `gini_index`
- `income_top1`
- `income_top10`
- `income_bottom50`
- `population`
- `iso_code`

---

## Methodology

### 1) Data Understanding
- Checked shape, schema, missingness, and variable meanings
- Confirmed panel structure: **country × year**

### 2) Data Cleaning
- Converted fields to numeric where required
- Removed duplicates
- Ensured proper time indexing with `year`
- Standardized country naming where needed

### 3) Feature Engineering
Created:
- `GDP_growth = (GDP_t - GDP_t-1) / GDP_t-1`
- `lag_GDP`
- `lag_poverty`
- `lag_Gini`
- `income_share_ratio_top_bottom = income_top1 / income_bottom50` (when available)
- `log_gdp = ln(gdp)` for modeling stability

### 4) Exploratory Data Analysis
- Distribution and summary statistics
- Correlation heatmap
- Global trend lines:
  - average inequality over time
  - average poverty over time
- Scatter + regression visuals:
  - GDP vs Gini
  - Poverty vs Gini

### 5) Country Case Studies
Focused on:
- India
- China
- United States

Compared trajectories of GDP, poverty, and inequality.

### 6) Econometric Modeling
- **Model 1: Pooled OLS**
  - `gini_index ~ log_gdp + poverty_rate`
- **Model 2: Two-way Fixed Effects**
  - Country FE + Year FE
  - `gini_it = β1 log_gdp_it + β2 poverty_it + α_i + γ_t + ε_it`
- **Model 3: Lag Structure**
  - `gini_t ~ lag_GDP + lag_poverty (+ FE controls)` for temporal direction

Robust standard errors and FE controls were used to improve interpretability.

---

## Key Insights (Template to Customize with Your Final Coefficients)

1. **Growth alone may not guarantee lower inequality.**  
   In many contexts, GDP growth can coexist with persistent inequality.

2. **Poverty is a strong and consistent correlate of inequality.**  
   Where poverty remains elevated, inequality tends to remain high.

3. **Within-country dynamics matter.**  
   Fixed effects show that country-specific structural factors and global shocks strongly influence outcomes.

4. **Lag analysis improves temporal interpretation.**  
   Using lagged predictors helps reduce simultaneity concerns and gives a better approximation of directional relationships.

---

## Policy Recommendations
- Prioritize **inclusive growth**, not growth-only targets
- Combine growth strategy with:
  - targeted social transfers
  - labor formalization
  - education and health access
  - place-based development
- Track inequality KPIs alongside GDP metrics

---

## Business Implications
Before market entry, firms should assess:
- Income distribution structure (not just average GDP)
- Poverty depth and affordability constraints
- Demand polarization (premium vs value segments)
- Social/regulatory risk in high-inequality environments

---

## Project Structure
```text
.
├── does_growth_reduce_inequality_panel_analysis.ipynb
├── does_growth_reduce_inequality_panel_analysis.py
├── data/
│   └── disuguaglianza-economica-globale-e-povert-1980-2024.csv
└── README.md
```

---

## Tech Stack
- Python
- pandas, numpy
- matplotlib, seaborn
- statsmodels
- linearmodels (for panel fixed effects, optional fallback included)

---

## How to Run

1. Clone the repository
2. Place dataset in the expected path (or update `DATA_PATH` in the script/notebook)
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn statsmodels linearmodels
   ```
4. Run notebook top-to-bottom, or run script:
   ```bash
   python does_growth_reduce_inequality_panel_analysis.py
   ```

---

## Notes
- This project intentionally uses **only one dataset**.
- Missingness in inequality/poverty fields is substantial in some country-year pairs; models are estimated on complete cases for required variables.
- Interpret results as observational evidence, not definitive causal proof.

---

## Author
**Utkarsh Kherwal**  
Data Science | Economics | Applied Econometrics
