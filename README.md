# It's Not Just Risk—It's Responsibility: Changing Drivers of Home Flood Protection

This repository contains the analysis code and data for the paper **"It's Not Just Risk—It's Responsibility: Changing Drivers of Home Flood Protection"** by Mikhail Sirenko and Tatiana Filatova.

Using two waves of a nationally representative household survey in the Netherlands (2020 and 2023), we examine how the drivers of household intentions to implement flood protection measures change over time. 

## Repository Structure

```
├── data/
│   ├── processed/
│   │   └── scalar/
│   │       ├── wave_1.csv              # Processed 2020 survey data
│   │       └── wave_5.csv              # Processed 2023 survey data
│   └── raw/
│       └── scalar/                     # Original SCALAR survey data (see Data section)
├── figures/                            # Generated figures
├── notebooks/
│   ├── 1-prepare-survey-data.ipynb     # Data cleaning and preparation
│   ├── 2-sample-overview.ipynb         # Sample composition analysis
│   ├── 3-descriptive-statistics.ipynb  # Descriptive analysis and visualizations
│   ├── 4-logistic-regression.ipynb     # Logistic regression models
│   ├── correlation-vif-analyses.ipynb  # Multicollinearity diagnostics
│   └── modelling.py                    # Helper functions for regression analysis
└── README.md
```

## Reproducing the Analysis

### Requirements

The analysis uses just a few basic Python packages:

- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Data preprocessing
- `statsmodels` - Logistic regression modeling, multiple tests and VIF calculation
- `matplotlib`, `seaborn`, `plot_likert` - Visualization

### Running the Notebooks

The analysis is organized into sequential notebooks. After getting access to  the data and downloading it (see **Data** section below), run notebooks in sequence:

1. **`1-prepare-survey-data.ipynb`** - Cleans and processes raw survey data from both waves
2. **`2-sample-overview.ipynb`** - Analyzes sample composition and representativeness (creates Fig 1)
3. **`3-descriptive-statistics.ipynb`** - Generates descriptive statistics and visualizations (creates Figs 2-4, Table 3)
4. **`4-logistic-regression.ipynb`** - Performs logistic regression analysis (creates Tables 4-5, S1-S12)

**Note**: `correlation-vif-analyses.ipynb` can be run after step 1 to examine multicollinearity diagnostics.

Each notebook is self-contained and saves outputs to the `figures/` directory.

## Data & Model

### SCALAR Survey Data

The analysis uses two nationally representative waves of the **SCALAR (Household Climate-Adaptation and Resilience Survey)** conducted in the Netherlands in 2020 and 2023:

- **Wave 2020**: N = 1,251
- **Wave 2023**: N = 420

**Original data source**: Available at [DANS Data Station Social Sciences and Humanities](https://doi.org/10.17026/dans-x9h-nj3w)

### Data Preparation

After downloading the raw SCALAR data:

1. Place the raw CSV files in `data/raw/scalar/`
2. Run `notebooks/1-prepare-survey-data.ipynb` to:
   - Clean and harmonize variables across waves
   - Encode ordinal and nominal variables
   - Create aggregated intention variables
   - Generate processed datasets in `data/processed/scalar/`

**Processed data outputs:**
- `wave_1.csv` - Cleaned 2020 survey data
- `wave_5.csv` - Cleaned 2023 survey data

## Statical Analyses

We use binary logistic regression (logit link, binomial family) with robust (HC0) covariance to model the intention to implement each structural measure. To account for multiple hypothesis testing, we utilise Benjamini-Hochberg False Discovery Rate (FDR) procedure.

Dependent variable: Intention to implement each measure (binary: 0 = no intention, 1 = planning to implement). "Already implemented" responses excluded from analysis.

Independent variables: After screening for multicollinearity (VIF > 5), final models include: Perceived flood frequency, Flood worry, Flood experience, Self-efficacy (measure-specific), Responsibility perception.

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{sirenko2025responsibility,
  title={It's Not Just Risk—It's Responsibility: Changing Drivers of Home Flood Protection},
  author={Sirenko, Mikhail and Filatova, Tatiana},
  journal={Under review},
  year={2025}
}
```

**Survey data citation**:
```bibtex
@dataset{filatova2022scalar,
  author = {Filatova, Tatiana and Noll, Brayton and Rijcken, Taneha and Wagenaar, Dennis},
  title = {SCALAR - Household climate-adaptation and resilience survey},
  year = {2022},
  publisher = {DANS Data Station Social Sciences and Humanities},
  doi = {10.17026/dans-x9h-nj3w},
  url = {https://doi.org/10.17026/dans-x9h-nj3w}
}
```

## Authors

* *Mikhail Sirenko* - [:octocat: github.com/miskh](https://github.com/miskh)
* *Tatiana Filatova* - [:briefcase: LinkedIn](https://www.linkedin.com/in/tatiana-filatova-a586163/)

## License

Please refer to the specific license terms associated with the SCALAR survey data at [DANS](https://doi.org/10.17026/dans-x9h-nj3w).

## Acknowledgments

This research is part of the Climate Risk Labels project funded by the Convergence Resilient Delta seed grant.