# MODIS Pixel Analysis

Advanced time series analysis and statistical modeling of MODIS albedo data for master's thesis research.

## Project Overview

This project conducts comprehensive statistical analysis of MODIS (Moderate Resolution Imaging Spectroradiometer) albedo data spanning 2010-2024. The analysis includes time series decomposition, trend analysis, change point detection, and spectral analysis to understand long-term patterns in surface albedo measurements.

## Project Structure

```
MODIS Pixel analysis/
├── src/                          # Source code
│   ├── analysis/                 # Core analysis scripts
│   │   ├── advanced_time_series_analysis.py    # Advanced time series methods
│   │   ├── comprehensive_statistical_analysis.py # Statistical modeling
│   │   ├── modis_trend_analysis.py             # Trend detection and analysis
│   │   └── modis_visualization.py              # Data visualization tools
│   └── reports/                  # Report generation scripts
│       ├── create_comprehensive_document.py    # Main document generator
│       ├── create_statistical_report.py       # Statistical report builder
│       ├── create_word_document.py            # Word document creator
│       └── execute_word_creation.py           # Execution wrapper
├── data/                         # Data files
│   ├── raw/                      # Original MODIS data
│   │   └── MOD10A1_albedo_pixel_level_full_2010_2024.csv
│   └── processed/                # Processed/derived datasets
│       ├── annual_stats.csv                   # Annual statistics
│       ├── modis_annual_stats_analysis.csv   # Analysis results
│       └── modis_trend_results.csv           # Trend analysis outputs
├── results/                      # Analysis outputs
│   ├── plots/                    # Basic visualizations
│   ├── advanced_plots/           # Advanced time series plots
│   └── statistical_analysis/     # Statistical analysis results
├── docs/                         # Documentation
│   ├── reports/                  # Generated analysis reports
│   └── LaTeX_Document_Summary.md # Documentation summary
├── config/                       # Configuration files
│   └── word_document_plan.txt    # Document generation configuration
└── README.md                     # This file
```

## Features

### Time Series Analysis
- **Seasonal Decomposition**: STL (Seasonal and Trend decomposition using Loess)
- **Autocorrelation Analysis**: Identification of temporal dependencies
- **Spectral Analysis**: Frequency domain analysis using FFT
- **Change Point Detection**: Statistical detection of regime changes
- **Trend Analysis**: Linear and non-linear trend modeling

### Statistical Methods
- **Descriptive Statistics**: Comprehensive summary statistics
- **Distribution Analysis**: Normality testing and distribution fitting
- **Outlier Detection**: Multiple outlier identification methods
- **Correlation Analysis**: Temporal and spatial correlation assessment
- **Quality Assessment**: Data quality flags and filtering

### Visualization
- **Time Series Plots**: Comprehensive temporal visualizations
- **Distribution Plots**: Histograms, box plots, and density plots
- **Correlation Matrices**: Heatmaps and correlation networks
- **Trend Diagnostics**: Residual analysis and model validation
- **Spectral Plots**: Power spectral density and periodograms

## Data Description

The project analyzes MODIS MOD10A1 albedo data at pixel level covering:
- **Temporal Coverage**: 2010-2024 (15 years)
- **Spatial Resolution**: 500m pixel level
- **Variables**: Surface albedo measurements with quality flags
- **Processing**: Annual aggregation and statistical summaries

## Requirements

### Python Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- subprocess
- os
- warnings

### External Dependencies
- DuckDB (for data storage and querying)
- LaTeX (for document generation)

## Usage

### Running Analysis Scripts

1. **Time Series Analysis**:
   ```bash
   python src/analysis/advanced_time_series_analysis.py
   ```

2. **Statistical Analysis**:
   ```bash
   python src/analysis/comprehensive_statistical_analysis.py
   ```

3. **Trend Analysis**:
   ```bash
   python src/analysis/modis_trend_analysis.py
   ```

4. **Visualization**:
   ```bash
   python src/analysis/modis_visualization.py
   ```

### Generating Reports

1. **Comprehensive Document**:
   ```bash
   python src/reports/create_comprehensive_document.py
   ```

2. **Statistical Report**:
   ```bash
   python src/reports/create_statistical_report.py
   ```

3. **Word Document**:
   ```bash
   python src/reports/execute_word_creation.py
   ```

## Output Files

### Visualizations
- `results/plots/`: Basic trend and distribution plots
- `results/advanced_plots/`: Advanced time series visualizations
- `results/statistical_analysis/`: Statistical analysis outputs

### Reports
- `docs/reports/`: Generated analysis reports in multiple formats
  - Markdown reports
  - LaTeX documents
  - Word documents

### Data Products
- `data/processed/`: Derived datasets and analysis results

## Key Findings

The analysis reveals:
- Long-term trends in surface albedo over the 15-year period
- Seasonal patterns and their variability
- Change points indicating significant shifts in albedo patterns
- Spectral characteristics of the time series
- Quality assessment of MODIS data products

## Contributing

This project is part of master's thesis research. For questions or collaboration opportunities, please contact the project maintainer.

## License

This project is for academic research purposes. Please cite appropriately if using any methods or findings.

## References

- MODIS/Terra Snow Cover Daily L3 Global 500m SIN Grid, Version 6
- Seasonal and Trend decomposition using Loess (STL)
- Statistical methods for time series analysis

---

*Last updated: 2025-06-24*