# OECM Favourability Tool

A GIS decision-support tool for identifying and assessing candidate territories for Other Effective Area-based Conservation Measures (OECMs), aligned with KMGBF Target 3 (CBD COP15 decision 15/4) and the global 30×30 biodiversity commitment. Developed for the PAREUS project (Biodiversa programme).

## Overview

The tool is organised into five steps:

| Step | Module | Description |
|---|---|---|
| ① | **Data Upload** | Load WDPA protected areas, NUTS study-area boundaries, and MCE criterion rasters |
| ② | **Parameters** | Study area, thresholds, normalisation, aggregation method, weights, bonuses |
| ③ | **Weight Calibration (AHP)** | Set criterion importance using Analytic Hierarchy Process pairwise comparisons |
| ④ | **Protection Network Diagnostic** | WDPA coverage statistics, KMGBF indicator, ecosystem representativity, gap analysis |
| ⑤ | **OECM Favourability Analysis** | Multi-criteria evaluation, candidate site delineation, sensitivity analysis, and GeoTIFF / DOCX export |

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for the input data each step requires (also available in-app, on the main page below the module table).

## Setup Instructions

### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PkdRoche/oecm-favourability-tool.git
   cd oecm-favourability-tool
   ```

2. Create a Python virtual environment (Python 3.11 recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Review and adapt the YAML files in `config/` — criteria defaults, transformation functions, IUCN and land-use classification, general settings.

5. Run the application:
   - **Windows**: `.\startOECMTool.ps1` — also handles two common local-network issues automatically: an optional corporate SSL certificate override (see the script's own header for details) and a PROJ/PostGIS conflict that otherwise breaks CRS operations.
   - **Any platform**: `streamlit run app.py`

6. Open `http://localhost:8501` in your browser.

## Project Structure

```
oecm-favourability-tool/
├── app.py                          # Streamlit entry point
├── startOECMTool.ps1                # Windows launcher (recommended)
├── requirements.txt                 # Python dependencies
├── INSTRUCTIONS.md                  # Input data requirements
├── config/                          # YAML configuration files
│   ├── settings.yaml
│   ├── iucn_classification.yaml
│   ├── criteria_defaults.yaml
│   ├── transformation_functions.yaml
│   └── land_use_compatibility.yaml
├── modules/                         # Core analytical modules
│   ├── module1_protected_areas/     # Protection Network Diagnostic
│   ├── module2_favourability/       # OECM Favourability Analysis
│   └── utils/
├── ui/                               # Streamlit UI components (one per tab)
├── data/                             # Input data directory (not versioned)
├── outputs/                          # Generated outputs (not versioned)
└── tests/                            # Unit and integration tests
```

Analysis settings (weights, thresholds, aggregation method, file paths, etc.) can be saved to and reloaded from a `.ini` project file directly in the app, so a session doesn't need to be reconfigured from scratch each time.

## License

[Specify license]

## Contact

[Specify contact information]
