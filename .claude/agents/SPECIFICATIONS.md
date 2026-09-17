# Functional and Technical Specifications
## Territorial Favourability Analysis Tool for Protected Areas and OECMs

**Version**: 0.1 (working document)
**Status**: Preliminary specifications
**Context**: Decision-support tool for identifying territories candidate for Other Effective Area-based Conservation Measures (OECMs, CBD COP14 decision 14/8), designed for shared use with institutional partners in an international project context.

---

## 1. Context and Objectives

### 1.1 Regulatory Framework

OECMs are defined by the Convention on Biological Diversity (COP14, decision 14/8) as *"geographically defined areas other than protected areas, that are governed and managed in ways that achieve positive and sustained long-term outcomes for the in situ conservation of biodiversity, with associated ecosystem functions and services and where applicable, cultural, spiritual, socio-economic, and other locally relevant values"*.

Three definitional criteria structure the tool:

1. **Positive outcomes for biodiversity** — measurable and sustained
2. **Active governance and management** — not necessarily oriented towards conservation as a primary objective
3. **Production or use function** — coexistence with compatible human activities (distinguishing criterion with respect to classical protected areas)

The tool also aligns with the **Kunming-Montreal Global Biodiversity Framework (KMGBF)**, in particular Target 3 (30x30: 30% of land and seas under effective protection by 2030).

### 1.2 Tool Objectives

- **Primary objective**: Identify and map territories presenting the most favourable conditions for OECM designation, at the scale of an administrative region.
- **Secondary objective**: Produce a diagnostic of the existing protection network (PAs and recorded OECMs) to identify spatial and ecosystemic gaps.
- **Tertiary objective**: Provide an interactive interface allowing users to vary criteria weights and thresholds, and visualise their impact on the favourability map.

### 1.3 Target Users

| Profile | Main needs |
|---|---|
| Ecology researchers | Fine parameterisation, data export, reproducibility |
| Government agency officers | Quick visualisation, exportable reports, criteria transparency |
| Territorial authorities | Simple interface, map readability, statistics by territory |
| Nature reserve managers | Identification of their sites as OECM candidates |

---

## 2. General Architecture

The tool comprises two sequential functional modules, accessible via an interactive web interface.

```
┌─────────────────────────────────────────────────────────────┐
│  MODULE 1 — Existing protection network diagnostic          │
│  Sources: WDPA, (INPN/Carmen optional)                      │
│  Outputs: statistics, gap analysis, spatial masks           │
└───────────────────────────┬─────────────────────────────────┘
                            │ gap mask + priority weighting
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 2 — OECM favourability analysis                     │
│  Inputs: raster layers + Module 1 masks                     │
│  Method: MCE with eliminatory criteria                      │
│  Outputs: favourability map, GeoTIFF export                 │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 User Interface

**Selected option: Streamlit web application**

Justification relative to alternatives:

| Option | Advantages | Disadvantages | Suitability |
|---|---|---|---|
| **Streamlit** ✓ | Rapid development, easy deployment, accessible without GIS software | Limited performance on very large rasters | **Recommended** |
| QGIS plugin | Native GIS integration, direct layer access | Requires QGIS, less accessible to non-GIS users | Internal use only |
| Google Earth Engine App | Computational power, global datasets | GEE dependency, JavaScript, restricted access | Alternative if data are in GEE |
| Jupyter + Voilà | Flexible, well-suited for prototyping | Less robust for institutional deployment | Prototyping only |

The interface is organised into **two main tabs** corresponding to the two modules, with a lateral parameter panel.

---

## 3. Module 1 — Existing Protection Network Diagnostic

### 3.1 Data Sources

**Primary source: WDPA (World Database on Protected Areas)**
- Access: Protected Planet API (`requests` + `geopandas`) or monthly download (shapefile/GDB)
- Fields used: `IUCN_CAT`, `DESIG_TYPE`, `DESIG`, `STATUS`, `GIS_AREA`, `PARENT_ISO`
- Known limitation: incomplete coverage for certain national designation categories depending on the country — to be explicitly documented in the interface

**Optional complementary source: WD-OECM**
- OECMs already recorded globally
- Very limited data available for most countries to date

**Extensible architecture**: the module is designed to accept additional vector layers (national databases, regional data) via a configuration file, without modification of the main code.

### 3.2 Legal Classification

The WDPA is recoded into classes interpretable by institutional partners. The correspondence table is externalised in a modifiable YAML configuration file:

| Displayed class | WDPA / designation correspondence | Map colour |
|---|---|---|
| Strict protection — core zones | IUCN Ia, Ib, II | Dark green |
| Regulatory protection | IUCN III, IV | Medium green |
| Contractual management / Natura 2000 | IUCN V, VI / SAC, SPA | Light green |
| Unassigned status | `IUCN_CAT = "Not Reported"` | Grey |
| Recorded OECMs | WD-OECM source | Blue |

### 3.3 Coverage Statistics Block

Indicators calculated automatically for the study territory (extent defined by the user):

**By legal category:**
- Total area (ha) and percentage of territory
- Number of sites / polygons
- Mean and median area per site
- Fragmentation index (number of patches / total protected area)

**Aggregated:**
- Percentage of territory under strict protection (IUCN I–II) — KMGBF 30x30 indicator
- Percentage under Natura 2000 or equivalent
- **Net deduplicated area**: calculated by geometric union before aggregation (avoids overestimation due to overlaps between categories)

> **Methodological note**: The net deduplicated area is the indicator to be used for any official reporting. A simple sum of areas by category systematically overestimates actual coverage due to overlaps.

### 3.4 Ecosystemic Representativity Block

Spatial cross-analysis between PA layers and an ecosystem type layer (CLC level 3, EUNIS, or regional land cover map).

**Indicators per ecosystem type:**
- Total area within the territory
- Area under protection (all categories / strict protection only)
- Percentage of representation
- Comparison against two configurable reference thresholds: KMGBF threshold (30%) and an optional user-defined target threshold

**Synthetic representativity index (RI):**

```
RI = Σ [ min(coverage_e / target_threshold_e , 1) ] / N_types

RI = 1.0 : all ecosystem types reach their target threshold
RI < 1.0 : under-representation — identifies priority types for new PAs/OECMs
```

**Visualisation**: horizontal bar chart by ecosystem type, with a vertical line at the target threshold.

### 3.5 Initial Weighting Proposal for Module 2

Module 1 can provide a **default weight proposal** for Group A criteria (ecological integrity), based on the representativity deficit of ecosystem types in the existing protection network:

```
For each criterion i associated with ecosystem type e:
  deficit_e = max(0, target_threshold_e − current_coverage_e)
  w_i_proposed = deficit_e / Σ deficit_e    [normalised, Σ = 1]
```

**Interpretation**: ecological criteria corresponding to the most under-represented types in existing PAs receive a higher weight — gap-filling conservation logic.

These values are proposed as a starting point in the Module 2 interface, with explicit indication of their origin (*"weights suggested from Module 1 diagnostic"*). The user remains free to modify them.

**Explicit limits of this approach:**
- Module 1 informs relative weights *within Group A* only
- It cannot inform *inter-group* weights (A vs B vs C), which involve a value judgement on the relative importance of ecological integrity, co-benefits, and uses
- The correspondence between raster criteria and ecosystem types must be defined manually in the configuration file

### 3.6 Spatial Gap Analysis Block

Production of three vector layers transmitted to Module 2:

| Layer | Definition | Use in Module 2 |
|---|---|---|
| Strict gaps | Areas with no PA coverage | Optional positive weighting |
| Qualitative gaps | Areas covered only by weak protection (IUCN V–VI) | Candidates for OECM complementarity |
| Potential corridors | Unprotected areas connecting existing PAs (simplified connectivity analysis) | Configurable priority bonus |

**Connectivity analysis options:**

| Option | Method | Complexity | Relevance |
|---|---|---|---|
| **Simplified** ✓ (default) | Buffer + Euclidean distance between PA patches | Low | Sufficient for initial identification |
| Intermediate | Cost-distance on land cover resistance | Medium | Recommended if resistance layer is available |
| Full | Graphab or Circuitscape (external) | High | For dedicated connectivity analysis |

---

## 4. Module 2 — OECM Favourability Analysis

### 4.1 Input Data

All raster layers must be provided in a common CRS (recommended: ETRS89 / EPSG:3035 for Europe; adaptable for other regions), at a user-defined reference resolution. The module automatically reprojects and resamples non-conforming layers.

**Required layers:**

| Layer | Type | Role in analysis |
|---|---|---|
| Ecosystem condition | Continuous [0–1] | Co-structuring criterion — Group A |
| Anthropogenic pressure | Raw continuous (population density or road density) | Dual: eliminatory (Group D) if > threshold; Group A criterion if ≤ threshold |
| Land use / production | Categorical (CLC, OSO, or equivalent) | Dual: eliminatory (Group D) for incompatible classes; Group C criterion for compatible classes |
| Provisioning ES capacity | Continuous [0–1] | Co-structuring criterion — Group C (Gaussian function) |
| Regulating ES capacity | Continuous [0–1] | Co-structuring criterion — Group A |
| Cultural ES capacity | Continuous [0–1] | Co-structuring criterion — Group B |

No optional layers are provided in this version. Addition of further layers is possible via the configuration file without modification of the main code.

### 4.2 Layer Normalisation

Each raster layer is transformed into a [0–1] score via a transformation function adapted to its nature:

| Layer | Function | Justification |
|---|---|---|
| Ecosystem condition | Linear or sigmoid | Monotone increasing relationship with favourability |
| Anthropogenic pressure (score part) | Inverted linear | Low pressure = favourable; monotone decreasing relationship |
| Regulating ES capacity | Linear or sigmoid | Monotone increasing relationship |
| Cultural ES capacity | Linear or sigmoid | Monotone increasing relationship |
| Provisioning ES capacity | **Gaussian** | Intermediate optimum: zero use (absence of production) and very high use (overexploitation) are both unfavourable for OECM eligibility |
| Land use — compatible classes (score part) | Ordinal recoding | Configurable compatibility table: CLC/OSO classes → score [0–1] |

Transformation function parameters are configurable in the interface (advanced panel) and exported in the results report to ensure reproducibility.

**Provisioning ES calibration.** The Gaussian's default (mean=0.45, std=0.35)
is a universal fallback, not a territory-specific claim — a narrower default
(std=0.20) was found to systematically under-score any territory whose real
provisioning-capacity distribution did not sit near 0.45, including both
near-pristine (low-extraction) and intensively-used candidate profiles. The
interface offers a second, explicit **auto-calibrate** mode that centres the
Gaussian on this territory's own observed provisioning_es mean/std instead of
the fixed default. Auto-calibration is *off* by default: it answers a
different question (is this pixel typical for its own territory?) than the
fixed default (does this pixel have absolutely moderate, sustainable-use
character?), and the latter is what the Group C use-presence threshold /
classical_pa_mask discrimination (§4.3) assumes — a territory that is
uniformly poor OECM material should score poorly here, not be auto-normalised
to "optimal". Auto-calibration is intended for deliberate use (e.g. ranking
candidate micro-sites within one already-selected territory), not as a
general remedy for low favourability scores.

### 4.3 Criteria Structure and Functional Groups

Criteria are organised into four functional groups. The land use and anthropogenic pressure layers play a **dual role**: their extreme values feed the eliminatory criteria (Group D); their intermediate values feed the co-structuring criteria (Groups A and C).

#### Group D — Strict eliminatory criteria

Applied as a priority before any score calculation. Produces a binary mask of ineligible zones.

| Criterion | Source | Default threshold | Configurable |
|---|---|---|---|
| Incompatible land use | Categorical layer (CLC/OSO) | Urban classes (CLC 1.x), industrial (CLC 1.2–1.4), intensive irrigated crops (CLC 2.1 depending on context) | Via YAML compatibility table |
| Excessive anthropogenic pressure | Raw layer (population or road density) | To be defined per territory | Slider in interface |

The CLC/OSO → eliminatory/compatible compatibility table is externalised in a modifiable YAML file, enabling territorial adaptation without code modification.

#### Group A — Ecological integrity

Co-structuring criteria expressing the intrinsic conservation value of the territory.

| Criterion | Source | Transformation function |
|---|---|---|
| Ecosystem condition | Required layer [0–1] | Linear or sigmoid (configurable) |
| Regulating ES capacity | Required layer [0–1] | Linear or sigmoid (configurable) |
| Low anthropogenic pressure | Raw layer (values ≤ threshold_max) | Inverted linear → score [0–1] |

#### Group B — Co-benefits and social compatibility

Co-structuring criterion reinforcing the socio-political justification and governance durability of the OECM.

| Criterion | Source | Transformation function |
|---|---|---|
| Cultural ES capacity | Required layer [0–1] | Linear or sigmoid (configurable) |

> **Note**: Group B contains a single criterion in this version. Its inter-group weight (W_B) can be set to zero by the user if they wish to disregard it, without altering the model structure.

#### Group C — Production and compatible use function

Co-structuring criteria expressing the use function required by the OECM definition. The complete absence of a production function disqualifies a zone as an OECM candidate (flag *"Classical PA preferable"*).

| Criterion | Source | Transformation function |
|---|---|---|
| Provisioning ES capacity | Required layer [0–1] | **Gaussian** — intermediate optimum |
| Compatible land use | Categorical layer (compatible classes) | Configurable ordinal recoding |

**Use presence threshold (inverted eliminatory)**: if the aggregated Group C score is below a configurable minimum threshold (default: 0.1), the pixel receives the flag *"Classical PA preferable"* and is excluded from the OECM score, but retained in a dedicated output layer.

### 4.4 Conceptual Foundation: Co-structuring Criteria and Non-compensability

The criteria mobilised in this tool are **co-structuring**: each constitutes a necessary condition for OECM favourability, and no high score on one criterion can compensate for a low score on another. This position is consistent with the CBD definition of OECMs, which requires the conjunction of conservation outcomes, active governance, and a compatible use function.

Accordingly, the **weighted linear combination (WLC) is rejected** as the main aggregation method, as it assumes total compensability between criteria.

**Role of weights**: weights are not parameters calibrated on empirical data, but **preference vectors** expressing the priorities of the decision-maker or experimenter. Different weight configurations produce different favourability maps from the same data — this is the primary function of the interactive interface.

### 4.5 Aggregation Method: OWA and Weighted Geometric Mean

Two formulations are available, selectable in the interface:

#### Option A — Weighted geometric mean (recommended default)

```
S = Π (criterion_i ^ w_i)     with Σ w_i = 1
```

Properties:
- Strictly non-compensatory: a zero criterion nullifies the total score
- Weights retain their interpretation as relative importance
- Intuitive for decision-makers: a very low criterion strongly degrades the final score
- Limiting case: if all weights are equal → standard geometric mean

#### Option B — Yager OWA with orness parameter

```
OWA score = Σ (v_j × b_j)

where:
  b_j  = j-th criterion value sorted in descending order
  v_j  = OWA weights calculated according to orness parameter α ∈ [0, 1]
  α → 0 : AND logic (minimum) — no compensation
  α = 0.5 : balanced partial compensation
  α → 1 : OR logic (maximum) — total compensation
```

The α parameter is exposed in the interface as a **tolerance-to-non-satisfaction slider**, with an interpretive legend:

```
[Strict AND]  ←————————————————→  [Permissive OR]
     0        0.25      0.5      0.75        1
  "All criteria          "Balance"        "One criterion
   required"                               sufficient"
```

> **Note**: criterion weights (relative importance) and the α parameter (compensation tolerance) are two orthogonal dimensions of the parameterisation. The interface presents them separately to avoid confusion.

#### Full aggregation workflow

```
Step 0 : Apply Group D mask → eligible pixels
Step 1 : Verify Group C presence → flag "OECM" or "Classical PA preferable"
Step 2 : Favourability score on eligible pixels

  Intra-group (A, B, C):
    score_G = OWA or geometric aggregation of group criteria
              with user-defined weights w_i

  Inter-group:
    S = OWA or geometric aggregation (score_A, score_B, score_C)
        with user-defined weights W_A, W_B, W_C
        W_A + W_B + W_C = 1  [constraint verified by interface]

Step 3 : Gap analysis modulation (optional)
  S_final = S × (1 + gap_bonus)
  gap_bonus ∈ [0, 0.2] — configurable
```

### 4.6 Outputs and Visualisation

**Favourability map (main display):**
- Score raster [0–1] displayed as a colour gradient
- Overlay of existing PAs (Module 1)
- Visual distinction: *"OECM favourable"* / *"Classical PA preferable"* / *"Not favourable"* zones

**Territorial statistics:**
- Score distribution (histogram)
- Area by favourability class (configurable: terciles, quartiles, manual thresholds)
- Statistics by administrative unit (municipality, intercommunality, grid cell)

**Available exports:**

| Format | Content |
|---|---|
| GeoTIFF | Favourability score raster |
| Shapefile / GeoJSON | Favourable zone polygons (configurable threshold) |
| CSV | Statistics by territorial unit |
| PDF | Automated report (map + statistics + parameters used) |

---

## 5. Technical Stack

### 5.1 Main Python Dependencies

| Component | Library | Minimum recommended version |
|---|---|---|
| Interface | `streamlit` | ≥ 1.32 |
| Raster processing | `rasterio`, `numpy`, `xarray` | — |
| Reprojection | `pyproj`, `rasterio.warp` | — |
| Vector analysis | `geopandas`, `shapely` | — |
| Cartographic visualisation | `folium`, `streamlit-folium` | — |
| WDPA access | `requests` + JSON/GDB parsing | — |
| PDF export | `reportlab` or `weasyprint` | — |
| Configuration | `pyyaml` | — |

### 5.2 Deployment Options

| Option | Context | Advantages | Constraints |
|---|---|---|---|
| **Streamlit Community Cloud** | Demonstration, public sharing | Free, deployment in minutes | Limited resources, public data only |
| **Institutional server (Docker)** | Partners, sensitive data | Full control, local data | Requires server infrastructure |
| **Local execution** | Researchers | No network dependency, maximum performance | Python installation required |
| **Hugging Face Spaces** | Cloud alternative | Free, good performance | Less known by institutions |

**Recommendation**: local development, Docker deployment for institutional partners, with a Streamlit Community Cloud instance for public demonstrations.

### 5.3 Project File Structure

```
oecm-favourability-tool/
│
├── app.py                              # Streamlit entry point
├── requirements.txt
├── Dockerfile
├── README.md
├── SPECIFICATIONS.md
├── WINDSURF_RULES.md
│
├── config/
│   ├── settings.yaml                  # CRS, resolution, paths, logging
│   ├── iucn_classification.yaml       # WDPA → displayed classes
│   ├── criteria_defaults.yaml         # Default weights and thresholds
│   ├── transformation_functions.yaml  # Transformation function parameters
│   └── land_use_compatibility.yaml    # CLC/OSO → eliminatory/compatible/score
│
├── modules/
│   ├── module1_protected_areas/
│   │   ├── wdpa_loader.py
│   │   ├── coverage_stats.py
│   │   ├── representativity.py
│   │   └── gap_analysis.py
│   │
│   └── module2_favourability/
│       ├── raster_preprocessing.py
│       ├── criteria_manager.py
│       ├── mce_engine.py
│       └── export.py
│
├── ui/
│   ├── sidebar.py
│   ├── tab_module1.py
│   └── tab_module2.py
│
├── data/
│   └── [user raster layers — not versioned]
│
└── tests/
    ├── generate_synthetic_data.py
    ├── test_raster_preprocessing.py
    ├── test_mce_engine.py
    └── test_representativity.py
```

---

## 6. Development Plan

| Phase | Content | Estimated duration |
|---|---|---|
| **Phase 1** — Module 1 prototype | WDPA loading, classification, coverage statistics, basic map | 2 weeks |
| **Phase 2** — Module 2 prototype | Raster loading, normalisation, basic geometric mean, score visualisation | 2 weeks |
| **Phase 3** — Full MCE | Eliminatory criteria, groups A/B/C/D, PA/OECM flags, M1→M2 connection | 2 weeks |
| **Phase 4** — Full interface | Sliders, weight sum = 1 constraint, ecosystemic representativity, gap analysis | 2 weeks |
| **Phase 5** — Exports and reports | GeoTIFF, automated PDF, statistics by territorial unit | 1 week |
| **Phase 6** — Validation and deployment | Partner testing, UX adjustments, Docker, user documentation | 2 weeks |

**Total estimated duration: 11 weeks** (part-time development)

---

## 7. Limitations and Points of Attention

- **WDPA coverage**: incomplete for certain national designation categories. To be explicitly documented in the interface and supplemented with national sources where available.
- **Raster layer availability**: result quality is directly conditioned by the availability and resolution of input layers. The tool does not produce ecological data — it aggregates existing data.
- **MCE method validity**: the weighted geometric mean is strictly non-compensatory, but requires that no input layer contains zero values across large areas (which would systematically nullify scores). NoData and zero values must be carefully handled during preprocessing.
- **Non-substitutability of field expertise**: tool outputs constitute decision support and not automatic designation. OECM status confirmation requires an assessment of governance and management practices not capturable by raster data.
- **Temporal dimension**: the tool operates on static data (snapshot). The OECM definition requires *sustained* outcomes — a diachronic analysis (ecosystem condition change over time) would be a relevant future development.
- **Transferability**: the CLC/OSO compatibility table and eliminatory thresholds must be adapted to each study territory. Default values provided in the configuration files are indicative only.

---

*Document produced as part of the development of the oecm-favourability-tool — Version 0.1*