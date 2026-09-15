"""OECM Favourability Tool — Streamlit entry point."""
import streamlit as st
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ===================================================================
# Page configuration
# ===================================================================
st.set_page_config(
    page_title="OECM Favourability Tool",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Tab styling: bold labels, 2 rows of 2 (flex-wrap)
st.markdown(
    """
    <style>
    /* Wrap tabs onto 2 rows of 2 */
    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: wrap;
        gap: 8px 12px;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 0 0 calc(50% - 8px);
        box-sizing: border-box;
        font-size: 2.1rem;
        font-weight: 700;
        padding: 16px 24px;
        justify-content: flex-start;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 2.1rem !important;
        font-weight: 700 !important;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #2E7D32;
        color: #2E7D32;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===================================================================
# Import UI components
# ===================================================================
from ui import tab_data_upload
from ui.tab_parameters import render_parameters_tab
from ui.tab_ahp import render_tab_ahp
from ui.tab_module1 import render_tab_module1
from ui import tab_module2

# ===================================================================
# Main title and description
# ===================================================================
st.title("OECM Conservation Planning Tool")

st.markdown(
    """
    GIS decision-support tool for identifying and assessing candidate territories for
    **Other Effective Area-based Conservation Measures (OECMs)**,
    aligned with **KMGBF Target 3** (CBD COP15 decision 15/4) and the global **30×30** biodiversity commitment.

    | Step | Module | Description |
    |---|---|---|
    | ① | **Data Upload** | Load WDPA protected areas, NUTS study-area boundaries, and MCE criterion rasters |
    | ② | **Parameters** | Study area, thresholds, normalisation, aggregation method, weights, bonuses |
    | ③ | **Weight Calibration (AHP)** | Set criterion importance using Analytic Hierarchy Process pairwise comparisons |
    | ④ | **Protection Network Diagnostic** | WDPA coverage statistics, KMGBF indicator, ecosystem representativity, gap analysis |
    | ⑤ | **OECM Favourability Analysis** | Multi-criteria evaluation, candidate site delineation, sensitivity analysis, and GeoTIFF / DOCX export |
    """
)

st.markdown("---")

# ===================================================================
# Tabs: Data Upload, Parameters, AHP, Module 1, and Module 2
# ===================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① Data Upload",
    "② Parameters",
    "③ Weight Calibration (AHP)",
    "④ Protection Network Diagnostic",
    "⑤ OECM Favourability Analysis"
])

with tab1:
    tab_data_upload.render()

with tab2:
    parameters = render_parameters_tab()

    # Store parameters in session state for access across tabs
    st.session_state['parameters'] = parameters

    # Store study area geometry separately for Module 1 compatibility
    if 'study_area_geometry' in parameters:
        st.session_state['territory_geom'] = parameters['study_area_geometry']

    # Log current parameters (DEBUG level)
    logger.debug(f"Current parameters: {parameters}")

with tab3:
    render_tab_ahp()

with tab4:
    # Retrieve PA data from session state if available
    pa_gdf = st.session_state.get('pa_gdf', None)
    territory_geom = st.session_state.get('territory_geom', None)
    ecosystem_layer = st.session_state.get('ecosystem_layer', None)

    # Render Module 1 tab
    render_tab_module1(
        pa_gdf=pa_gdf,
        territory_geom=territory_geom,
        ecosystem_layer=ecosystem_layer
    )

with tab5:
    tab_module2.render_module2_tab()

# ===================================================================
# Footer
# ===================================================================
st.markdown("---")
st.caption(
    "OECM Favourability Tool v0.1 | "
    "Developed with Claude Code | "
    "Full specifications: .claude/agents/SPECIFICATIONS.md"
)
