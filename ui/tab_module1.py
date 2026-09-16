"""Streamlit UI for Module 1 — Protection Network Diagnostic."""
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from pathlib import Path
import yaml
import tempfile
import logging

from ui.export_utils import save_and_download

logger = logging.getLogger(__name__)


def load_iucn_classification():
    """
    Load IUCN classification colour scheme from config.

    Returns
    -------
    dict
        Dictionary mapping protection classes to colour codes and labels.
    """
    config_path = Path(__file__).parent.parent / "config" / "iucn_classification.yaml"

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('classes', {})
    except FileNotFoundError:
        st.warning(f"IUCN classification config not found: {config_path}")
        return {}


@st.fragment
def render_tab_module1(pa_gdf=None, territory_geom=None, ecosystem_layer=None):
    """
    Render Module 1 interface with WDPA loading, coverage stats, and gap analysis.

    Decorated with @st.fragment so interactions inside this tab (Run Gap
    Analysis, Compute Ecosystem RI, etc.) do not trigger a full app rerun —
    and therefore do not re-trigger Module 2's MCE recomputation. All
    mutable state is re-read from st.session_state below rather than relying
    on the pa_gdf/territory_geom/ecosystem_layer arguments, since those are
    only refreshed on a full rerun (not on a fragment-scoped rerun).

    Parameters
    ----------
    pa_gdf : gpd.GeoDataFrame, optional
        Protected areas GeoDataFrame with 'protection_class' column.
        If None, displays placeholder UI.
    territory_geom : shapely.geometry, optional
        Territory boundary geometry for statistics.
    ecosystem_layer : gpd.GeoDataFrame, optional
        Ecosystem types layer for representativity analysis.
    """
    st.header("④ Protection Network Diagnostic")

    # ===================================================================
    # Load WDPA and run analysis if not already done
    # ===================================================================
    wdpa_file = st.session_state.get('wdpa_file')
    territory_geom = st.session_state.get('territory_geom')
    pa_gdf = st.session_state.get('pa_gdf')

    if territory_geom is None:
        st.info("Select your study region in **② Parameters** (Country → NUTS2 region) first.")
        return

    if wdpa_file is None:
        st.info("Upload your WDPA protected areas file in the **① Data Upload** tab first.")
        return

    if pa_gdf is None:
        st.info("WDPA file loaded. Click the button below to run the diagnostic.")
        if st.button("▶ Run Protection Network Diagnostic", type="primary", width='stretch'):
            status = st.empty()
            status.info("Loading and classifying protected areas…")
            try:
                from modules.module1_protected_areas.wdpa_loader import (
                    load_wdpa_local, filter_to_extent, classify_iucn
                )
                import yaml
                config_path = Path(__file__).parent.parent / "config" / "iucn_classification.yaml"
                with open(config_path, 'r', encoding='utf-8') as f:
                    classification_table = yaml.safe_load(f)
                gdf = load_wdpa_local(wdpa_file)
                if st.session_state.get('exclude_marine_pa', True):
                    if 'REALM' in gdf.columns:
                        n_marine = int((gdf['REALM'] == 'Marine').sum())
                        gdf = gdf[gdf['REALM'] != 'Marine'].copy()
                        if n_marine:
                            st.info(f"Excluded {n_marine} fully marine PA(s) (REALM = 'Marine').")
                    elif 'MARINE' in gdf.columns:
                        # Fallback for older WDPA exports using integer MARINE column
                        n_marine = int((gdf['MARINE'] == 2).sum())
                        gdf = gdf[gdf['MARINE'] != 2].copy()
                        if n_marine:
                            st.info(f"Excluded {n_marine} fully marine PA(s) (MARINE = 2).")
                    else:
                        st.caption("Neither REALM nor MARINE column found in WDPA file — marine filter not applied.")
                gdf = filter_to_extent(gdf, territory_geom)
                gdf = classify_iucn(gdf, classification_table)
                st.session_state['pa_gdf'] = gdf
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load WDPA data: {e}")
                logger.exception("WDPA load failed")
            finally:
                status.empty()
        return

    pa_gdf = st.session_state['pa_gdf']

    # Button to re-run with fresh data
    if st.button("↺ Re-run Diagnostic", width='content'):
        st.session_state.pop('pa_gdf', None)
        st.rerun()

    # Import module functions (only when data is available)
    from modules.module1_protected_areas.coverage_stats import (
        compute_net_area,
        coverage_by_class,
        kmgbf_indicator
    )
    from modules.module1_protected_areas.zonal_stats import iucn_category_sort_key
    from modules.module1_protected_areas.representativity import (
        propose_group_a_weights
    )
    from modules.module1_protected_areas.gap_analysis import (
        strict_gaps,
        qualitative_gaps,
        potential_corridors
    )

    # Ensure data is in EPSG:3035
    if pa_gdf.crs != 'EPSG:3035':
        pa_gdf = pa_gdf.to_crs('EPSG:3035')

    # Compute territory area
    territory_area_ha = territory_geom.area / 10000.0

    # ===================================================================
    # Row 1: Metric cards
    # ===================================================================
    st.subheader("Key Indicators")

    col1, col2, col3, col4 = st.columns(4)

    # Metric 1: Net protected area (all classes, deduplicated)
    net_area = compute_net_area(pa_gdf, territory_geom)

    with col1:
        st.metric(
            label="Net Protected Area",
            value=f"{net_area:,.0f} ha",
            help="Deduplicated area via geometric union (excludes overlaps) — all PA classes"
        )

    # Metric 2: Full KMGBF Target 3 indicator (IUCN I–VI + OECMs)
    # Per CBD COP15 decision 15/4: all IUCN categories I–VI and OECMs count.
    kmgbf_pct = kmgbf_indicator(pa_gdf, territory_area_ha)          # I–VI + OECM
    strict_pct = kmgbf_indicator(pa_gdf, territory_area_ha,
                                 classes=['strict_core'])             # I–II only
    target_30 = 30.0
    strict_target = 10.0

    with col2:
        delta = kmgbf_pct - target_30
        st.metric(
            label="KMGBF T3 — All IUCN I–VI + OECMs",
            value=f"{kmgbf_pct:.1f}%",
            delta=f"{delta:+.1f}% vs 30% target",
            help=(
                "KMGBF Target 3 (CBD COP15 decision 15/4): percentage of territory "
                "covered by PAs or OECMs of any IUCN category (I–VI) or recorded OECM. "
                "Unassigned areas (no confirmed IUCN category) are excluded. "
                "30% by 2030 is the global commitment."
            )
        )

    with col3:
        st.metric(
            label="Strict protection only (IUCN I–II)",
            value=f"{strict_pct:.1f}%",
            delta=f"{strict_pct - strict_target:+.1f}% vs 10% target",
            help=(
                "Sub-indicator: area under the most restrictive management regimes "
                "(IUCN Ia strict nature reserves, Ib wilderness areas, II national parks). "
                "Target: 10% of territory. Many national frameworks and IPBES reports use "
                "this narrower metric alongside the full KMGBF indicator."
            )
        )

    # Metric 4: Number of PA sites
    n_sites = len(pa_gdf)

    with col4:
        st.metric(
            label="Number of PA Sites",
            value=f"{n_sites:,}",
            help="Total number of protected-area polygons in the network (before spatial deduplication)"
        )

    st.markdown("---")

    # ===================================================================
    # Row 2: Interactive folium map
    # ===================================================================
    st.subheader("Protected Area Network Map")

    # Load colour scheme
    iucn_classes = load_iucn_classification()

    # Reproject to EPSG:4326 for folium (needs lat/lon). Cached in session
    # state keyed on pa_gdf's identity — pa_gdf keeps the same identity
    # across fragment reruns once it's already in EPSG:3035 (see the CRS
    # guard above), so this avoids re-reprojecting on every interaction
    # inside this tab (Run Gap Analysis, Compute Ecosystem RI, etc.).
    _pa_cache_key = id(pa_gdf)
    _pa_cache = st.session_state.get('_pa_gdf_4326_cache')
    if _pa_cache is not None and _pa_cache.get('key') == _pa_cache_key:
        pa_gdf_4326 = _pa_cache['gdf']
    else:
        pa_gdf_4326 = pa_gdf.to_crs('EPSG:4326')
        st.session_state['_pa_gdf_4326_cache'] = {'key': _pa_cache_key, 'gdf': pa_gdf_4326}
    territory_gs = gpd.GeoSeries([territory_geom], crs='EPSG:3035').to_crs('EPSG:4326')
    centroid = territory_gs.iloc[0].centroid

    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=8,
        tiles='OpenStreetMap'
    )

    # Pre-compute area in ha from EPSG:3035 (metric, accurate) and attach to 4326 copy
    pa_display = pa_gdf_4326.copy()
    pa_display['area_ha'] = (pa_gdf.geometry.area / 10000).round(2).values

    # Filter invalid/empty geometries (folium expects GeoJSON geometries with coordinates)
    if 'geometry' in pa_display.columns:
        pa_display = pa_display[pa_display.geometry.notna()].copy()
        pa_display = pa_display[~pa_display.geometry.is_empty].copy()
        try:
            pa_display = pa_display[pa_display.is_valid].copy()
        except Exception:
            pass

        # Folium's bounds computation assumes geometries expose a GeoJSON 'coordinates' key;
        # GeometryCollection uses 'geometries' instead and can trigger KeyError('coordinates').
        # Keep only Polygon/MultiPolygon for display.
        try:
            geom_types = pa_display.geometry.geom_type
            allowed = geom_types.isin(['Polygon', 'MultiPolygon'])
            dropped = int((~allowed).sum())
            if dropped:
                logger.warning(
                    f"Dropping {dropped} feature(s) with unsupported geometry types for folium: "
                    f"{geom_types[~allowed].value_counts().to_dict()}"
                )
            pa_display = pa_display[allowed].copy()
        except Exception:
            pass

    # Deduplicate column names — folium/__geo_interface__ raises ValueError otherwise
    # (WDPA files sometimes have duplicate columns e.g. two 'NAME' columns)
    seen = {}
    new_cols = []
    for col in pa_display.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    pa_display.columns = new_cols

    # Build colour lookup once
    _colour_map = {
        cn: iucn_classes.get(cn, {}).get('colour', '#B4B2A9')
        for cn in pa_display['protection_class'].unique()
    } if 'protection_class' in pa_display.columns else {}

    # Single GeoJson layer (replaces one folium object per PA — massive speed-up)
    iucn_col = 'IUCN_MAX' if 'IUCN_MAX' in pa_display.columns else 'IUCN_CAT'
    tooltip_fields   = [c for c in ['WDPA_NAME', 'protection_class', iucn_col, 'area_ha']
                        if c in pa_display.columns]
    tooltip_aliases  = [a for c, a in [('WDPA_NAME', 'Name'), ('protection_class', 'Class'),
                                        (iucn_col, 'IUCN Cat.'), ('area_ha', 'Area (ha)')]
                        if c in pa_display.columns]

    if len(pa_display) > 0:
        folium.GeoJson(
            pa_display,
            style_function=lambda feature: {
                'fillColor': _colour_map.get(
                    feature['properties'].get('protection_class', 'unassigned'), '#B4B2A9'
                ),
                'color': _colour_map.get(
                    feature['properties'].get('protection_class', 'unassigned'), '#B4B2A9'
                ),
                'weight': 1,
                'fillOpacity': 0.5
            },
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases)
        ).add_to(m)
    else:
        st.warning("No valid protected-area geometries to display on the map.")

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 250px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px">
    <p style="margin-bottom:5px; font-weight:bold;">Protection Classes</p>
    """

    for class_name, class_info in iucn_classes.items():
        if class_name == 'oecm':  # Skip OECM for now (not in WDPA)
            continue
        colour = class_info.get('colour', '#B4B2A9')
        label = class_info.get('label', class_name)
        legend_html += f"""
        <p style="margin:3px 0;">
            <span style="background-color:{colour}; width:20px; height:15px;
                         display:inline-block; margin-right:5px;"></span>
            {label}
        </p>
        """

    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    # Display map
    st_folium(m, width=None, height=500)

    st.markdown("---")

    # ===================================================================
    # Row 3: Two columns - Coverage stats & Representativity chart
    # ===================================================================
    st.subheader("Detailed Statistics")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Coverage by Protection Class")

        # Compute coverage statistics
        coverage_df = coverage_by_class(pa_gdf, territory_area_ha)

        # Format for display
        display_df = coverage_df.copy()
        display_df['area_ha'] = display_df['area_ha'].apply(lambda x: f"{x:,.0f}")
        display_df['pct_territory'] = display_df['pct_territory'].apply(lambda x: f"{x:.2f}%")

        st.dataframe(
            display_df,
            hide_index=True,
            width='stretch',
            column_config={
                'protection_class': st.column_config.TextColumn('Class'),
                'area_ha': st.column_config.TextColumn('Area (ha)'),
                'pct_territory': st.column_config.TextColumn('% Territory'),
                'n_sites': st.column_config.NumberColumn('Sites', format="%d")
            }
        )

        # IUCN category breakdown
        st.markdown("#### Coverage by IUCN Category")
        iucn_col_stats = 'IUCN_MAX' if 'IUCN_MAX' in pa_gdf.columns else 'IUCN_CAT'
        iucn_rows = []  # also reused by "Generate DOCX Report" below — avoids recomputation
        if iucn_col_stats in pa_gdf.columns:
            # Compute total union once — reused for TOTAL row (avoids 3× union_all)
            total_pa_area_ha = pa_gdf.geometry.union_all().area / 10000.0
            cats_sorted_stats = sorted(
                pa_gdf[iucn_col_stats].unique(), key=iucn_category_sort_key
            )
            for cat in cats_sorted_stats:
                grp = pa_gdf[pa_gdf[iucn_col_stats] == cat]
                # NOTE: intentionally NOT named `net_area` — that name is used
                # above for the territory-wide net protected area metric, and
                # reusing it here would silently overwrite it for any code
                # later in this function (e.g. the DOCX report below).
                cat_area_ha = grp.geometry.union_all().area / 10000.0
                iucn_rows.append({
                    'IUCN Category': cat,
                    'Area (ha)': f"{cat_area_ha:,.0f}",
                    '% Territory': f"{cat_area_ha / territory_area_ha * 100:.2f}%",
                    'Sites': len(grp)
                })
            iucn_rows.append({
                'IUCN Category': 'TOTAL',
                'Area (ha)': f"{total_pa_area_ha:,.0f}",
                '% Territory': f"{total_pa_area_ha / territory_area_ha * 100:.2f}%",
                'Sites': len(pa_gdf)
            })
            st.dataframe(pd.DataFrame(iucn_rows), hide_index=True, width='stretch')

    with col_right:
        st.markdown("#### Ecosystem Representativity (CLC-based)")
        st.caption(
            "PA coverage per broad CLC ecosystem type — "
            "Forests, Grasslands, Wetlands, Water bodies, "
            "Semi-natural open land, Agricultural areas. "
            "Requires the CLC land-use raster to be loaded in the Data Upload tab."
        )

        clc_path = st.session_state.get('criterion_raster_paths', {}).get('landuse')
        # Include territory_geom identity in key so stale results (computed
        # without a study-area mask) are not served after the geom is available.
        _ri_cache_key = (clc_path, id(pa_gdf), id(territory_geom))
        _cache_valid  = (st.session_state.get('_ri_cache_key') == _ri_cache_key
                         and 'ri_df' in st.session_state)

        if clc_path:
            _btn_label = "↻ Recompute" if _cache_valid else "Compute Ecosystem RI"
            _compute   = st.button(
                _btn_label,
                key='btn_compute_ri',
                help="Pixel-count representativity from the CLC raster. "
                     "May take 10–30 s for large rasters.",
            )
            if _compute:
                try:
                    with st.spinner("Computing ecosystem representativity from CLC… (may take ~30 s)"):
                        from modules.module1_protected_areas.representativity import (
                            representativity_from_clc_raster
                        )
                        ri_df, class_df = representativity_from_clc_raster(
                            clc_path=clc_path,
                            pa_gdf=pa_gdf,
                            target_threshold=0.30,
                            study_area_geom=territory_geom,
                        )
                    st.session_state['ri_df']         = ri_df
                    st.session_state['ri_class_df']   = class_df
                    st.session_state['_ri_cache_key'] = _ri_cache_key
                    _cache_valid = True
                except Exception as _e:
                    st.warning(f"Representativity computation failed: {_e}")
                    st.session_state.pop('ri_df', None)
                    _cache_valid = False
        else:
            st.info(
                "Load the **CLC land-use raster** in the ① Data Upload tab "
                "to enable ecosystem representativity."
            )

        ri_df = st.session_state.get('ri_df') if _cache_valid else None

        if ri_df is not None and len(ri_df) > 0:
            synthetic_ri = ri_df['RI'].mean()
            _ri_label = "Good" if synthetic_ri >= 0.7 else ("Fair" if synthetic_ri >= 0.3 else "Poor")
            st.metric(
                "Synthetic RI",
                f"{synthetic_ri:.2f}",
                f"{_ri_label} — {(ri_df['RI'] >= 1.0).sum()}/{len(ri_df)} types at 30% target",
                help=(
                    "Mean Representativity Index across ecosystem types. "
                    "RI = min(PA coverage / 30%, 1.0). "
                    "1.0 = meets the KMGBF 30% target, 0.0 = no protection."
                )
            )

            # ── Coverage RI table ─────────────────────────────────────────────
            display_ri = ri_df[['ecosystem_type', 'total_ha', 'protected_ha',
                                 'coverage_pct', 'RI', 'gap_ha']].copy()
            display_ri['total_ha']     = display_ri['total_ha'].apply(lambda x: f"{x:,.0f}")
            display_ri['protected_ha'] = display_ri['protected_ha'].apply(lambda x: f"{x:,.0f}")
            display_ri['coverage_pct'] = display_ri['coverage_pct'].apply(lambda x: f"{x:.1f}%")
            display_ri['RI']           = display_ri['RI'].apply(lambda x: f"{x:.3f}")
            display_ri['gap_ha']       = display_ri['gap_ha'].apply(lambda x: f"{x:,.0f}")
            st.dataframe(
                display_ri,
                hide_index=True,
                use_container_width=True,
                column_config={
                    'ecosystem_type': 'Ecosystem Type',
                    'total_ha':       st.column_config.TextColumn('Total (ha)'),
                    'protected_ha':   st.column_config.TextColumn('Protected (ha)'),
                    'coverage_pct':   st.column_config.TextColumn('PA Coverage'),
                    'RI':             st.column_config.TextColumn('RI'),
                    'gap_ha':         st.column_config.TextColumn('Gap to 30% (ha)'),
                }
            )

            # ── Ecosystem share by protection class ───────────────────────────
            class_df = st.session_state.get('ri_class_df')
            if class_df is not None and len(class_df) > 0:
                st.markdown("**Ecosystem coverage by protection class (% of ecosystem area)**")
                st.caption(
                    "strict_core = IUCN Ia/Ib/II · "
                    "regulatory = IUCN III–VI · "
                    "contractual = OECM/contractual · "
                    "unassigned = no IUCN category"
                )
                _cls_cols = [c for c in
                             ['strict_core', 'regulatory', 'contractual', 'unassigned']
                             if c in class_df.columns]
                display_cls = class_df[['ecosystem_type', 'total_ha'] + _cls_cols].copy()
                display_cls['total_ha'] = display_cls['total_ha'].apply(lambda x: f"{x:,.0f}")
                for _c in _cls_cols:
                    display_cls[_c] = display_cls[_c].apply(lambda x: f"{x:.1f}%")
                _col_labels = {
                    'ecosystem_type': 'Ecosystem Type',
                    'total_ha':       st.column_config.TextColumn('Total (ha)'),
                    'strict_core':    st.column_config.TextColumn('Strict core\n(Ia/Ib/II)'),
                    'regulatory':     st.column_config.TextColumn('Regulatory\n(III–VI)'),
                    'contractual':    st.column_config.TextColumn('Contractual\n(OECM)'),
                    'unassigned':     st.column_config.TextColumn('Unassigned'),
                }
                st.dataframe(
                    display_cls,
                    hide_index=True,
                    use_container_width=True,
                    column_config={k: v for k, v in _col_labels.items()
                                   if k in display_cls.columns},
                )

        elif clc_path and not _cache_valid:
            st.caption("Click **Compute Ecosystem RI** above to run the analysis.")

    st.markdown("---")

    # ===================================================================
    # Row 4: Gap analysis
    # ===================================================================
    st.subheader("Gap Analysis")

    if st.button("Run Gap Analysis"):
        status = st.empty()
        status.info("Computing gap layers...")
        try:
            # Compute gap layers
            strict_gaps_gdf = strict_gaps(pa_gdf, territory_geom)
            qual_gaps_gdf = qualitative_gaps(pa_gdf, territory_geom)
            corridors_gdf = potential_corridors(pa_gdf, territory_geom, max_gap_m=5000.0)

            # Store layers in session state
            st.session_state['gap_layers'] = {
                'strict_gaps': strict_gaps_gdf,
                'qualitative_gaps': qual_gaps_gdf,
                'corridors': corridors_gdf
            }

            # Store raw areas only — % computed dynamically from current
            # territory_area_ha at render time (so they stay correct if
            # the user switches territory without re-running the analysis)
            st.session_state['gap_stats'] = {
                'strict_area':   strict_gaps_gdf.geometry.area.sum() / 10000.0 if len(strict_gaps_gdf) > 0 else 0.0,
                'qual_area':     qual_gaps_gdf.geometry.area.sum()   / 10000.0 if len(qual_gaps_gdf)   > 0 else 0.0,
                'corridor_area': corridors_gdf.geometry.area.sum()   / 10000.0 if len(corridors_gdf)   > 0 else 0.0,
            }

        except Exception as e:
            st.error(f"Gap analysis failed: {str(e)}")
        finally:
            status.empty()

    # Display statistics + map whenever results are available in session state
    # (rendered OUTSIDE the button block so they survive Streamlit re-runs)
    if 'gap_layers' in st.session_state:

        # --- Summary metrics (persist across re-runs via session_state) ---
        if 'gap_stats' in st.session_state:
            gs = st.session_state['gap_stats']
            st.success("Gap analysis complete!")

            # % computed fresh from current territory_area_ha so they are
            # always consistent with the selected region denominator
            def _pct(area_ha):
                return area_ha / territory_area_ha * 100.0 if territory_area_ha > 0 else 0.0

            st.caption(
                f"Reference territory: **{territory_area_ha:,.0f} ha** "
                f"({territory_area_ha / 100:.0f} km²) — "
                "percentages below are relative to this total area."
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Strict Gaps",
                    f"{gs['strict_area']:,.0f} ha",
                    f"{_pct(gs['strict_area']):.1f}% of territory",
                    delta_color="off",
                    help="Areas with NO PA coverage of any kind. "
                         "strict_gap% + total_PA_coverage% ≈ 100%."
                )
            with col2:
                st.metric(
                    "Qualitative Gaps",
                    f"{gs['qual_area']:,.0f} ha",
                    f"{_pct(gs['qual_area']):.1f}% of territory",
                    delta_color="off",
                    help="Areas covered ONLY by weak protection classes "
                         "(contractual / unassigned). Superset of strict gaps."
                )
            with col3:
                st.metric(
                    "Potential Corridors",
                    f"{gs['corridor_area']:,.0f} ha",
                    f"{_pct(gs['corridor_area']):.1f}% of territory",
                    delta_color="off",
                    help="Unprotected areas within 5 km of two or more PA patches "
                         "(potential ecological corridors). Independent of gap metrics."
                )

        st.markdown("#### Gap Layers Map")

        gap_layers = st.session_state['gap_layers']

        # Create folium map (centroid already in EPSG:4326 from above)
        m_gaps = folium.Map(
            location=[centroid.y, centroid.x],
            zoom_start=8,
            tiles='OpenStreetMap'
        )

        # Helper: filter out empty/null geometries that crash folium
        def _clean_for_folium(gdf):
            """Remove empty/null/coordinate-less geometries and reproject to EPSG:4326.

            Folium's get_bounds() calls iter_coords() which expects a
            'coordinates' key in every GeoJSON feature geometry.  Geometry
            types like GeometryCollection use 'geometries' instead, causing
            a KeyError.  We convert GeometryCollections to their union and
            drop anything that still lacks coordinates.
            """
            from shapely.ops import unary_union
            from shapely.geometry import mapping as _shp_mapping, GeometryCollection

            clean = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
            if len(clean) > 0:
                # Flatten GeometryCollections → their unary_union
                def _flatten(geom):
                    if isinstance(geom, GeometryCollection):
                        u = unary_union(geom.geoms) if geom.geoms else geom
                        return u
                    return geom
                clean = clean.copy()
                clean['geometry'] = clean.geometry.apply(_flatten)
                # Re-filter after flattening
                clean = clean[~clean.geometry.is_empty & clean.geometry.notnull()]

            if len(clean) > 0:
                # Final safety: drop any geometry whose GeoJSON lacks 'coordinates'
                def _has_coords(geom):
                    try:
                        m = _shp_mapping(geom)
                        return 'coordinates' in m
                    except Exception:
                        return False
                clean = clean[clean.geometry.apply(_has_coords)]

            if len(clean) > 0 and hasattr(clean, 'to_crs'):
                clean = clean.to_crs('EPSG:4326')
            return clean

        # Add gap layers as toggleable overlays (reproject to 4326)
        strict_4326 = _clean_for_folium(gap_layers['strict_gaps'])
        if len(strict_4326) > 0:
            folium.GeoJson(
                strict_4326,
                name='Strict Gaps',
                style_function=lambda x: {
                    'fillColor': '#D0021B',
                    'color': '#D0021B',
                    'weight': 1,
                    'fillOpacity': 0.4
                }
            ).add_to(m_gaps)

        qual_4326 = _clean_for_folium(gap_layers['qualitative_gaps'])
        if len(qual_4326) > 0:
            folium.GeoJson(
                qual_4326,
                name='Qualitative Gaps',
                style_function=lambda x: {
                    'fillColor': '#F6A623',
                    'color': '#F6A623',
                    'weight': 1,
                    'fillOpacity': 0.4
                }
            ).add_to(m_gaps)

        corr_4326 = _clean_for_folium(gap_layers['corridors'])
        if len(corr_4326) > 0:
            folium.GeoJson(
                corr_4326,
                name='Potential Corridors',
                style_function=lambda x: {
                    'fillColor': '#4A90E2',
                    'color': '#4A90E2',
                    'weight': 1,
                    'fillOpacity': 0.4
                }
            ).add_to(m_gaps)

        # Add layer control
        folium.LayerControl().add_to(m_gaps)

        st_folium(m_gaps, width='stretch', height=400)

    else:
        st.info("Click 'Run Gap Analysis' to compute and display gap layers.")

    st.markdown("---")

    # ===================================================================
    # Criterion profiles within existing protected areas
    # ===================================================================
    st.markdown("### Criterion profiles within existing protected areas")

    try:
        # Check if raster paths are available in session state
        raster_paths = st.session_state.get('criterion_raster_paths', None)

        if raster_paths is None or len(raster_paths) == 0:
            st.info(
                "Raster paths for MCE criteria not yet available. "
                "Run the ⑤ OECM Favourability Analysis first to enable criterion profiling within PAs."
            )
        else:
            # Import zonal stats functions
            from modules.module1_protected_areas.zonal_stats import (
                zonal_stats_by_pa_class,
                criterion_coverage_summary,
                iucn_category_sort_key
            )

            if st.button("Compute Criterion Profiles"):
                status = st.empty()
                status.info("Computing zonal statistics for all criteria...")
                try:
                    # Compute zonal statistics
                    # Use the same IUCN column as the coverage table so categories match
                    _iucn_col = 'IUCN_MAX' if 'IUCN_MAX' in pa_gdf.columns else 'IUCN_CAT'
                    zonal_df = zonal_stats_by_pa_class(pa_gdf, raster_paths, iucn_col=_iucn_col)

                    if len(zonal_df) == 0:
                        st.warning("No zonal statistics computed (no valid overlaps)")
                    else:
                        # Store in session state
                        st.session_state['zonal_stats'] = zonal_df

                        st.success(
                            f"Computed statistics for {len(zonal_df['criterion'].unique())} criteria "
                            f"across {len(zonal_df['iucn_cat'].unique())} IUCN categories"
                        )

                except Exception as e:
                    st.error(f"Failed to compute zonal statistics: {str(e)}")
                    logger.exception("Zonal statistics computation error:")
                finally:
                    status.empty()

            # Display results if available
            if 'zonal_stats' in st.session_state and st.session_state['zonal_stats'] is not None:
                zonal_df = st.session_state['zonal_stats']

                # Row 1: Grouped bar chart
                st.markdown("#### Mean Criterion Scores by IUCN Category")
                st.caption(
                    "All criteria normalised to [0–1] for display. "
                    "Anthropogenic pressure is inverted (lower raw value = higher score). "
                    "Land use (CLC categorical codes) is excluded — see breakdown below."
                )

                try:
                    import plotly.express as px

                    chart_data = zonal_df[zonal_df['criterion'] != 'landuse'][['criterion', 'iucn_cat', 'mean']].copy()
                    for crit in chart_data['criterion'].unique():
                        mask = chart_data['criterion'] == crit
                        vals = chart_data.loc[mask, 'mean']
                        vmin, vmax = vals.min(), vals.max()
                        if vmax > vmin:
                            chart_data.loc[mask, 'mean'] = (vals - vmin) / (vmax - vmin)
                        if crit == 'anthropogenic_pressure':
                            chart_data.loc[mask, 'mean'] = 1.0 - chart_data.loc[mask, 'mean']

                    cat_order = sorted(
                        [c for c in chart_data['iucn_cat'].unique() if c != 'outside'],
                        key=iucn_category_sort_key
                    )
                    if 'outside' in chart_data['iucn_cat'].unique():
                        cat_order.append('outside')

                    fig = px.bar(
                        chart_data,
                        x='criterion',
                        y='mean',
                        color='iucn_cat',
                        barmode='group',
                        category_orders={'iucn_cat': cat_order},
                        labels={
                            'criterion': 'Criterion',
                            'mean': 'Normalised Score [0–1]',
                            'iucn_cat': 'IUCN Category'
                        },
                        title='',
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )

                    fig.update_layout(
                        xaxis_tickangle=-45,
                        height=400,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1.0,
                            xanchor="left",
                            x=1.02
                        )
                    )

                    st.plotly_chart(fig, width='stretch')

                except ImportError:
                    st.warning("Plotly not available. Install plotly for interactive charts.")
                    st.bar_chart(
                        zonal_df[zonal_df['criterion'] != 'landuse'].pivot(
                            index='criterion', columns='iucn_cat', values='mean'
                        )
                    )

                # Row 2: Summary pivot table (continuous criteria only)
                st.markdown("#### Summary: Mean Scores by IUCN Category")

                try:
                    summary_df = criterion_coverage_summary(
                        zonal_df[zonal_df['criterion'] != 'landuse']
                    )

                    display_summary = summary_df.copy()
                    for col in display_summary.columns:
                        display_summary[col] = display_summary[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

                    st.dataframe(display_summary, width='stretch')

                except Exception as e:
                    st.error(f"Failed to generate summary table: {str(e)}")

                # Row 3: Expandable detailed statistics
                with st.expander("Detailed Statistics (min/median/max/std)"):
                    st.markdown("**Per-criterion statistics by IUCN category (continuous criteria only):**")

                    detailed_stats = []

                    for criterion in zonal_df['criterion'].unique():
                        if criterion == 'landuse':
                            continue
                        criterion_data = zonal_df[zonal_df['criterion'] == criterion]

                        _cats_present = criterion_data['iucn_cat'].unique()
                        _cats_sorted = sorted(
                            [c for c in _cats_present if c != 'outside'],
                            key=iucn_category_sort_key
                        )
                        if 'outside' in _cats_present:
                            _cats_sorted.append('outside')

                        for iucn_cat in _cats_sorted:
                            class_data = criterion_data[criterion_data['iucn_cat'] == iucn_cat]

                            if len(class_data) > 0:
                                row = class_data.iloc[0]
                                detailed_stats.append({
                                    'Criterion': criterion,
                                    'IUCN Category': iucn_cat,
                                    'Min': f"{row['min']:.3f}",
                                    'Median': f"{row['median']:.3f}",
                                    'Max': f"{row['max']:.3f}",
                                    'Std': f"{row['std']:.3f}",
                                    'Pixel Count': f"{row['pixel_count']:,}"
                                })

                    detailed_df = pd.DataFrame(detailed_stats)
                    st.dataframe(detailed_df, hide_index=True, width='stretch')

                # Row 4: CLC land use class breakdown — whole study area + per IUCN category
                if 'landuse' in st.session_state.get('criterion_raster_paths', {}):
                    st.markdown("#### Land Use Composition (CLC Level 1)")
                    st.caption(
                        "Share of each Corine Land Cover Level 1 category (% of valid pixels), "
                        "for the whole study area and within each IUCN protection category."
                    )
                    try:
                        from modules.module1_protected_areas.zonal_stats import (
                            clc_composition_by_iucn_category
                        )

                        lu_path = st.session_state['criterion_raster_paths']['landuse']

                        # Cached (session_state) so this isn't recomputed every
                        # rerun of this tab, and so the DOCX report below can
                        # reuse the exact same result instead of recomputing it.
                        _clc_key = (lu_path, id(pa_gdf))
                        _clc_cache = st.session_state.get('_clc_composition_cache')
                        if _clc_cache is not None and _clc_cache.get('key') == _clc_key:
                            clc_df = _clc_cache['df']
                        else:
                            clc_df = clc_composition_by_iucn_category(lu_path, pa_gdf)
                            st.session_state['_clc_composition_cache'] = {
                                'key': _clc_key, 'df': clc_df
                            }
                        st.session_state['clc_composition_df'] = clc_df

                        # Format for display: rename columns, add % suffix
                        clc_display = clc_df.rename(
                            columns={'level1': 'Level 1', 'category': 'Category'}
                        ).copy()
                        for col in clc_display.columns[2:]:
                            clc_display[col] = clc_display[col].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(clc_display, hide_index=True, width='stretch')

                    except Exception as e:
                        st.caption(f"CLC breakdown unavailable: {e}")

    except Exception as e:
        st.warning(
            f"Criterion profiling unavailable: {str(e)}\n\n"
            "This feature requires raster paths to be available in session state."
        )

    st.markdown("---")

    # ===================================================================
    # Ecosystem Account (SEEA-EA style: extent + condition + services)
    # ===================================================================
    st.subheader("Ecosystem Account")
    st.markdown(
        "Habitat-level extent, condition and service-capacity table — one row per "
        "habitat type, giving its **surface area**, mean **ecosystem condition**, "
        "and mean **regulating / cultural / provisioning service capacity**. "
        "Mirrors the SEEA Ecosystem Accounting logic (extent + condition + "
        "services accounts), computed directly from the aligned criterion "
        "rasters — the same data used in the **⑤ OECM Favourability Analysis**."
    )

    _aa_eco  = st.session_state.get('_aligned_arrays')
    _prof_eco = st.session_state.get('_aligned_profile')

    if _aa_eco is None or _prof_eco is None:
        st.info(
            "Click **Load & Align Rasters** in the **⑤ OECM Favourability Analysis** "
            "tab first — the ecosystem account reuses those same aligned rasters."
        )
    else:
        _clc_level_label = st.radio(
            "CLC classification level",
            options=[
                "Level 1 — 5 broad classes",
                "Level 2 — 15 classes",
                "Level 3 — 44 classes (full nomenclature)",
            ],
            horizontal=True,
            key='ecosystem_account_clc_level',
            help="Corine Land Cover hierarchy level used to group habitats in the table "
                 "below. Level 1 gives a quick overview; Level 3 gives the full "
                 "nomenclature (e.g. distinguishing broad-leaved from coniferous forest)."
        )
        _clc_level = int(_clc_level_label.split(" ")[1])

        try:
            from modules.module1_protected_areas.representativity import ecosystem_account_table

            _eco_key = (st.session_state.get('_aligned_key'), _clc_level)
            _eco_cache = st.session_state.get('_ecosystem_account_cache')
            if _eco_cache is not None and _eco_cache.get('key') == _eco_key:
                _eco_df = _eco_cache['df']
            else:
                _eco_df = ecosystem_account_table(_aa_eco, _prof_eco, clc_level=_clc_level)
                st.session_state['_ecosystem_account_cache'] = {'key': _eco_key, 'df': _eco_df}

            _eco_disp = _eco_df.rename(columns={
                'ecosystem_type': 'Habitat',
                'area_ha': 'Surface (ha)',
                'pct_area': '% of area',
                'condition_mean': 'Condition',
                'regulating_es_mean': 'Regulating ES',
                'cultural_es_mean': 'Cultural ES',
                'provisioning_es_mean': 'Provisioning ES',
            }).copy()
            for _c in ['Surface (ha)']:
                _eco_disp[_c] = _eco_disp[_c].apply(lambda x: f"{x:,.0f}")
            for _c in ['% of area']:
                _eco_disp[_c] = _eco_disp[_c].apply(lambda x: f"{x:.1f}%")
            for _c in ['Condition', 'Regulating ES', 'Cultural ES', 'Provisioning ES']:
                _eco_disp[_c] = _eco_disp[_c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

            st.dataframe(
                _eco_disp.style.apply(
                    lambda row: ['font-weight: bold' if row.name == len(_eco_disp) - 1 else '' for _ in row],
                    axis=1
                ),
                hide_index=True, width='stretch'
            )
            st.caption(
                "Condition and service-capacity columns are the mean of each aligned "
                "criterion raster within the pixels of that habitat type (all normalised "
                "[0-1] at source). The last row totals the whole study area."
            )

            save_and_download(
                "Download CSV",
                data=_eco_df.to_csv(index=False),
                file_name="ecosystem_account.csv",
                mime="text/csv",
                key='ecosystem_account_csv_download',
            )
        except Exception as _e:
            st.error(f"Ecosystem account failed: {_e}")

    st.markdown("---")

    # ===================================================================
    # Footer: Export buttons
    # ===================================================================
    st.subheader("Export & Analysis Tools")

    # ── Report generation ────────────────────────────────────────────────
    st.markdown("#### Diagnostic Report")
    st.caption(
        "Generate a fully documented report (DOCX) containing all maps, "
        "charts and tables computed above.  Requires **python-docx** "
        "(`pip install python-docx`)."
    )

    report_col1, report_col2, _ = st.columns([1, 1, 2])
    with report_col1:
        if st.button("Generate DOCX Report", type="primary", use_container_width=True):
            with st.spinner("Building report…"):
                try:
                    from modules.module1_protected_areas.report_generator import (
                        generate_docx_report
                    )

                    # Reuse tables already computed above (Key Indicators /
                    # Coverage by Protection Class / Coverage by IUCN Category)
                    # instead of recomputing them — union_all() on the full PA
                    # layer is the same expensive operation whether it's done
                    # here or above, so recomputing it was pure duplicate work.
                    _cov_df     = coverage_df
                    _kmgbf      = kmgbf_pct
                    _strict_pct = strict_pct
                    _net_area   = net_area
                    _iucn_df    = pd.DataFrame(iucn_rows) if iucn_rows else None

                    docx_bytes = generate_docx_report(
                        territory_name=st.session_state.get('parameters', {}).get(
                            'study_area_name', 'Unknown territory'),
                        territory_area_ha=territory_area_ha,
                        pa_gdf=pa_gdf,
                        territory_geom=territory_geom,
                        iucn_classes=iucn_classes,
                        coverage_df=_cov_df,
                        iucn_coverage_df=_iucn_df,
                        gap_layers=st.session_state.get('gap_layers'),
                        gap_stats=st.session_state.get('gap_stats'),
                        ri_df=st.session_state.get('ri_df'),
                        zonal_df=st.session_state.get('zonal_stats'),
                        kmgbf_pct=_kmgbf,
                        net_area_ha=_net_area,
                        strict_pct=_strict_pct,
                        clc_composition_df=st.session_state.get('clc_composition_df'),
                    )
                    st.session_state['_module1_report_bytes'] = docx_bytes
                    st.success("Report ready — click Download below.")

                except ImportError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Report generation failed: {e}")
                    logger.exception("Module 1 report generation error")

    with report_col2:
        report_bytes = st.session_state.get('_module1_report_bytes')
        territory_slug = (
            st.session_state.get('parameters', {})
            .get('study_area_name', 'territory')
            .replace(' ', '_')[:30]
        )
        save_and_download(
            label="Download DOCX",
            data=report_bytes or b'',
            file_name=f"module1_diagnostic_{territory_slug}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            disabled=report_bytes is None,
            use_container_width=True,
        )

    st.markdown("---")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        if st.button("Export PA Statistics (CSV)"):
            # Reuse the coverage table already computed above ("Coverage by
            # Protection Class") instead of recomputing union_all() on pa_gdf.

            # Create temporary CSV file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
                coverage_df.to_csv(tmp.name, index=False)

                # Provide download button
                with open(tmp.name, 'r') as f:
                    csv_data = f.read()

                save_and_download(
                    label="Download CSV",
                    data=csv_data,
                    file_name="pa_coverage_statistics.csv",
                    mime="text/csv"
                )

    with col_exp2:
        if 'ri_df' in st.session_state and st.session_state['ri_df'] is not None:
            if st.button("Apply Weight Suggestions to ⑤ Favourability Analysis →"):
                # Propose Group A weights from representativity deficits
                try:
                    ri_df = st.session_state['ri_df']

                    # Load criterion-ecosystem mapping from config
                    config_path = Path(__file__).parent.parent / "config" / "criteria_defaults.yaml"
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)

                    criterion_mapping = config.get('criterion_ecosystem_mapping', {})

                    # Propose weights
                    proposed_weights = propose_group_a_weights(ri_df, criterion_mapping)

                    # Validate handoff before storing
                    from modules.module1_protected_areas.handoff import (
                        validate_weight_handoff,
                        format_weights_for_mce
                    )

                    validate_weight_handoff(proposed_weights, str(config_path))
                    formatted_weights = format_weights_for_mce(proposed_weights)

                    # Store validated weights in session state
                    st.session_state['proposed_group_a_weights'] = formatted_weights

                    st.success(
                        "Weight suggestions validated and applied! "
                        "Go to **② Parameters** Section 7 to apply them to ⑤ OECM Favourability Analysis."
                    )

                except ValueError as e:
                    st.error(f"Weight validation failed: {str(e)}")
                except Exception as e:
                    st.error(f"Failed to propose weights: {str(e)}")
        else:
            st.info("Run representativity analysis with ecosystem layer to enable weight suggestions.")
