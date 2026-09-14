"""Streamlit UI for Module 2 — OECM Favourability Analysis."""
import logging
import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import io
import base64
import json
from datetime import datetime
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def _to_multipolygon(geom):
    """Flatten any geometry to a MultiPolygon, extracting only polygon parts.

    Folium's GeoJSON serialiser chokes on GeometryCollection because that type
    uses "geometries" rather than "coordinates" in GeoJSON. Converting to a
    MultiPolygon fixes this.
    """
    if geom is None or geom.is_empty:
        return None
    from shapely.geometry import MultiPolygon, Polygon, GeometryCollection
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        from shapely.ops import unary_union
        return unary_union(polys)
    return geom


def render_tab_module2(score_array=None, oecm_mask=None, classical_pa_mask=None,
                       eliminatory_mask=None, profile=None, params=None):
    """
    Render Module 2 results. Called only when score_array is available.

    Parameters
    ----------
    score_array : np.ndarray
        Favourability score array [0-1], NaN where eliminated.
    oecm_mask : np.ndarray
        Boolean mask: True = OECM favourable (Group C >= threshold).
    classical_pa_mask : np.ndarray
        Boolean mask: True = classical PA preferable (Group C < threshold).
    eliminatory_mask : np.ndarray, optional
        Boolean mask: True = pixel passed Group D (not eliminated).
    profile : dict
        Rasterio profile (CRS, transform, dimensions).
    params : dict, optional
        Full parameter dictionary from sidebar for reproducibility logging.
    """
    # ===================================================================
    # Row 1: Summary metric cards
    # ===================================================================
    st.subheader("Favourability Summary")

    # Use |pixel_width| × |pixel_height| to support non-square pixels.
    # transform[0] = pixel width (x), transform[4] = pixel height (y, typically negative).
    pixel_area_ha = (abs(profile['transform'][0]) * abs(profile['transform'][4])) / 10000.0
    territory_area_ha = profile['width'] * profile['height'] * pixel_area_ha

    valid_mask = ~np.isnan(score_array)
    n_valid = int(np.sum(valid_mask))
    n_total = score_array.size
    n_eliminated = n_total - n_valid

    oecm_area_ha = float(np.sum(oecm_mask)) * pixel_area_ha if oecm_mask is not None else 0.0
    classical_pa_area_ha = float(np.sum(classical_pa_mask)) * pixel_area_ha if classical_pa_mask is not None else 0.0
    oecm_pct = (oecm_area_ha / territory_area_ha * 100.0) if territory_area_ha > 0 else 0.0
    classical_pa_pct = (classical_pa_area_ha / territory_area_ha * 100.0) if territory_area_ha > 0 else 0.0

    valid_scores = score_array[valid_mask]
    median_score = float(np.median(valid_scores)) if len(valid_scores) > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)

    eligible_area_ha = n_valid * pixel_area_ha
    eligible_pct = (eligible_area_ha / territory_area_ha * 100.0) if territory_area_ha > 0 else 0.0

    with col1:
        st.metric(
            label="Eligible Area (passed Group D)",
            value=f"{eligible_area_ha:,.0f} ha",
            delta=f"{eligible_pct:.1f}% of territory",
            help="Pixels not eliminated by pressure or incompatible land use"
        )

    with col2:
        st.metric(
            label="OECM Favourable",
            value=f"{oecm_area_ha:,.0f} ha",
            delta=f"{oecm_pct:.1f}% of territory",
            help="Eligible pixels with sufficient production function (Group C ≥ threshold)"
        )

    with col3:
        st.metric(
            label="Low use-function (MCE Group C < 0.10)",
            value=f"{classical_pa_area_ha:,.0f} ha",
            delta=f"{classical_pa_pct:.1f}% of territory",
            help=(
                "Eligible pixels where Group C score (provisioning ES + compatible land use) "
                "is below 0.10 — these areas lack sufficient production function for OECM "
                "and may be better candidates for classical PA designation. "
                "This is NOT the existing WDPA protected area network."
            )
        )

    with col4:
        st.metric(
            label="Median Score (eligible)",
            value=f"{median_score:.2f}",
            help="Median favourability score across all eligible pixels (0-1 scale)"
        )

    # Elimination diagnostic
    elim_pct = (n_eliminated / n_total * 100.0) if n_total > 0 else 0.0
    if n_eliminated > 0:
        st.caption(
            f"Group D eliminated {n_eliminated:,} pixels ({elim_pct:.1f}% of grid) — "
            "high anthropogenic pressure or incompatible land use. "
            "Adjust the pressure threshold in the sidebar to change this."
        )

    st.markdown("---")

    # ===================================================================
    # Sub-tabs: Map and Statistics
    # ===================================================================
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "Map", "Statistics", "Sensitivity Analysis", "Candidate Sites"
    ])

    with subtab1:
        # =================================================================
        # MAP TAB: Threshold explorer then map
        # =================================================================
        st.subheader("Favourability Map")

        # ── Controls row ─────────────────────────────────────────────────
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 1, 2])
        with ctrl_col1:
            map_threshold = st.slider(
                "Display threshold — show areas ≥",
                min_value=0.0, max_value=1.0,
                value=st.session_state.get('export_threshold', 0.5),
                step=0.05, key='map_threshold_slider',
                help="Only pixels with favourability score ≥ this value are shown on the map"
            )
            st.session_state['export_threshold'] = map_threshold

        with ctrl_col2:
            above_threshold = valid_mask & (score_array >= map_threshold)
            area_above_ha = np.sum(above_threshold) * pixel_area_ha
            pct_above = (area_above_ha / territory_area_ha * 100.0) if territory_area_ha > 0 else 0.0
            st.metric(
                label=f"Area ≥ {map_threshold:.2f}",
                value=f"{area_above_ha:,.0f} ha",
                delta=f"{pct_above:.1f}% of study area"
            )

        with ctrl_col3:
            # PA overlay toggle — inline so it's discoverable without hunting the sidebar
            _pa_available = st.session_state.get('pa_gdf') is not None
            show_pa_ov = st.checkbox(
                "Show PA network on map",
                value=st.session_state.get('show_pa_overlay', True),
                key='show_pa_overlay',
                disabled=not _pa_available,
                help="Overlay the PA polygons (colour-coded by protection class). "
                     "Run the PA Diagnostic first to enable this."
            )
            _excl_active = (params or {}).get('exclude_pa_pixels', False)
            if _excl_active:
                _excl_cls = (params or {}).get('exclude_pa_classes', [])
                st.caption(f"PA exclusion ON — {', '.join(_excl_cls) or 'all classes'}")
            else:
                st.caption("PA exclusion OFF (sidebar › 6e to change)")

        # Create folium map with raster overlay
        if n_valid == 0:
            st.warning(
                "All pixels were eliminated by Group D criteria (pressure threshold or "
                "incompatible land use). Increase the pressure threshold in the sidebar "
                "or verify your input layers cover the study area."
            )

        try:
            import matplotlib.cm as mcm
            from rasterio.warp import (reproject, calculate_default_transform, Resampling)
            from rasterio.transform import array_bounds

            cmap = mcm.get_cmap('RdYlGn')
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

            # ------------------------------------------------------------------
            # Reproject score_array from EPSG:3035 → EPSG:4326 once and cache
            # the result in session state.  Moving the threshold slider only
            # recolours the cached array — no warp on every interaction.
            # ------------------------------------------------------------------
            _warp_key = st.session_state.get('_aligned_key')
            _cached   = st.session_state.get('_score_4326_cache')

            if _cached is None or _cached.get('key') != _warp_key:
                src_crs = profile['crs']
                dst_crs = 'EPSG:4326'
                transform_4326, w_4326, h_4326 = calculate_default_transform(
                    src_crs, dst_crs,
                    profile['width'], profile['height'],
                    *array_bounds(profile['height'], profile['width'], profile['transform'])
                )
                score_4326 = np.full((h_4326, w_4326), np.nan, dtype=np.float32)
                reproject(
                    source=score_array.astype(np.float32),
                    destination=score_4326,
                    src_transform=profile['transform'],
                    src_crs=src_crs,
                    dst_transform=transform_4326,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    src_nodata=np.nan,
                    dst_nodata=np.nan
                )
                west, south, east, north = array_bounds(h_4326, w_4326, transform_4326)
                st.session_state['_score_4326_cache'] = {
                    'key': _warp_key,
                    'score_4326': score_4326,
                    'bounds': (west, south, east, north),
                }
            else:
                score_4326 = _cached['score_4326']
                west, south, east, north = _cached['bounds']

            # Colour-map only the pixels at or above threshold (cheap, no warp)
            valid_mask_4326   = ~np.isnan(score_4326)
            display_mask_4326 = valid_mask_4326 & (score_4326 >= map_threshold)
            rgba = np.zeros((*score_4326.shape, 4), dtype=np.uint8)
            rgba[display_mask_4326] = (
                cmap(norm(score_4326[display_mask_4326])) * 255
            ).astype(np.uint8)

            img_pil = Image.fromarray(rgba, mode='RGBA')
            buffered = io.BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            # Create folium map — Esri World Light Gray Canvas: muted, low-contrast
            # basemap that doesn't compete with the red-yellow-green score overlay
            # (unlike standard OpenStreetMap, whose greens/tans clash with the
            # legend). Free, no API key. CartoDB Positron — the original choice —
            # now requires a CARTO account/API key, so it can't be used unauthenticated.
            center_lat = (south + north) / 2
            center_lon = (west + east) / 2

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=8,
                tiles=None,          # no default tile — we add it named below
            )
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
                attr='Esri, HERE, Garmin, © OpenStreetMap contributors, and the GIS user community',
                name='Basemap',
                control=True,
            ).add_to(m)

            # Add raster overlay — bounds in [[south, west], [north, east]] order
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{img_base64}",
                bounds=[[south, west], [north, east]],
                opacity=0.85,
                name='Favourability Score',
            ).add_to(m)

            # Study-area boundary
            study_area_geom = st.session_state.get('study_area_geometry')
            if study_area_geom is not None:
                import geopandas as gpd
                clean_geom = _to_multipolygon(study_area_geom)
                if clean_geom is not None and not clean_geom.is_empty:
                    _sa_gdf = gpd.GeoDataFrame(
                        [{'geometry': clean_geom}], crs='EPSG:3035'
                    ).to_crs('EPSG:4326')
                    folium.GeoJson(
                        _sa_gdf,
                        name='Study Area',
                        style_function=lambda x: {
                            'fillColor': 'none', 'color': '#ffffff',
                            'weight': 2, 'fillOpacity': 0,
                        },
                    ).add_to(m)

            # ── PA network overlay ───────────────────────────────────────
            if show_pa_ov:
                _pa_gdf_ov = st.session_state.get('pa_gdf')
                if _pa_gdf_ov is not None and len(_pa_gdf_ov) > 0:
                    try:
                        import geopandas as _gpd2
                        _ov_colours = {
                            'strict_core': '#1B5E20',
                            'regulatory':  '#388E3C',
                            'contractual': '#81C784',
                            'unassigned':  '#9E9E9E',
                        }
                        _iucn_col_ov = ('IUCN_MAX' if 'IUCN_MAX' in _pa_gdf_ov.columns
                                        else 'IUCN_CAT')
                        # Keep only the columns needed for display to reduce payload
                        _keep = [c for c in ['WDPA_NAME', 'protection_class', _iucn_col_ov, 'geometry']
                                 if c in _pa_gdf_ov.columns]
                        _pa_slim = _pa_gdf_ov[_keep].copy()

                        # 1. Simplify in projected CRS (100 m tolerance)
                        _pa_slim['geometry'] = _pa_slim.geometry.simplify(
                            100, preserve_topology=True
                        )
                        # 2. Flatten any GeometryCollection → MultiPolygon so
                        #    Folium/GeoJSON can serialise ("coordinates" key)
                        _pa_slim['geometry'] = _pa_slim.geometry.apply(_to_multipolygon)
                        _pa_slim = _pa_slim[_pa_slim.geometry.notna()].copy()
                        # 3. Reproject to WGS-84 (once, from the projected source CRS)
                        _pa_slim = _pa_slim.to_crs('EPSG:4326')

                        _tt_fields = [c for c in ['WDPA_NAME', 'protection_class', _iucn_col_ov]
                                      if c in _pa_slim.columns]

                        # Embed colour into each feature via style_function closure
                        _colours_snap = dict(_ov_colours)   # capture for closure

                        folium.GeoJson(
                            _pa_slim,
                            name='PA Network',
                            style_function=lambda feat, _c=_colours_snap: {
                                'fillColor': _c.get(
                                    feat['properties'].get('protection_class', ''),
                                    '#9E9E9E'
                                ),
                                'color':       '#333333',
                                'weight':      0.6,
                                'fillOpacity': 0.50,
                            },
                            tooltip=folium.GeoJsonTooltip(
                                fields=_tt_fields,
                                aliases=[f.replace('_', ' ').title()
                                         for f in _tt_fields],
                            ) if _tt_fields else None,
                        ).add_to(m)
                    except Exception as _e_ov:
                        st.caption(f"PA overlay unavailable: {_e_ov}")

            # Add color legend (gradient bar with labels)
            legend_html = """
            <div style="position: fixed; bottom: 50px; left: 50px; width: 220px;
                        background-color: white; border:2px solid grey; z-index:9999;
                        font-size:12px; padding: 10px">
            <p style="margin:3px 0; font-weight:bold;">Favourability Score</p>
            <div style="background: linear-gradient(to right, #d7191c, #fdae61, #ffffbf, #a6d96a, #1a9641);
                        height: 20px; margin: 5px 0;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 10px;">
                <span>0.0</span>
                <span>0.5</span>
                <span>1.0</span>
            </div>
            <p style="margin:5px 0; font-size:10px; color:#666;">
                Red = Low favourability<br>
                Green = High favourability
            </p>
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))

            # Add layer control
            folium.LayerControl().add_to(m)

            # Display map
            st_folium(m, width="100%", height=550)

        except Exception as e:
            st.error(f"Map rendering failed: {str(e)}")
            st.info("Export GeoTIFF for visualization in QGIS.")


    with subtab2:
        # =================================================================
        # STATISTICS TAB: Score distribution for all eligible pixels
        # =================================================================
        st.subheader("Score Distribution")

        if len(valid_scores) > 0:
            n_oecm = int(np.sum(oecm_mask)) if oecm_mask is not None else 0
            n_classical = int(np.sum(classical_pa_mask)) if classical_pa_mask is not None else 0

            # Histogram of ALL eligible pixel scores (not just OECM-favourable)
            hist_counts, bin_edges = np.histogram(valid_scores, bins=20, range=(0, 1))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            bin_colors = []
            for bc in bin_centers:
                if bc < 0.33:
                    bin_colors.append('#d7191c')   # Red
                elif bc < 0.67:
                    bin_colors.append('#fdae61')   # Yellow
                else:
                    bin_colors.append('#1a9641')   # Green

            fig_hist = go.Figure(data=[go.Bar(
                x=bin_centers,
                y=hist_counts,
                marker_color=bin_colors,
                name='All eligible pixels'
            )])

            export_threshold = st.session_state.get('export_threshold', 0.5)
            fig_hist.add_vline(
                x=export_threshold,
                line_dash="dash",
                line_color="black",
                line_width=2,
                annotation_text=f"Export threshold: {export_threshold:.2f}",
                annotation_position="top"
            )

            fig_hist.update_layout(
                xaxis_title="Favourability Score",
                yaxis_title="Pixel Count (eligible pixels)",
                showlegend=False,
                height=350
            )

            st.plotly_chart(fig_hist, width='stretch')

            if n_oecm == 0:
                st.warning(
                    "No pixels classified as OECM-favourable — Group C score "
                    f"(provisioning ES + compatible land use) is below the minimum "
                    "threshold (0.10) for all eligible pixels. "
                    "Check that provisioning_es and land use layers have valid values."
                )

            # Summary statistics for all eligible pixels
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Mean (eligible)", f"{np.mean(valid_scores):.3f}")
            with col_stat2:
                st.metric("Median (eligible)", f"{np.median(valid_scores):.3f}")
            with col_stat3:
                st.metric("Std Dev", f"{np.std(valid_scores):.3f}")
            with col_stat4:
                above_05 = np.sum(valid_scores >= 0.5) / len(valid_scores) * 100
                st.metric("% eligible ≥ 0.5", f"{above_05:.1f}%")

            col_stat5, col_stat6 = st.columns(2)
            with col_stat5:
                above_07 = np.sum(valid_scores >= 0.7) / len(valid_scores) * 100
                st.metric("% eligible ≥ 0.7", f"{above_07:.1f}%")
            with col_stat6:
                area_above_07_ha = np.sum(valid_scores >= 0.7) * pixel_area_ha
                st.metric("Area ≥ 0.7 (ha)", f"{area_above_07_ha:,.0f}")

        else:
            st.warning(
                "No eligible pixels found — all pixels were eliminated by Group D criteria. "
                "Increase the pressure threshold in the sidebar or verify input layers."
            )

        st.markdown("---")

        # =================================================================
        # Per-criterion contribution chart
        # =================================================================
        st.markdown("#### Per-Criterion Contribution")

        try:
            # Retrieve criterion arrays from session state if available
            # This requires storing intermediate criterion scores during MCE
            # For now, display parameter weights instead

            if params is not None:
                # Extract weights
                weights_data = []

                # Group A criteria
                if 'w_condition' in params:
                    weights_data.append({
                        'Criterion': 'Ecosystem Condition',
                        'Group': 'A — Ecological',
                        'Weight': params['w_condition'] * params['W_A'],
                        'Group_Letter': 'A'
                    })

                if 'w_regulating_es' in params:
                    weights_data.append({
                        'Criterion': 'Regulating ES',
                        'Group': 'A — Ecological',
                        'Weight': params['w_regulating_es'] * params['W_A'],
                        'Group_Letter': 'A'
                    })

                if 'w_pressure' in params:
                    weights_data.append({
                        'Criterion': 'Low Pressure',
                        'Group': 'A — Ecological',
                        'Weight': params['w_pressure'] * params['W_A'],
                        'Group_Letter': 'A'
                    })

                # Group B criteria
                if 'w_cultural_es' in params:
                    weights_data.append({
                        'Criterion': 'Cultural ES',
                        'Group': 'B — Co-benefits',
                        'Weight': params['w_cultural_es'] * params['W_B'],
                        'Group_Letter': 'B'
                    })

                # Group C criteria
                if 'w_provisioning_es' in params:
                    weights_data.append({
                        'Criterion': 'Provisioning ES',
                        'Group': 'C — Production',
                        'Weight': params['w_provisioning_es'] * params['W_C'],
                        'Group_Letter': 'C'
                    })

                if 'w_landuse_compatible' in params:
                    weights_data.append({
                        'Criterion': 'Compatible Land Use',
                        'Group': 'C — Production',
                        'Weight': params['w_landuse_compatible'] * params['W_C'],
                        'Group_Letter': 'C'
                    })

                weights_df = pd.DataFrame(weights_data)

                # Create horizontal bar chart
                color_map = {
                    'A': '#1a9641',  # Green
                    'B': '#378ADD',  # Blue
                    'C': '#F6A623'   # Orange
                }

                fig_weights = go.Figure()

                for group_letter in ['A', 'B', 'C']:
                    group_data = weights_df[weights_df['Group_Letter'] == group_letter]

                    if len(group_data) > 0:
                        fig_weights.add_trace(go.Bar(
                            y=group_data['Criterion'],
                            x=group_data['Weight'],
                            name=group_data['Group'].iloc[0],
                            orientation='h',
                            marker_color=color_map[group_letter],
                            text=group_data['Weight'].apply(lambda x: f"{x:.3f}"),
                            textposition='auto'
                        ))

                fig_weights.update_layout(
                    xaxis_title="Effective Weight (intra × inter)",
                    yaxis_title="Criterion",
                    barmode='group',
                    height=350,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                st.plotly_chart(fig_weights, width='stretch')

                st.caption(
                    "Effective weight = intra-group weight × inter-group weight. "
                    "Bars show the relative contribution of each criterion to the final score."
                )

            else:
                st.info("Parameter information not available.")

        except Exception as e:
            st.warning(f"Could not display criterion contributions: {str(e)}")

        st.markdown("---")

        # =================================================================
        # Parameter summary table
        # =================================================================
        st.markdown("#### Current Parameter Configuration")

        if params is not None:
            param_display = pd.DataFrame([
                {'Parameter': 'Aggregation Method', 'Value': params.get('method', 'N/A')},
                {'Parameter': 'Alpha (OWA)', 'Value': f"{params.get('alpha', 0.0):.2f}" if params.get('method') == 'owa' else 'N/A'},
                {'Parameter': 'W_A (Ecological)', 'Value': f"{params.get('W_A', 0.0):.2f}"},
                {'Parameter': 'W_B (Co-benefits)', 'Value': f"{params.get('W_B', 0.0):.2f}"},
                {'Parameter': 'W_C (Production)', 'Value': f"{params.get('W_C', 0.0):.2f}"},
                {'Parameter': 'Max Pressure (hab/km²)', 'Value': f"{params.get('threshold_pressure', 0.0):.0f}"},
                {'Parameter': 'Gap Bonus', 'Value': f"{params.get('gap_bonus', 0.0):.2f}"},
            ])

            st.dataframe(
                param_display,
                hide_index=True,
                width='stretch'
            )

            st.caption("Full parameter log available at bottom of page.")
            if params.get('percentile_norm'):
                st.info("Percentile normalisation active: input rasters clipped to 2nd–98th percentile before scoring.")
        else:
            st.info("Parameter information not available.")

    # =================================================================
    # SUBTAB 3: Sensitivity Analysis
    # =================================================================
    with subtab3:
        st.subheader("Weight Sensitivity Analysis")
        st.markdown(
            "Monte Carlo analysis: MCE is run **N times** with weights randomly "
            "perturbed around your chosen values (Dirichlet distribution). "
            "The **stability map** shows how often each pixel exceeds the display "
            "threshold — values close to 1.0 are robust to weight uncertainty, "
            "values close to 0.5 are ambiguous."
        )

        normalised_arrays = st.session_state.get('normalised_arrays', {})
        group_scores_sens = st.session_state.get('group_scores', {})
        eliminatory_mask  = st.session_state.get('eliminatory_mask')

        if not group_scores_sens or eliminatory_mask is None:
            st.info("Run the MCE analysis first (tab ④) to enable sensitivity analysis.")
        else:
            # ── Correlation heatmap ─────────────────────────────────────
            st.markdown("#### Criterion Inter-Correlation")
            st.caption(
                "Pearson correlation matrix between normalised criterion rasters "
                "(eligible pixels only). Pairs with |r| > 0.7 are highlighted — "
                "they may double-count the same spatial pattern."
            )
            try:
                _names = [k for k in normalised_arrays if normalised_arrays[k] is not None]
                _valid = eliminatory_mask.ravel().astype(bool)
                _mat   = np.stack(
                    [normalised_arrays[k].ravel()[_valid] for k in _names], axis=0
                )
                _nan_rows = np.any(np.isnan(_mat), axis=0)
                _mat = _mat[:, ~_nan_rows]

                if _mat.shape[1] > 10:
                    _corr = np.corrcoef(_mat)
                    import plotly.graph_objects as _go
                    _fig_corr = _go.Figure(data=_go.Heatmap(
                        z=_corr,
                        x=_names, y=_names,
                        colorscale='RdBu_r',
                        zmin=-1, zmax=1,
                        text=[[f"{v:.2f}" for v in row] for row in _corr],
                        texttemplate="%{text}",
                        colorbar=dict(title='r')
                    ))
                    _fig_corr.update_layout(
                        height=380,
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(_fig_corr, use_container_width=True)
                    # Flag high correlations
                    _issues = []
                    for _i in range(len(_names)):
                        for _j in range(_i + 1, len(_names)):
                            if abs(_corr[_i, _j]) > 0.7:
                                _issues.append(
                                    f"**{_names[_i]}** ↔ **{_names[_j]}**: "
                                    f"r = {_corr[_i, _j]:.2f}"
                                )
                    if _issues:
                        st.warning(
                            "High inter-criterion correlation detected — these pairs "
                            "may double-count the same spatial signal:\n\n" +
                            "\n\n".join(_issues)
                        )
                    else:
                        st.success("No highly correlated criterion pairs (|r| ≤ 0.7).")
                else:
                    st.warning("Too few valid pixels to compute correlations.")
            except Exception as _e:
                st.warning(f"Correlation matrix unavailable: {_e}")

            st.markdown("---")

            # ── Sensitivity run ─────────────────────────────────────────
            st.markdown("#### Stability Map")
            _map_threshold = st.session_state.get('export_threshold', 0.5)
            _n_runs        = params.get('sensitivity_runs', 200) if params else 200
            _conc          = params.get('sensitivity_concentration', 20) if params else 20
            _perturb_intra = params.get('sensitivity_perturb_intra', True) if params else True

            if st.button(
                f"Run Sensitivity Analysis ({_n_runs} runs)",
                type="primary",
                key="run_sensitivity_btn"
            ):
                try:
                    from modules.module2_favourability.sensitivity import run_sensitivity
                    _weights = {
                        'inter_group_weights': {
                            'W_A': params['W_A'], 'W_B': params['W_B'],
                            'W_C': params['W_C']
                        },
                        'group_a_weights': {
                            'ecosystem_condition': params['w_condition'],
                            'regulating_es':       params['w_regulating_es'],
                            'low_pressure':        params['w_pressure'],
                        },
                        'group_c_weights': {
                            'provisioning_es':    params['w_provisioning_es'],
                            'compatible_landuse': params['w_landuse_compatible'],
                        },
                    }

                    # Progress bar + counter
                    _prog_bar   = st.progress(0, text=f"Starting {_n_runs} Monte Carlo runs…")
                    _prog_text  = st.empty()

                    def _on_progress(current: int, total: int) -> None:
                        pct = current / total
                        _prog_bar.progress(pct, text=f"Run {current} / {total}")
                        _prog_text.caption(f"Monte Carlo iteration {current} of {total} — {pct*100:.0f}% complete")

                    _stab, _std = run_sensitivity(
                        group_scores=group_scores_sens,
                        base_weights=_weights,
                        eliminatory_mask=eliminatory_mask,
                        threshold=_map_threshold,
                        n_runs=_n_runs,
                        concentration=float(_conc),
                        perturb_intra=_perturb_intra,
                        progress_callback=_on_progress,
                    )
                    _prog_bar.empty()
                    _prog_text.empty()
                    st.session_state['sensitivity_stability'] = _stab
                    st.session_state['sensitivity_std']       = _std
                    st.success(f"Sensitivity analysis complete — {_n_runs} runs finished.")
                except Exception as _e:
                    st.error(f"Sensitivity analysis failed: {_e}")

            if 'sensitivity_stability' in st.session_state:
                _stab = st.session_state['sensitivity_stability']
                _std  = st.session_state['sensitivity_std']
                _prof = st.session_state.get('raster_profile')

                # Render stability map
                try:
                    from rasterio.warp import reproject, calculate_default_transform, Resampling
                    from rasterio.transform import array_bounds

                    _src_crs = _prof['crs']
                    _t4326, _w4326, _h4326 = calculate_default_transform(
                        _src_crs, 'EPSG:4326',
                        _prof['width'], _prof['height'],
                        *array_bounds(_prof['height'], _prof['width'], _prof['transform'])
                    )
                    _stab_4326 = np.full((_h4326, _w4326), np.nan, dtype=np.float32)
                    reproject(
                        source=_stab, destination=_stab_4326,
                        src_transform=_prof['transform'], src_crs=_src_crs,
                        dst_transform=_t4326, dst_crs='EPSG:4326',
                        resampling=Resampling.nearest,
                        src_nodata=np.nan, dst_nodata=np.nan
                    )
                    _west, _south, _east, _north = array_bounds(_h4326, _w4326, _t4326)
                    _center_lat = (_south + _north) / 2
                    _center_lon = (_west  + _east)  / 2

                    import matplotlib.cm as _mcm
                    import matplotlib.colors as _mcol
                    _cmap_s = _mcm.get_cmap('RdYlGn')
                    _norm_s = _mcol.Normalize(vmin=0.0, vmax=1.0)
                    _valid_s = ~np.isnan(_stab_4326)
                    _rgba_s  = np.zeros((_h4326, _w4326, 4), dtype=np.uint8)
                    _rgba_s[_valid_s] = (
                        _cmap_s(_norm_s(_stab_4326[_valid_s])) * 255
                    ).astype(np.uint8)
                    _img_s = Image.fromarray(_rgba_s, mode='RGBA')
                    _buf_s = io.BytesIO()
                    _img_s.save(_buf_s, format='PNG')
                    _b64_s = base64.b64encode(_buf_s.getvalue()).decode()

                    _m_s = folium.Map(
                        location=[_center_lat, _center_lon], zoom_start=8,
                        tiles=None,
                    )
                    folium.TileLayer(
                        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
                        attr='Esri, HERE, Garmin, © OpenStreetMap contributors, and the GIS user community',
                        name='Basemap',
                    ).add_to(_m_s)
                    folium.raster_layers.ImageOverlay(
                        image=f"data:image/png;base64,{_b64_s}",
                        bounds=[[_south, _west], [_north, _east]],
                        opacity=0.85, name='Stability'
                    ).add_to(_m_s)

                    _legend_s = """
                    <div style="position:fixed;bottom:50px;left:50px;width:200px;
                                background:white;border:2px solid grey;z-index:9999;
                                font-size:12px;padding:10px">
                    <b>Stability (fraction of runs ≥ threshold)</b>
                    <div style="background:linear-gradient(to right,#d7191c,#ffffbf,#1a9641);
                                height:16px;margin:5px 0;"></div>
                    <div style="display:flex;justify-content:space-between;font-size:10px">
                        <span>0% (unstable)</span><span>100% (robust)</span>
                    </div></div>"""
                    _m_s.get_root().html.add_child(folium.Element(_legend_s))
                    st_folium(_m_s, width="100%", height=480)

                    # Summary statistics
                    _el = eliminatory_mask.ravel().astype(bool)
                    _s_flat = _stab.ravel()[_el]
                    _s_flat = _s_flat[~np.isnan(_s_flat)]
                    if len(_s_flat) > 0:
                        _c1, _c2, _c3 = st.columns(3)
                        with _c1:
                            st.metric("Highly stable (≥80%)",
                                      f"{(_s_flat >= 0.8).mean()*100:.1f}% of pixels")
                        with _c2:
                            st.metric("Ambiguous (40–60%)",
                                      f"{((_s_flat >= 0.4) & (_s_flat < 0.6)).mean()*100:.1f}% of pixels")
                        with _c3:
                            st.metric("Unstable (<20%)",
                                      f"{(_s_flat < 0.2).mean()*100:.1f}% of pixels")
                except Exception as _e:
                    st.error(f"Stability map rendering failed: {_e}")

    # =================================================================
    # SUBTAB 4: Candidate Sites
    # =================================================================
    with subtab4:
        st.subheader("Candidate OECM Sites")
        st.markdown(
            "Delineates spatially contiguous patches above the score threshold, "
            "filters by the Minimum Mapping Unit, and ranks them by a composite "
            "of mean score, gap overlap, patch area and proximity to existing PAs."
        )

        _score  = st.session_state.get('score_array')
        _prof   = st.session_state.get('raster_profile')
        _el_msk = st.session_state.get('eliminatory_mask')

        if _score is None or _prof is None:
            st.info("Run the MCE analysis first to enable candidate site delineation.")
        else:
            _threshold_patch = st.slider(
                "Score threshold for patch delineation",
                min_value=0.0, max_value=1.0,
                value=st.session_state.get('export_threshold', 0.5),
                step=0.05, key='patch_threshold_slider'
            )
            _mmu = params.get('mmu_ha', 100) if params else 100
            st.caption(
                f"MMU = {_mmu} ha (adjust in sidebar → 6d). "
                "Patches smaller than this area are discarded."
            )

            if st.button("Delineate Candidate Sites", type="primary",
                         key="delineate_btn"):
                with st.spinner("Delineating patches and computing attributes…"):
                    try:
                        from modules.module2_favourability.patch_delineation import (
                            delineate_patches
                        )
                        _pa_gdf    = st.session_state.get('pa_gdf')
                        _gap_lyrs  = st.session_state.get('gap_layers', {})
                        _strict_gp = _gap_lyrs.get('strict_gaps')

                        _sites = delineate_patches(
                            score_array=_score,
                            profile=_prof,
                            threshold=_threshold_patch,
                            mmu_ha=float(_mmu),
                            pa_gdf=_pa_gdf,
                            strict_gaps_gdf=_strict_gp,
                        )
                        st.session_state['candidate_sites'] = _sites
                        if len(_sites) == 0:
                            st.warning(
                                "No patches found above threshold with area ≥ MMU. "
                                "Try lowering the threshold or the MMU."
                            )
                        else:
                            st.success(
                                f"Delineated **{len(_sites)}** candidate OECM sites "
                                f"(MMU = {_mmu} ha, threshold = {_threshold_patch:.2f})."
                            )
                    except Exception as _e:
                        st.error(f"Patch delineation failed: {_e}")

            _sites = st.session_state.get('candidate_sites')
            if _sites is not None and len(_sites) > 0:
                # ── Ranked table ────────────────────────────────────────
                st.markdown("#### Ranked Candidate Sites")
                _disp = _sites[[
                    'patch_id', 'area_ha', 'mean_score', 'max_score',
                    'compactness', 'dist_to_pa_km', 'gap_overlap_pct', 'rank_score'
                ]].copy()
                _disp.columns = [
                    'Rank', 'Area (ha)', 'Mean Score', 'Max Score',
                    'Compactness', 'Dist to PA (km)', 'Gap Overlap (%)', 'Rank Score'
                ]
                st.dataframe(
                    _disp.style.background_gradient(
                        subset=['Rank Score'], cmap='YlGn'
                    ),
                    hide_index=True, use_container_width=True
                )
                st.caption(
                    "Rank Score = 50% mean score + 20% gap overlap + "
                    "20% log(area) + 10% PA proximity. "
                    "Compactness: 1.0 = circular patch (Polsby-Popper index)."
                )

                # ── Site map ────────────────────────────────────────────
                st.markdown("#### Site Map")
                try:
                    import geopandas as gpd
                    _sites_4326 = _sites.to_crs('EPSG:4326')
                    _centroid   = _sites_4326.geometry.union_all().centroid
                    _m_sites    = folium.Map(
                        location=[_centroid.y, _centroid.x], zoom_start=9,
                        tiles=None,
                    )
                    folium.TileLayer(
                        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
                        attr='Esri, HERE, Garmin, © OpenStreetMap contributors, and the GIS user community',
                        name='Basemap',
                    ).add_to(_m_sites)
                    folium.GeoJson(
                        _sites_4326,
                        name='Candidate Sites',
                        style_function=lambda f: {
                            'fillColor': '#2E7D32', 'color': '#1B5E20',
                            'weight': 1.5, 'fillOpacity': 0.45
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=['patch_id', 'area_ha', 'mean_score',
                                    'gap_overlap_pct', 'rank_score'],
                            aliases=['Site #', 'Area (ha)', 'Mean Score',
                                     'Gap Overlap (%)', 'Rank Score']
                        )
                    ).add_to(_m_sites)
                    folium.LayerControl().add_to(_m_sites)
                    st_folium(_m_sites, width="100%", height=480)
                except Exception as _e:
                    st.warning(f"Site map unavailable: {_e}")

                # ── GeoPackage download ──────────────────────────────────
                st.markdown("#### Download Candidate Sites")
                try:
                    import tempfile, zipfile
                    from pathlib import Path as _Path
                    with tempfile.TemporaryDirectory() as _tmp:
                        _gpkg = _Path(_tmp) / "candidate_oecm_sites.gpkg"
                        _sites.to_file(str(_gpkg), driver='GPKG')
                        with open(_gpkg, 'rb') as _f:
                            _gpkg_bytes = _f.read()
                    st.download_button(
                        "Download GeoPackage",
                        data=_gpkg_bytes,
                        file_name="candidate_oecm_sites.gpkg",
                        mime="application/geopackage+sqlite3",
                    )
                except Exception as _e:
                    st.caption(f"GeoPackage export unavailable: {_e}")

    st.markdown("---")

    # ===================================================================
    # Row 4: Export panel
    # ===================================================================
    st.subheader("Export & Reporting")

    # Threshold slider for polygon export
    export_threshold = st.slider(
        "Favourability threshold for polygon export",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Pixels above this threshold will be vectorised and exported as shapefile"
    )

    # Store threshold in session state for histogram display
    st.session_state['export_threshold'] = export_threshold

    col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)

    # Import export functions
    from modules.module2_favourability import export as export_module

    with col_exp1:
        if st.button("Export GeoTIFF"):
            try:
                with st.spinner("Exporting GeoTIFF..."):
                    # Create temp file, close it immediately so rasterio can open it on Windows
                    import tempfile as _tf
                    _tmp = _tf.NamedTemporaryFile(delete=False, suffix='.tif')
                    _tmp_path = _tmp.name
                    _tmp.close()

                    export_module.export_geotiff(
                        array=score_array,
                        profile=profile,
                        output_path=_tmp_path
                    )
                    with open(_tmp_path, 'rb') as f:
                        geotiff_bytes = f.read()
                    try:
                        import os as _os
                        _os.unlink(_tmp_path)
                    except Exception:
                        pass

                st.download_button(
                    label="Download GeoTIFF",
                    data=geotiff_bytes,
                    file_name="favourability_scores.tif",
                    mime="image/tiff"
                )
                st.success("GeoTIFF ready for download!")

            except Exception as e:
                st.error(f"GeoTIFF export failed: {str(e)}")

    with col_exp2:
        if st.button("Export Shapefile (ZIP)"):
            try:
                with st.spinner("Exporting shapefile..."):
                    import zipfile

                    # Create temporary directory for shapefile components
                    with tempfile.TemporaryDirectory() as tmpdir:
                        shp_path = Path(tmpdir) / "favourable_zones.shp"

                        export_module.export_shapefile(
                            score_array=score_array,
                            profile=profile,
                            threshold=export_threshold,
                            output_path=str(shp_path)
                        )

                        # Zip all shapefile components
                        zip_path = Path(tmpdir) / "favourable_zones.zip"
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                                component = shp_path.with_suffix(ext)
                                if component.exists():
                                    zipf.write(component, component.name)

                        # Read zip for download
                        with open(zip_path, 'rb') as f:
                            zip_bytes = f.read()

                        st.download_button(
                            label="Download Shapefile (ZIP)",
                            data=zip_bytes,
                            file_name="favourable_zones.zip",
                            mime="application/zip"
                        )

                        st.success(f"Shapefile ready (threshold: {export_threshold:.2f})!")

            except Exception as e:
                st.error(f"Shapefile export failed: {str(e)}")

    with col_exp3:
        if st.button("Export CSV Stats"):
            try:
                with st.spinner("Exporting CSV statistics..."):
                    # Build comprehensive stats table
                    pa_gdf = st.session_state.get('pa_gdf')
                    wdpa_area_ha = 0.0
                    n_pa_sites = 0
                    if pa_gdf is not None:
                        try:
                            wdpa_area_ha = float(pa_gdf.geometry.union_all().area) / 10000.0
                            n_pa_sites = len(pa_gdf)
                        except Exception:
                            pass

                    stats_df = pd.DataFrame([
                        {'metric': 'Territory grid area (ha)',
                         'value': f"{territory_area_ha:.0f}"},
                        {'metric': 'Eligible area — passed Group D (ha)',
                         'value': f"{eligible_area_ha:.0f}"},
                        {'metric': 'Eliminated — high pressure or incompatible LU (ha)',
                         'value': f"{n_eliminated * pixel_area_ha:.0f}"},
                        {'metric': 'OECM favourable area — MCE Group C ≥ 0.10 (ha)',
                         'value': f"{oecm_area_ha:.0f}"},
                        {'metric': 'Low use-function area — MCE Group C < 0.10 (ha)',
                         'value': f"{classical_pa_area_ha:.0f}"},
                        {'metric': 'Median favourability score (eligible pixels)',
                         'value': f"{median_score:.3f}"},
                        {'metric': 'Existing WDPA protected area — union (ha)',
                         'value': f"{wdpa_area_ha:.0f}" if wdpa_area_ha > 0 else 'Not loaded'},
                        {'metric': 'Existing WDPA sites (count)',
                         'value': str(n_pa_sites) if n_pa_sites > 0 else 'Not loaded'},
                    ])

                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
                        export_module.export_csv_stats(
                            stats_df=stats_df,
                            output_path=tmp.name
                        )

                        # Read file for download
                        with open(tmp.name, 'r') as f:
                            csv_data = f.read()

                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name="favourability_statistics.csv",
                            mime="text/csv"
                        )

                        st.success("CSV ready for download!")

            except Exception as e:
                st.error(f"CSV export failed: {str(e)}")

    with col_exp4:
        if st.button("Generate PDF Report"):
            try:
                with st.spinner("Generating PDF report..."):
                    # Create temporary map image from score_array
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                        import matplotlib.cm as mcm
                        _cmap = mcm.get_cmap('RdYlGn')
                        _norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
                        _rgba = np.zeros((*score_array.shape, 4), dtype=np.uint8)
                        _vm = ~np.isnan(score_array)
                        _rgba[_vm] = (_cmap(_norm(score_array[_vm])) * 255).astype(np.uint8)
                        fig_map, ax = plt.subplots(figsize=(10, 8))
                        ax.imshow(_rgba)
                        ax.axis('off')
                        plt.savefig(tmp_img.name, bbox_inches='tight', dpi=150)
                        plt.close()

                        # Build comprehensive stats table for PDF
                        pa_gdf = st.session_state.get('pa_gdf')
                        wdpa_area_ha_pdf = 0.0
                        n_pa_sites_pdf = 0
                        if pa_gdf is not None:
                            try:
                                wdpa_area_ha_pdf = float(pa_gdf.geometry.union_all().area) / 10000.0
                                n_pa_sites_pdf = len(pa_gdf)
                            except Exception:
                                pass

                        stats_df = pd.DataFrame([
                            {'Metric': 'Territory grid area (ha)',
                             'Value': f"{territory_area_ha:.0f}"},
                            {'Metric': 'Eligible area — passed Group D (ha)',
                             'Value': f"{eligible_area_ha:.0f}"},
                            {'Metric': 'Eliminated — pressure / incompatible LU (ha)',
                             'Value': f"{n_eliminated * pixel_area_ha:.0f}"},
                            {'Metric': 'OECM favourable area — MCE Group C ≥ 0.10 (ha)',
                             'Value': f"{oecm_area_ha:.0f}"},
                            {'Metric': 'Low use-function — MCE Group C < 0.10 (ha)',
                             'Value': f"{classical_pa_area_ha:.0f}"},
                            {'Metric': 'Median favourability score (eligible pixels)',
                             'Value': f"{median_score:.3f}"},
                            {'Metric': 'Existing WDPA protected area — union (ha)',
                             'Value': f"{wdpa_area_ha_pdf:.0f}" if wdpa_area_ha_pdf > 0 else 'Not loaded'},
                            {'Metric': 'Existing WDPA sites (count)',
                             'Value': str(n_pa_sites_pdf) if n_pa_sites_pdf > 0 else 'Not loaded'},
                        ])

                        # Add timestamp and spec version to params
                        if params is not None:
                            params_full = params.copy()
                            params_full['timestamp'] = datetime.now().isoformat()
                            params_full['spec_version'] = 'v0.1'
                        else:
                            params_full = {
                                'timestamp': datetime.now().isoformat(),
                                'spec_version': 'v0.1'
                            }

                        # Generate PDF
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                            export_module.generate_pdf_report(
                                map_image_path=tmp_img.name,
                                stats_df=stats_df,
                                parameters=params_full,
                                output_path=tmp_pdf.name
                            )

                            # Read PDF for download
                            with open(tmp_pdf.name, 'rb') as f:
                                pdf_bytes = f.read()

                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_bytes,
                                file_name="favourability_report.pdf",
                                mime="application/pdf"
                            )

                            st.success("PDF report ready for download!")

            except ImportError:
                st.error("PDF generation requires reportlab. Install with: pip install reportlab")
            except Exception as e:
                st.error(f"PDF generation failed: {str(e)}")

    st.markdown("---")

    # ===================================================================
    # Footer: Reproducibility note
    # ===================================================================
    st.subheader("Reproducibility & Parameter Log")

    with st.expander("View Current Parameter Configuration (JSON)"):
        # Use params passed to function or retrieve from session state
        params_to_display = params if params is not None else st.session_state.get('parameters')

        if params_to_display:
            # Add timestamp and version info if not present
            params_full = params_to_display.copy()
            if 'timestamp' not in params_full:
                params_full['timestamp'] = datetime.now().isoformat()
            if 'spec_version' not in params_full:
                params_full['spec_version'] = 'v0.1'

            def _json_safe(obj):
                """Convert non-serializable objects to strings."""
                if hasattr(obj, 'wkt'):  # shapely geometry
                    return obj.wkt
                if hasattr(obj, 'isoformat'):  # datetime
                    return obj.isoformat()
                return str(obj)

            json_str = json.dumps(params_full, indent=2, default=_json_safe)
            st.json(json.loads(json_str))
            st.download_button(
                label="Download Parameters (JSON)",
                data=json_str,
                file_name="mce_parameters.json",
                mime="application/json"
            )
        else:
            st.info("Parameter configuration not available. Run analysis to generate parameter log.")

    st.caption(
        "All parameters used in MCE computation are logged for full reproducibility. "
        "Include this JSON file with exported outputs to document analysis settings."
    )


@st.fragment
def render_module2_tab():
    """
    Orchestrate Module 2 — load/align rasters, run the MCE computation, and
    render results. Decorated with @st.fragment so that interactions inside
    other tabs (e.g. Module 1's "Run Gap Analysis") no longer force this
    tab's full raster alignment + MCE recomputation on every app rerun.

    All inputs (parameters, raster paths, gap layers, PA data) are read
    fresh from st.session_state on every call, since a fragment-scoped
    rerun does not re-execute app.py's top-level script.
    """
    from ui.sidebar import load_settings

    st.header("Module 2 — OECM Favourability Analysis")

    parameters = st.session_state.get('parameters', {})

    # Check if data has been uploaded
    data_ready_module2 = st.session_state.get('data_ready_module2', False)

    if not data_ready_module2:
        st.info(
            "Upload all 6 criterion rasters in the **① Data Upload** tab first."
        )
    else:
        # Retrieve raster paths from session state
        raster_paths = st.session_state.get('criterion_raster_paths', {})

        # Check if weights sum to 1.0
        weight_sum = parameters.get('W_A', 0) + parameters.get('W_B', 0) + parameters.get('W_C', 0)
        weights_valid = abs(weight_sum - 1.0) < 0.001

        st.markdown("---")

        # -------------------------------------------------------
        # Cached load + align: only re-runs when files or study
        # area change. Changing weights/method/threshold does NOT
        # trigger re-alignment (major speed-up on every re-run).
        # -------------------------------------------------------
        @st.cache_data(show_spinner=False)
        def _load_and_align(paths_frozen, study_area_wkt, resolution, crs):
            """Load and align all rasters. Cache key = paths + study area.

            When a study area is available, each raster is read via a windowed
            read limited to the study area bounding box before alignment.
            This avoids loading full EU-wide extents into memory.
            """
            from shapely.wkt import loads as _wkt_loads
            from modules.module2_favourability import raster_preprocessing as _rp
            sa_geom = _wkt_loads(study_area_wkt) if study_area_wkt else None
            raster_dict = {}
            for name, path in paths_frozen:
                if sa_geom is not None:
                    arr, prof = _rp.load_raster_windowed(
                        path, clip_geom=sa_geom, geom_crs=crs
                    )
                else:
                    arr, prof = _rp.load_raster(path)
                raster_dict[name] = (arr, prof)
            return _rp.align_rasters(
                raster_dict,
                study_area_geom=sa_geom,
                resolution=resolution,
                crs=crs
            )

        # Build hashable cache key from current raster paths + study area
        _settings = load_settings()
        target_resolution = _settings.get('resolution_m', 100.0)
        target_crs = _settings.get('crs', 'EPSG:3035')
        study_area_geom = parameters.get('study_area_geometry')

        layer_order = [
            'ecosystem_condition', 'regulating_es',
            'anthropogenic_pressure', 'cultural_es',
            'provisioning_es', 'landuse'
        ]
        paths_frozen = tuple((k, raster_paths[k]) for k in layer_order)
        study_area_wkt = study_area_geom.wkt if study_area_geom else ''

        # Check whether aligned arrays are already in session state for the
        # current set of paths (so we know if Load & Align has been run)
        aligned_key = st.session_state.get('_aligned_key')
        current_key = (paths_frozen, study_area_wkt, target_resolution, target_crs)
        rasters_aligned = aligned_key == current_key

        # ------------------------------------------------------------------
        # Step 1 — Load & Align button (only needed when files/area change)
        # ------------------------------------------------------------------
        load_col1, load_col2, load_col3 = st.columns([1, 2, 1])
        with load_col2:
            load_button = st.button(
                "Load & Align Rasters",
                type="secondary" if rasters_aligned else "primary",
                use_container_width=True,
                help="Re-run only when you change raster files or the study area."
            )

        if load_button:
            with st.spinner("Loading and aligning raster layers (cached after first run)..."):
                try:
                    from modules.module2_favourability import mce_engine  # noqa: F401 — ensure importable
                    aligned = _load_and_align(
                        paths_frozen, study_area_wkt, target_resolution, target_crs
                    )
                    # Store aligned arrays and profile in session state
                    st.session_state['_aligned_arrays'] = {
                        k: aligned[k][0] for k in layer_order
                    }
                    st.session_state['_aligned_profile'] = aligned['ecosystem_condition'][1]
                    st.session_state['_aligned_key'] = current_key
                    # Clear previous MCE results so they are recomputed below
                    for _k in ('score_array', 'oecm_mask', 'classical_pa_mask',
                               'eliminatory_mask', 'raster_profile'):
                        st.session_state.pop(_k, None)
                    st.success("Rasters loaded and aligned!")
                    rasters_aligned = True
                except Exception as e:
                    st.error(f"Raster loading/alignment failed: {str(e)}")
                    logger.exception("Raster loading/alignment error:")

        # ------------------------------------------------------------------
        # Step 2 — MCE computation: auto-runs on every rerun when aligned
        # ------------------------------------------------------------------
        if rasters_aligned:
            if not weights_valid:
                st.error(
                    f"Inter-group weights must sum to 1.0 (current sum: {weight_sum:.3f})"
                )
            else:
                from modules.module2_favourability import mce_engine

                aligned_arrays  = st.session_state['_aligned_arrays']
                reference_profile = st.session_state['_aligned_profile']

                # Prepare weight structure — normalize inter-group weights to exactly
                # 1.0 to avoid floating-point drift triggering the engine's atol=1e-6 check.
                _wa, _wb, _wc = parameters['W_A'], parameters['W_B'], parameters['W_C']
                _wsum = _wa + _wb + _wc
                weights = {
                    'inter_group_weights': {
                        'W_A': _wa / _wsum,
                        'W_B': _wb / _wsum,
                        'W_C': _wc / _wsum,
                    },
                    'group_a_weights': {
                        'ecosystem_condition': parameters['w_condition'],
                        'regulating_es': parameters['w_regulating_es'],
                        'low_pressure': parameters['w_pressure']
                    },
                    'group_b_weights': {
                        'cultural_es': parameters['w_cultural_es']
                    },
                    'group_c_weights': {
                        'provisioning_es': parameters['w_provisioning_es'],
                        'compatible_landuse': parameters['w_landuse_compatible']
                    }
                }

                # Build gap mask from Module 1 gap layers (if available)
                gap_mask = None
                gap_bonus_val = parameters.get('gap_bonus', 0.0)
                _gap_layers_marker = None
                if gap_bonus_val > 0.0 and 'gap_layers' in st.session_state:
                    try:
                        import numpy as np
                        from rasterio.features import rasterize as _rasterize

                        gap_layers = st.session_state['gap_layers']
                        _gap_layers_marker = id(gap_layers)
                        target_crs_obj = reference_profile['crs']
                        target_shape = (reference_profile['height'],
                                        reference_profile['width'])
                        target_transform = reference_profile['transform']

                        all_geoms = []
                        for _gkey in ('strict_gaps', 'qualitative_gaps'):
                            gdf = gap_layers.get(_gkey)
                            if gdf is not None and len(gdf) > 0:
                                reprojected = gdf.to_crs(target_crs_obj)
                                valid = reprojected[
                                    ~reprojected.geometry.is_empty
                                    & reprojected.geometry.notnull()
                                ]
                                all_geoms.extend(
                                    (geom, 1) for geom in valid.geometry
                                )

                        if all_geoms:
                            gap_raster = _rasterize(
                                shapes=all_geoms,
                                out_shape=target_shape,
                                transform=target_transform,
                                fill=0,
                                dtype='uint8'
                            )
                            gap_mask = gap_raster.astype(bool)
                            logger.info(
                                f"Gap mask built: {np.sum(gap_mask)} gap pixels "
                                f"out of {gap_mask.size}"
                            )
                        else:
                            logger.info("No gap geometries found, skipping gap bonus")
                    except Exception as e:
                        logger.warning(f"Failed to build gap mask: {e}")
                        gap_mask = None

                # Inform the user if Module 1's gap layers were updated since
                # the last time this tab computed the gap bonus — since this
                # tab is now its own fragment, changes made in Module 1's tab
                # no longer trigger an automatic recompute here.
                if gap_bonus_val > 0.0 and 'gap_layers' in st.session_state:
                    _last_marker = st.session_state.get('_gap_layers_marker_used')
                    if _last_marker is not None and _last_marker != _gap_layers_marker:
                        st.info(
                            "Gap layers were updated in Module 1 since the last computation. "
                            "The gap bonus below has already been recalculated using the "
                            "latest gap layers."
                        )
                    st.session_state['_gap_layers_marker_used'] = _gap_layers_marker

                # ── PA proximity raster (distance transform from WDPA) ──────
                pa_proximity_raster = None
                if parameters.get('proximity_bonus', 0.0) > 0.0:
                    try:
                        import numpy as np
                        from scipy.ndimage import distance_transform_edt as _edt
                        from rasterio.features import rasterize as _rasterize2
                        pa_gdf_prox = st.session_state.get('pa_gdf')
                        if pa_gdf_prox is not None and len(pa_gdf_prox) > 0:
                            _pa_repr = pa_gdf_prox.to_crs(reference_profile['crs'])
                            _pa_bin  = _rasterize2(
                                shapes=[(g, 1) for g in _pa_repr.geometry
                                        if g and not g.is_empty],
                                out_shape=(reference_profile['height'],
                                           reference_profile['width']),
                                transform=reference_profile['transform'],
                                fill=0, dtype='uint8'
                            )
                            pixel_size_m = abs(reference_profile['transform'][0])
                            # distance_transform_edt returns distance in pixels
                            pa_proximity_raster = _edt(
                                1 - _pa_bin
                            ).astype(np.float32) * pixel_size_m
                            logger.info("PA proximity raster computed")
                    except Exception as _e:
                        logger.warning(f"PA proximity raster failed: {_e}")

                try:
                    with st.spinner("Computing favourability scores…"):
                        results = mce_engine.compute_favourability(
                            ecosystem_condition=aligned_arrays['ecosystem_condition'],
                            regulating_es=aligned_arrays['regulating_es'],
                            cultural_es=aligned_arrays['cultural_es'],
                            provisioning_es=aligned_arrays['provisioning_es'],
                            anthropogenic_pressure=aligned_arrays['anthropogenic_pressure'],
                            landuse=aligned_arrays['landuse'],
                            weights=weights,
                            method=parameters['method'],
                            alpha=parameters['alpha'],
                            threshold_pressure=parameters.get('threshold_pressure', 150.0),
                            gap_bonus=gap_bonus_val,
                            gap_mask=gap_mask,
                            percentile_norm=parameters.get('percentile_norm', False),
                            proximity_bonus=parameters.get('proximity_bonus', 0.0),
                            proximity_decay_km=parameters.get('proximity_decay_km', 10.0),
                            pa_proximity_raster=pa_proximity_raster,
                        )

                    score_out        = results['score'].copy()
                    oecm_mask_out    = results['oecm_mask'].copy()
                    classical_out    = results['classical_pa_mask'].copy()
                    elim_mask_out    = results['eliminatory_mask'].copy()

                    # ── Optional: mask out pixels inside existing PAs ────────
                    if parameters.get('exclude_pa_pixels') and 'pa_gdf' in st.session_state:
                        import numpy as np
                        from rasterio.features import rasterize as _rasterize_pa
                        _pa_gdf_ex  = st.session_state['pa_gdf']
                        _ex_classes = parameters.get('exclude_pa_classes', [])
                        if _ex_classes and 'protection_class' in _pa_gdf_ex.columns:
                            _pa_sel = _pa_gdf_ex[
                                _pa_gdf_ex['protection_class'].isin(_ex_classes)
                            ]
                        else:
                            _pa_sel = _pa_gdf_ex
                        if len(_pa_sel) > 0:
                            try:
                                _pa_repr = _pa_sel.to_crs(reference_profile['crs'])
                                _geoms   = [
                                    (g, 1) for g in _pa_repr.geometry
                                    if g is not None and not g.is_empty
                                ]
                                if _geoms:
                                    _pa_mask = _rasterize_pa(
                                        shapes=_geoms,
                                        out_shape=(reference_profile['height'],
                                                   reference_profile['width']),
                                        transform=reference_profile['transform'],
                                        fill=0, dtype='uint8'
                                    ).astype(bool)
                                    score_out[_pa_mask]     = np.nan
                                    oecm_mask_out[_pa_mask] = False
                                    classical_out[_pa_mask] = False
                                    elim_mask_out[_pa_mask] = False
                                    logger.info(
                                        "PA exclusion: masked %d pixels from %d %s",
                                        int(_pa_mask.sum()), len(_pa_sel),
                                        str(_ex_classes)
                                    )
                            except Exception as _e:
                                logger.warning("PA exclusion rasterization failed: %s", _e)

                    st.session_state['score_array']        = score_out
                    st.session_state['oecm_mask']          = oecm_mask_out
                    st.session_state['classical_pa_mask']  = classical_out
                    st.session_state['eliminatory_mask']   = elim_mask_out
                    st.session_state['raster_profile']     = reference_profile
                    st.session_state['normalised_arrays']  = results.get('normalised_arrays', {})
                    st.session_state['group_scores']       = results.get('group_scores', {})

                    render_tab_module2(
                        score_array=score_out,
                        oecm_mask=oecm_mask_out,
                        classical_pa_mask=classical_out,
                        eliminatory_mask=elim_mask_out,
                        profile=reference_profile,
                        params=parameters
                    )

                except Exception as e:
                    st.error(f"MCE computation failed: {str(e)}")
                    logger.exception("MCE computation error:")

        else:
            st.info(
                "Click **Load & Align Rasters** above to initialise the analysis. "
                "After that, scores update automatically when you adjust weights."
            )
