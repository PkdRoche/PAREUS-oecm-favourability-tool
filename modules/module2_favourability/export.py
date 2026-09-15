"""Export functionality for favourability results."""

import io
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.transform import array_bounds
import geopandas as gpd
from shapely.geometry import shape

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe inside Streamlit's script thread
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def export_geotiff(
    array: np.ndarray,
    profile: dict,
    output_path: str
) -> None:
    """
    Save favourability score array as GeoTIFF.

    Parameters
    ----------
    array : np.ndarray
        Favourability score array [0-1]. Shape: (height, width).
    profile : dict
        Rasterio profile (metadata) containing:
        - crs: coordinate reference system
        - transform: affine transformation matrix
        - width, height: raster dimensions
        - dtype: data type (typically float32)
        - nodata: NoData value
    output_path : str
        Path for output GeoTIFF file.

    Raises
    ------
    ValueError
        If array shape does not match profile dimensions.
    OSError
        If output file cannot be written.

    Notes
    -----
    - Output GeoTIFF uses LZW compression
    - NoData values are preserved from input profile
    - All metadata (CRS, transform, etc.) are copied from profile

    Examples
    --------
    >>> export_geotiff(
    ...     array=favourability_scores,
    ...     profile=reference_raster.profile,
    ...     output_path='outputs/favourability.tif'
    ... )
    """
    # Verify array shape matches profile
    if array.shape != (profile['height'], profile['width']):
        raise ValueError(
            f"Array shape {array.shape} does not match profile dimensions "
            f"({profile['height']}, {profile['width']})"
        )

    logger.info(f"Exporting GeoTIFF to {output_path}...")

    # Update profile for output
    output_profile = profile.copy()
    output_profile.update({
        'dtype': 'float32',
        'count': 1,
        'compress': 'lzw'
    })

    # Write to file
    with rasterio.open(output_path, 'w', **output_profile) as dst:
        dst.write(array.astype('float32'), 1)

    logger.info(f"GeoTIFF exported successfully: {output_path}")


def export_shapefile(
    score_array: np.ndarray,
    profile: dict,
    threshold: float,
    output_path: str
) -> None:
    """
    Vectorise pixels above threshold, dissolve, export as shapefile.

    Parameters
    ----------
    score_array : np.ndarray
        Favourability score array [0-1]. Shape: (height, width).
    profile : dict
        Rasterio profile with crs, transform, width, height.
    threshold : float
        Minimum favourability score for inclusion (e.g., 0.6).
        Pixels with score >= threshold are vectorised.
    output_path : str
        Path for output shapefile (.shp extension).

    Raises
    ------
    ValueError
        If array shape does not match profile dimensions.
    OSError
        If output file cannot be written.

    Notes
    -----
    - Creates binary mask from threshold: 1 = favourable, 0 = not favourable
    - Vectorises contiguous regions of favourable pixels
    - Dissolves all polygons into single MultiPolygon
    - Output CRS matches input profile CRS

    Examples
    --------
    >>> export_shapefile(
    ...     score_array=favourability_scores,
    ...     profile=reference_raster.profile,
    ...     threshold=0.6,
    ...     output_path='outputs/favourable_zones.shp'
    ... )
    """
    # Verify array shape
    if score_array.shape != (profile['height'], profile['width']):
        raise ValueError(
            f"Array shape {score_array.shape} does not match profile dimensions "
            f"({profile['height']}, {profile['width']})"
        )

    logger.info(f"Vectorising pixels with score >= {threshold}...")

    # Create binary mask
    mask = (score_array >= threshold).astype('uint8')

    # Vectorise using rasterio.features.shapes
    geoms = []
    values = []

    for geom_dict, value in shapes(mask, transform=profile['transform']):
        if value == 1:  # Only keep favourable pixels
            geoms.append(shape(geom_dict))
            values.append(score_array[mask == 1].mean())  # Mean score for this polygon

    if not geoms:
        logger.warning(f"No pixels found with score >= {threshold}. Exporting empty shapefile.")
        # Create empty GeoDataFrame
        gdf = gpd.GeoDataFrame(
            columns=['score_mean', 'geometry'],
            crs=profile['crs']
        )
    else:
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {'score_mean': values},
            geometry=geoms,
            crs=profile['crs']
        )

        # Dissolve all polygons (optional, for simplified output)
        # Comment out if individual polygons are preferred
        # gdf = gdf.dissolve().reset_index(drop=True)

    # Export to shapefile
    gdf.to_file(output_path)

    logger.info(f"Shapefile exported successfully: {output_path} ({len(gdf)} polygons)")


def export_csv_stats(
    stats_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Export territorial unit statistics as CSV.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with territorial unit statistics.
        Expected columns: unit_name, mean_score, oecm_area_ha, pct_oecm, etc.
    output_path : str
        Path for output CSV file.

    Raises
    ------
    OSError
        If output file cannot be written.

    Notes
    -----
    - CSV uses UTF-8 encoding
    - Index is not exported

    Examples
    --------
    >>> export_csv_stats(
    ...     stats_df=zonal_statistics,
    ...     output_path='outputs/unit_statistics.csv'
    ... )
    """
    logger.info(f"Exporting CSV statistics to {output_path}...")

    stats_df.to_csv(output_path, index=False, encoding='utf-8')

    logger.info(f"CSV exported successfully: {output_path} ({len(stats_df)} rows)")


def _nice_scalebar_km(extent_m: float) -> float:
    """Round scale-bar length (km) targeting ~1/4 of the map's horizontal extent."""
    target = max(extent_m / 1000.0 / 4.0, 0.001)
    magnitude = 10 ** np.floor(np.log10(target))
    for mult in (1, 2, 5, 10):
        candidate = mult * magnitude
        if candidate >= target:
            return candidate
    return 10 * magnitude


def _add_scalebar(ax, bounds: tuple) -> None:
    """Draw a simple linear scale bar (map units = metres, e.g. EPSG:3035)."""
    left, bottom, right, top = bounds
    km = _nice_scalebar_km(right - left)
    bar_m = km * 1000.0
    x0 = left + 0.05 * (right - left)
    y0 = bottom + 0.05 * (top - bottom)
    ax.plot([x0, x0 + bar_m], [y0, y0], color='black', linewidth=3, solid_capstyle='butt')
    ax.text(x0 + bar_m / 2, y0 + 0.015 * (top - bottom), f"{km:g} km",
            ha='center', va='bottom', fontsize=9)


def _raster_map_figure(
    array: np.ndarray,
    profile: dict,
    title: str,
    cmap_name: str = 'RdYlGn',
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_label: str = 'Score',
    cbar_ticks: Optional[list] = None,
    pa_gdf=None,
    figsize: tuple = (9, 7),
):
    """Static matplotlib map of a continuous raster with colorbar, title and scale bar."""
    left, bottom, right, top = array_bounds(profile['height'], profile['width'], profile['transform'])
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color='#e8e8e8')
    masked = np.ma.masked_invalid(array)
    im = ax.imshow(masked, extent=(left, right, bottom, top), origin='upper',
                    cmap=cmap, vmin=vmin, vmax=vmax)

    if pa_gdf is not None and len(pa_gdf) > 0:
        try:
            pa_gdf.boundary.plot(ax=ax, color='#333333', linewidth=0.7)
        except Exception as e:
            logger.warning(f"Could not overlay PA network on report map: {e}")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=cbar_ticks)
    cbar.set_label(cbar_label, fontsize=10)
    _add_scalebar(ax, (left, bottom, right, top))
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def _histogram_figure(valid_scores: np.ndarray, threshold: float, figsize: tuple = (9, 3.5)):
    """Static score-distribution histogram, colour-banded red/yellow/green."""
    fig, ax = plt.subplots(figsize=figsize)
    counts, edges = np.histogram(valid_scores, bins=20, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    bar_colors = ['#d7191c' if c < 0.33 else '#fdae61' if c < 0.67 else '#1a9641' for c in centers]
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color=bar_colors)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5)
    ax.text(threshold, ax.get_ylim()[1] * 0.97, f" threshold={threshold:.2f}",
            fontsize=8, va='top')
    ax.set_xlabel("Favourability score")
    ax.set_ylabel("Pixel count (eligible)")
    ax.set_title("Score Distribution", fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def _weights_bar_figure(weights_df: pd.DataFrame, figsize: tuple = (9, 3.5)):
    """Static horizontal bar chart of per-criterion effective weights."""
    color_map = {'A': '#1a9641', 'B': '#378ADD', 'C': '#F6A623'}
    fig, ax = plt.subplots(figsize=figsize)
    bar_colors = weights_df['Group_Letter'].map(color_map)
    ax.barh(weights_df['Criterion'], weights_df['Weight'], color=bar_colors)
    ax.invert_yaxis()
    ax.set_xlabel("Effective weight (intra × inter)")
    ax.set_title("Per-Criterion Contribution", fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def _sites_map_figure(sites_gdf: gpd.GeoDataFrame, title: str,
                       score_col: str = 'mean_score', figsize: tuple = (9, 7)):
    """Static map of candidate/imported site polygons, coloured by mean score."""
    fig, ax = plt.subplots(figsize=figsize)
    sites_gdf.plot(
        ax=ax, column=score_col, cmap='RdYlGn', vmin=0, vmax=1,
        edgecolor='#333333', linewidth=0.6, legend=True,
        legend_kwds={'label': 'Mean score', 'shrink': 0.6}
    )
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def _fig_to_rl_image(fig, width_cm: float = 16.0):
    """Rasterise a matplotlib figure into a reportlab Image, aspect-ratio preserved."""
    from reportlab.platypus import Image as RLImage
    from reportlab.lib.units import cm
    from PIL import Image as PILImage

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    pil_img = PILImage.open(buf)
    w, h = pil_img.size
    buf.seek(0)
    height_cm = width_cm * h / w
    return RLImage(buf, width=width_cm * cm, height=height_cm * cm)


def _df_to_rl_table(df: pd.DataFrame, styles, col_widths=None):
    """Convert a DataFrame to a styled reportlab Table, wrapping header text."""
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib import colors

    header_style = styles['Normal'].clone('header')
    header_style.textColor = colors.whitesmoke
    header_style.fontSize = 8
    header_style.fontName = 'Helvetica-Bold'
    cell_style = styles['Normal'].clone('cell')
    cell_style.fontSize = 8

    header_row = [Paragraph(str(c), header_style) for c in df.columns]
    body_rows = [
        [Paragraph(str(v), cell_style) for v in row]
        for row in df.values.tolist()
    ]
    table = Table([header_row] + body_rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E7D32')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    return table


def generate_pdf_report(
    output_path: str,
    parameters: dict,
    score_array: np.ndarray,
    profile: dict,
    stats_df: pd.DataFrame,
    display_threshold: float = 0.5,
    oecm_mask: Optional[np.ndarray] = None,
    pa_gdf: Optional[gpd.GeoDataFrame] = None,
    weights_df: Optional[pd.DataFrame] = None,
    sensitivity_stability: Optional[np.ndarray] = None,
    candidate_sites: Optional[gpd.GeoDataFrame] = None,
    imported_sites: Optional[gpd.GeoDataFrame] = None,
    scenario_scores: Optional[Dict[str, np.ndarray]] = None,
    scenario_profile: Optional[dict] = None,
) -> None:
    """
    Generate a comprehensive PDF report covering every Module 2 analysis that
    has actually been run in the current session.

    Parameters
    ----------
    output_path : str
        Path for output PDF file.
    parameters : dict
        Complete parameter dictionary (method, alpha, weights, thresholds,
        timestamp, spec_version, ...) — logged in full at the end of the report.
    score_array : np.ndarray
        Favourability score array [0-1], NaN where ineligible. Required —
        this is the core result every report must contain.
    profile : dict
        Rasterio profile (crs, transform, width, height) for score_array.
    stats_df : pd.DataFrame
        Summary statistics table (territory/eligible/OECM areas, median score, ...).
    display_threshold : float, optional
        Threshold used for the score-distribution vline and implicitly for any
        exported polygons. Default 0.5.
    oecm_mask : np.ndarray, optional
        Boolean OECM-favourable mask — currently informational only (already
        reflected in stats_df); kept for future use (e.g. a favourable-only map).
    pa_gdf : geopandas.GeoDataFrame, optional
        Existing WDPA protected-area network, overlaid as boundaries on maps.
    weights_df : pd.DataFrame, optional
        Per-criterion effective weights (see ui.tab_module2._criterion_weights_df).
        Adds the "Per-Criterion Contribution" section when provided.
    sensitivity_stability : np.ndarray, optional
        Per-pixel stability fraction [0-1] from the Monte Carlo sensitivity
        analysis. Adds the "Weight Sensitivity Analysis" section when provided.
    candidate_sites : geopandas.GeoDataFrame, optional
        Auto-delineated candidate OECM sites (from patch_delineation.delineate_patches).
        Adds the "Candidate OECM Sites" section when provided.
    imported_sites : geopandas.GeoDataFrame, optional
        Externally-proposed sites evaluated against the MCE score
        (patch_delineation.evaluate_external_sites). Adds the
        "Imported & Evaluated Sites" section when provided.
    scenario_scores : dict[str, np.ndarray], optional
        {scenario_name: score_array} from the Scenario Comparison subtab.
        Adds the "Scenario Comparison" section when provided.
    scenario_profile : dict, optional
        Rasterio profile matching scenario_scores arrays (required if
        scenario_scores is provided).

    Raises
    ------
    ImportError
        If reportlab is not installed.
    OSError
        If output file cannot be written.

    Notes
    -----
    Every optional section is skipped silently (not left blank) when its data
    was not provided — i.e. when that analysis was never run in this session.
    All maps are static matplotlib renders (colorbar + scale bar + title),
    since only they can be embedded in a PDF — the live app's interactive
    folium maps cannot.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        )
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.lib import colors
    except ImportError:
        raise ImportError(
            "reportlab is required for PDF generation. "
            "Install with: pip install reportlab"
        )

    logger.info(f"Generating comprehensive PDF report: {output_path}...")

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                             topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story = []
    styles = getSampleStyleSheet()

    # =========================================================================
    # Title page
    # =========================================================================
    story.append(Paragraph("OECM Favourability Analysis Report", styles['Title']))
    story.append(Spacer(1, 0.5 * cm))

    timestamp = parameters.get('timestamp', 'Not recorded')
    method = parameters.get('method', 'Not specified')
    alpha = parameters.get('alpha', 'N/A')
    metadata_text = f"""
    <b>Analysis Date:</b> {timestamp}<br/>
    <b>Aggregation Method:</b> {method}<br/>
    <b>Alpha Parameter:</b> {alpha}<br/>
    <b>Specifications Version:</b> {parameters.get('spec_version', 'v0.1')}
    """
    story.append(Paragraph(metadata_text, styles['Normal']))
    story.append(Spacer(1, 0.8 * cm))

    # =========================================================================
    # Section: Favourability map (WITH legend / colorbar / scale bar)
    # =========================================================================
    story.append(Paragraph("Favourability Map", styles['Heading1']))
    map_fig = _raster_map_figure(
        score_array, profile,
        title="OECM Favourability Score",
        cmap_name='RdYlGn', vmin=0.0, vmax=1.0,
        cbar_label="Favourability score (0 = unfavourable, 1 = highly favourable)",
        pa_gdf=pa_gdf,
    )
    story.append(_fig_to_rl_image(map_fig))
    if pa_gdf is not None and len(pa_gdf) > 0:
        story.append(Paragraph(
            "Grey outlines: existing WDPA protected-area network.",
            styles['Italic']
        ))
    story.append(Spacer(1, 0.5 * cm))

    if stats_df is not None and len(stats_df) > 0:
        story.append(Paragraph("Summary Statistics", styles['Heading2']))
        table_data = [stats_df.columns.tolist()] + stats_df.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)

    # =========================================================================
    # Section: Score distribution + per-criterion contribution
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Score Distribution & Criterion Weights", styles['Heading1']))

    valid_scores = score_array[~np.isnan(score_array)]
    if len(valid_scores) > 0:
        story.append(_fig_to_rl_image(
            _histogram_figure(valid_scores, display_threshold), width_cm=15
        ))
        dist_stats = pd.DataFrame([
            {'Metric': 'Mean', 'Value': f"{np.mean(valid_scores):.3f}"},
            {'Metric': 'Median', 'Value': f"{np.median(valid_scores):.3f}"},
            {'Metric': 'Std dev', 'Value': f"{np.std(valid_scores):.3f}"},
            {'Metric': '% eligible ≥ 0.5', 'Value': f"{(valid_scores >= 0.5).mean()*100:.1f}%"},
            {'Metric': '% eligible ≥ 0.7', 'Value': f"{(valid_scores >= 0.7).mean()*100:.1f}%"},
        ])
        story.append(_df_to_rl_table(dist_stats, styles, col_widths=[8 * cm, 8 * cm]))
        story.append(Spacer(1, 0.5 * cm))

    if weights_df is not None and len(weights_df) > 0:
        story.append(_fig_to_rl_image(_weights_bar_figure(weights_df), width_cm=15))
        story.append(Paragraph(
            "Effective weight = intra-group weight × inter-group weight.",
            styles['Italic']
        ))

    # =========================================================================
    # Section: Weight Sensitivity Analysis (optional)
    # =========================================================================
    if sensitivity_stability is not None:
        story.append(PageBreak())
        story.append(Paragraph("Weight Sensitivity Analysis", styles['Heading1']))
        story.append(Paragraph(
            "Monte Carlo analysis: the MCE was re-run with weights randomly "
            "perturbed around the chosen values (Dirichlet distribution). "
            "The stability map shows how often each pixel exceeds the display "
            "threshold — values close to 1.0 are robust to weight uncertainty.",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.3 * cm))
        story.append(_fig_to_rl_image(_raster_map_figure(
            sensitivity_stability, profile,
            title="Stability Map",
            cmap_name='RdYlGn', vmin=0.0, vmax=1.0,
            cbar_label="Stability (fraction of runs ≥ threshold)",
        )))
        stab_valid = sensitivity_stability[~np.isnan(sensitivity_stability)]
        if len(stab_valid) > 0:
            sens_stats = pd.DataFrame([
                {'Metric': 'Highly stable (≥80%)', 'Value': f"{(stab_valid >= 0.8).mean()*100:.1f}% of pixels"},
                {'Metric': 'Ambiguous (40-60%)', 'Value': f"{((stab_valid >= 0.4) & (stab_valid < 0.6)).mean()*100:.1f}% of pixels"},
                {'Metric': 'Unstable (<20%)', 'Value': f"{(stab_valid < 0.2).mean()*100:.1f}% of pixels"},
            ])
            story.append(_df_to_rl_table(sens_stats, styles, col_widths=[8 * cm, 8 * cm]))

    # =========================================================================
    # Section: Candidate OECM Sites (optional)
    # =========================================================================
    if candidate_sites is not None and len(candidate_sites) > 0:
        story.append(PageBreak())
        story.append(Paragraph("Candidate OECM Sites", styles['Heading1']))
        story.append(Paragraph(
            "Spatially contiguous patches above the score threshold, filtered by "
            "Minimum Mapping Unit and ranked by a composite of mean score, gap "
            "overlap, patch area and proximity to existing PAs.",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.3 * cm))
        story.append(_fig_to_rl_image(
            _sites_map_figure(candidate_sites, "Candidate OECM Sites — Mean Score")
        ))
        disp = candidate_sites[[
            'patch_id', 'area_ha', 'mean_score', 'max_score',
            'compactness', 'dist_to_pa_km', 'gap_overlap_pct', 'rank_score'
        ]].copy()
        disp.columns = ['Rank', 'Area (ha)', 'Mean', 'Max', 'Compact.', 'PA dist (km)', 'Gap (%)', 'Rank score']
        for c in disp.columns[1:]:
            disp[c] = disp[c].map(lambda v: f"{v:.2f}")
        story.append(_df_to_rl_table(disp, styles))

    # =========================================================================
    # Section: Imported & Evaluated Sites (optional)
    # =========================================================================
    if imported_sites is not None and len(imported_sites) > 0:
        story.append(PageBreak())
        story.append(Paragraph("Imported & Evaluated Candidate Sites", styles['Heading1']))
        story.append(Paragraph(
            "Externally-proposed site polygons evaluated against the same MCE "
            "favourability score and ranking formula as the auto-delineated sites above.",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.3 * cm))
        story.append(_fig_to_rl_image(
            _sites_map_figure(imported_sites, "Imported Sites — Mean Score")
        ))
        disp_i = imported_sites[[
            'site_name', 'area_ha', 'mean_score', 'pct_eliminated',
            'pct_oecm_favourable', 'compactness', 'dist_to_pa_km',
            'gap_overlap_pct', 'rank_score'
        ]].copy()
        disp_i.columns = ['Site', 'Area (ha)', 'Mean', '% Elim.', '% OECM',
                           'Compact.', 'PA dist (km)', 'Gap (%)', 'Rank score']
        for c in disp_i.columns[1:]:
            disp_i[c] = disp_i[c].map(lambda v: f"{v:.2f}")
        story.append(_df_to_rl_table(disp_i, styles))

    # =========================================================================
    # Section: Scenario Comparison (optional)
    # =========================================================================
    if scenario_scores is not None and scenario_profile is not None:
        story.append(PageBreak())
        story.append(Paragraph("Scenario Comparison", styles['Heading1']))
        story.append(Paragraph(
            "Favourability recomputed under three preset inter-group weight "
            "trade-offs (Biodiversity priority, Services priority, Compromise), "
            "all other settings held fixed.",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.3 * cm))

        pixel_area_ha_sc = abs(scenario_profile['transform'][0] * scenario_profile['transform'][4]) / 10_000.0
        summary_rows = []
        for name, arr in scenario_scores.items():
            valid = arr[~np.isnan(arr)]
            summary_rows.append({
                'Scenario': name,
                'Eligible (ha)': f"{len(valid) * pixel_area_ha_sc:,.0f}",
                'Mean score': f"{np.mean(valid):.3f}" if len(valid) else "N/A",
                'Area ≥0.5 (ha)': f"{int((valid >= 0.5).sum()) * pixel_area_ha_sc:,.0f}",
            })
        story.append(_df_to_rl_table(pd.DataFrame(summary_rows), styles))
        story.append(Spacer(1, 0.5 * cm))

        agreement = np.zeros_like(next(iter(scenario_scores.values())), dtype=np.int8)
        any_valid = np.zeros_like(agreement, dtype=bool)
        for arr in scenario_scores.values():
            vm = ~np.isnan(arr)
            any_valid |= vm
            agreement += ((arr >= 0.5) & vm).astype(np.int8)
        agreement_f = agreement.astype(np.float32)
        agreement_f[~any_valid] = np.nan

        story.append(_fig_to_rl_image(_raster_map_figure(
            agreement_f, scenario_profile,
            title="Scenario Agreement (pixels scoring ≥0.5 under N scenarios)",
            cmap_name='YlGn', vmin=0, vmax=len(scenario_scores),
            cbar_label="Scenarios agreeing",
            cbar_ticks=list(range(len(scenario_scores) + 1)),
        )))

    # =========================================================================
    # Section: Full parameter log
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Full Parameter Configuration", styles['Heading1']))
    story.append(Spacer(1, 0.3 * cm))
    param_text = "<br/>".join([f"<b>{k}:</b> {v}" for k, v in parameters.items()])
    story.append(Paragraph(param_text, styles['Normal']))

    doc.build(story)
    logger.info(f"PDF report generated successfully: {output_path}")
