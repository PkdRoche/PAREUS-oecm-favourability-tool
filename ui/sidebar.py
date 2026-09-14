"""Streamlit sidebar components for parameter configuration."""
import streamlit as st
import yaml
from pathlib import Path
from shapely.geometry import box


@st.cache_resource
def load_config_defaults():
    """
    Load default parameter values from config/criteria_defaults.yaml.

    Returns
    -------
    dict
        Configuration dictionary with inter-group weights, intra-group weights,
        aggregation settings, and thresholds.
    """
    config_path = Path(__file__).parent.parent / "config" / "criteria_defaults.yaml"

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_path}")
        return {}


@st.cache_resource
def load_settings():
    """
    Load general settings from config/settings.yaml.

    Returns
    -------
    dict
        Settings dictionary with CRS, resolution, paths, etc.
    """
    settings_path = Path(__file__).parent.parent / "config" / "settings.yaml"

    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)
        return settings
    except FileNotFoundError:
        st.warning(f"Settings file not found: {settings_path}")
        return {}


def render_sidebar():
    """
    Render sidebar with global parameters and file upload controls.

    Returns
    -------
    dict
        Dictionary of user-configured parameters including:
        - study_area_nuts_id: NUTS2 region ID or 'CUSTOM' (str)
        - study_area_name: Region name or custom bbox description (str)
        - study_area_geometry: Shapely geometry in EPSG:3035 (BaseGeometry)
        - threshold_pressure: eliminatory pressure threshold (float)
        - method: aggregation method ('geometric' or 'owa')
        - alpha: OWA orness parameter (float)
        - W_A, W_B, W_C: inter-group weights (float)
        - w_condition, w_regulating_es, w_pressure: Group A intra-weights (float)
        - w_cultural_es: Group B weight (float, always 1.0)
        - w_provisioning_es, w_landuse_compatible: Group C intra-weights (float)
        - gap_bonus: optional gap analysis bonus (float)
    """
    # Load defaults
    config = load_config_defaults()
    settings = load_settings()

    st.sidebar.title("OECM Favourability Tool")
    st.sidebar.markdown("---")

    # ===================================================================
    # Section 1: Study area
    # ===================================================================
    st.sidebar.header("1. Study Area")

    study_area_nuts_id = None
    study_area_name = None
    study_area_geometry = None

    nuts_file = st.session_state.get('nuts_file')

    if nuts_file and Path(nuts_file).exists():
        # ── Local NUTS file ──────────────────────────────────────────────
        @st.cache_data(show_spinner="Loading NUTS boundaries…")
        def _load_nuts_local(path: str):
            import geopandas as gpd, zipfile, tempfile, os
            # Extract ZIP to a temp dir so all sidecar files (.dbf, .shx, .prj) are present
            if path.lower().endswith('.zip'):
                tmp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(path, 'r') as zf:
                    zf.extractall(tmp_dir)
                # Find the main vector file inside the ZIP
                read_path = None
                for ext in ('.gpkg', '.shp', '.geojson'):
                    matches = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.lower().endswith(ext)]
                    if matches:
                        read_path = matches[0]
                        break
                if read_path is None:
                    raise ValueError("No .gpkg, .shp or .geojson found inside the ZIP archive.")
            else:
                read_path = path
            gdf = gpd.read_file(read_path)
            # Normalise column names (Eurostat vs. GISCO naming variations)
            col_map = {}
            cols_lower = {c.lower(): c for c in gdf.columns}
            for std, variants in [
                ('NUTS_ID',   ['nuts_id', 'nutsid', 'id']),
                ('NUTS_NAME', ['nuts_name', 'name_latn', 'name', 'nuts_name']),
                ('LEVL_CODE', ['levl_code', 'levl', 'level', 'nuts_level']),
                ('CNTR_CODE', ['cntr_code', 'cntr', 'country_code']),
            ]:
                if std not in gdf.columns:
                    for v in variants:
                        if v in cols_lower:
                            col_map[cols_lower[v]] = std
                            break
            if col_map:
                gdf = gdf.rename(columns=col_map)
            if gdf.crs is None:
                gdf = gdf.set_crs('EPSG:4326')
            if gdf.crs.to_epsg() != 3035:
                gdf = gdf.to_crs('EPSG:3035')
            return gdf

        try:
            nuts_gdf = _load_nuts_local(nuts_file)

            # Available NUTS levels
            if 'LEVL_CODE' in nuts_gdf.columns:
                available_levels = sorted(nuts_gdf['LEVL_CODE'].unique().tolist())
            else:
                available_levels = [2]

            level_labels = {0: 'NUTS 0 (Country)', 1: 'NUTS 1', 2: 'NUTS 2', 3: 'NUTS 3'}
            level_options = [l for l in available_levels if l in level_labels]

            selected_level = st.sidebar.selectbox(
                "NUTS level",
                options=level_options,
                format_func=lambda l: level_labels.get(l, f"Level {l}"),
                index=min(2, len(level_options) - 1) if len(level_options) > 2 else 0,
            )

            level_gdf = nuts_gdf[nuts_gdf['LEVL_CODE'] == selected_level].copy() \
                if 'LEVL_CODE' in nuts_gdf.columns else nuts_gdf

            # Country filter
            if 'CNTR_CODE' in level_gdf.columns:
                countries = sorted(level_gdf['CNTR_CODE'].unique().tolist())
                default_idx = countries.index('FR') if 'FR' in countries else 0
                selected_country = st.sidebar.selectbox(
                    "Country", options=countries, index=default_idx
                )
                level_gdf = level_gdf[level_gdf['CNTR_CODE'] == selected_country]

            name_col = 'NUTS_NAME' if 'NUTS_NAME' in level_gdf.columns else level_gdf.columns[0]
            level_gdf = level_gdf.sort_values(name_col).reset_index(drop=True)

            if 'NUTS_ID' in level_gdf.columns:
                region_options = (level_gdf[name_col].astype(str) + ' ('
                                  + level_gdf['NUTS_ID'].astype(str) + ')').tolist()
            else:
                region_options = level_gdf[name_col].astype(str).tolist()

            if region_options:
                selected_idx = st.sidebar.selectbox(
                    f"NUTS {selected_level} Region",
                    options=range(len(region_options)),
                    format_func=lambda i: region_options[i],
                )
                selected_row = level_gdf.iloc[selected_idx]
                study_area_geometry = selected_row.geometry
                study_area_name = str(selected_row.get(name_col, ''))
                study_area_nuts_id = str(selected_row['NUTS_ID']) \
                    if 'NUTS_ID' in level_gdf.columns else study_area_name
                area_km2 = study_area_geometry.area / 1_000_000
                st.sidebar.caption(f"**{study_area_nuts_id}** — {area_km2:,.0f} km²")
            else:
                st.sidebar.warning("No regions found for this selection.")

        except Exception as e:
            st.sidebar.error(f"Failed to load NUTS file: {e}")

    else:
        # ── Eurostat online fallback ─────────────────────────────────────
        from modules.utils.nuts2_loader import load_nuts2, get_countries, get_nuts2_for_country, get_nuts2_geometry

        nuts2_gdf = None
        try:
            nuts2_gdf = load_nuts2(year=2021, scale="20M")
        except Exception as e:
            st.sidebar.warning(
                f"Could not load NUTS2 boundaries from Eurostat: {e}\n\n"
                "Upload a local NUTS file in the **① Data Upload** tab, "
                "or fall back to manual bounding box below."
            )

        if nuts2_gdf is not None:
            countries = get_countries(nuts2_gdf)

            # One-time restore of a NUTS2 region loaded from a .ini project
            # (see tab_data_upload.py). Pre-seeds the keyed widgets' session
            # state before they're instantiated — Streamlit then uses that
            # value as the widget's current selection instead of the default
            # index below, and normal interactive changes take over from there.
            _pending_nuts_id = st.session_state.pop('_pending_nuts_id', None)
            if _pending_nuts_id and _pending_nuts_id[:2] in countries:
                st.session_state['sidebar_study_country'] = _pending_nuts_id[:2]

            default_country_idx = countries.index("FR") if "FR" in countries else 0
            selected_country = st.sidebar.selectbox(
                "Country", options=countries, index=default_country_idx,
                key='sidebar_study_country'
            )
            nuts2_regions = get_nuts2_for_country(nuts2_gdf, selected_country)
            region_options = [
                f"{row.NUTS_NAME} ({row.NUTS_ID})" for _, row in nuts2_regions.iterrows()
            ]
            region_nuts_ids = nuts2_regions['NUTS_ID'].tolist()
            if region_options:
                if _pending_nuts_id and _pending_nuts_id in region_nuts_ids:
                    st.session_state['sidebar_study_region_idx'] = region_nuts_ids.index(_pending_nuts_id)
                elif st.session_state.get('sidebar_study_region_idx', 0) >= len(region_options):
                    # Country changed (manually or via restore) and the previously
                    # stored index is out of range for the new region list.
                    st.session_state['sidebar_study_region_idx'] = 0

                selected_region_idx = st.sidebar.selectbox(
                    "NUTS2 Region",
                    options=range(len(region_options)),
                    format_func=lambda i: region_options[i],
                    key='sidebar_study_region_idx',
                )
                study_area_nuts_id = region_nuts_ids[selected_region_idx]
                study_area_name = nuts2_regions.iloc[selected_region_idx]['NUTS_NAME']
                study_area_geometry = get_nuts2_geometry(nuts2_gdf, study_area_nuts_id)
                if study_area_geometry is not None:
                    area_km2 = study_area_geometry.area / 1_000_000
                    st.sidebar.caption(f"**{study_area_nuts_id}** — {area_km2:,.0f} km²")
        else:
            # Last resort: manual bounding box
            st.sidebar.info("Enter bounding box coordinates in EPSG:3035 (meters)")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                xmin = st.number_input("X min", value=2500000.0, step=1000.0)
                ymin = st.number_input("Y min", value=1500000.0, step=1000.0)
            with col2:
                xmax = st.number_input("X max", value=3500000.0, step=1000.0)
                ymax = st.number_input("Y max", value=2500000.0, step=1000.0)
            study_area_geometry = box(xmin, ymin, xmax, ymax)
            study_area_name = f"Custom bbox ({xmin:.0f}, {ymin:.0f}, {xmax:.0f}, {ymax:.0f})"
            study_area_nuts_id = "CUSTOM"
            area_km2 = study_area_geometry.area / 1_000_000
            st.sidebar.caption(f"Area: {area_km2:,.0f} km²")

    # Display settings
    if settings:
        resolution_m = settings.get('resolution_m', 100)
        crs = settings.get('crs', 'EPSG:3035')
        st.sidebar.info(f"Target resolution: {resolution_m} m\nCRS: {crs}")

    st.sidebar.markdown("---")

    # ===================================================================
    # Section 2: Group D eliminatory thresholds
    # ===================================================================
    st.sidebar.header("2. Eliminatory Thresholds (Group D)")

    default_pressure = config.get('eliminatory', {}).get('max_anthropogenic_pressure', 150.0)

    threshold_pressure = st.sidebar.slider(
        "Max anthropogenic pressure (hab/km²)",
        min_value=0.0,
        max_value=500.0,
        value=default_pressure,
        step=10.0,
        help="Pixels above this threshold are excluded from OECM analysis"
    )

    st.sidebar.info(
        "Incompatible land use classes (urban, industrial) are defined in "
        "config/land_use_compatibility.yaml"
    )

    st.sidebar.markdown("---")

    # ===================================================================
    # Section 2b: Normalisation method
    # ===================================================================
    st.sidebar.header("2b. Normalisation Method")
    percentile_norm = st.sidebar.toggle(
        "Percentile normalisation (2nd–98th)",
        value=False,
        help=(
            "When ON: each criterion is clipped to its 2nd–98th percentile "
            "range before transformation. This prevents extreme outlier pixels "
            "from compressing the rest of the data into a narrow score range. "
            "Recommended when input rasters contain artefacts or extreme values."
        )
    )
    if percentile_norm:
        st.sidebar.caption(
            "Outlier-robust: the top 2% and bottom 2% of pixel values are "
            "clamped before normalisation. Score map is less sensitive to "
            "single extreme pixels."
        )

    st.sidebar.markdown("---")

    # ===================================================================
    # Section 3: Aggregation method
    # ===================================================================
    st.sidebar.header("3. Aggregation Method")

    default_method = config.get('aggregation', {}).get('default_method', 'geometric')
    method_display = st.sidebar.selectbox(
        "MCE aggregation method",
        options=["Weighted geometric mean", "Yager OWA"],
        index=0 if default_method == 'geometric' else 1,
        help="Weighted geometric mean: strictly non-compensatory. "
             "Yager OWA: adjustable compensation via alpha parameter."
    )

    method = 'geometric' if method_display == "Weighted geometric mean" else 'owa'

    # Alpha slider (only visible for OWA)
    default_alpha = config.get('aggregation', {}).get('default_alpha', 0.25)
    alpha = default_alpha

    if method == 'owa':
        alpha = st.sidebar.slider(
            "OWA orness parameter (α)",
            min_value=0.0,
            max_value=1.0,
            value=default_alpha,
            step=0.05,
            help=(
                "α = 0.0: AND logic (all criteria required)\n"
                "α = 0.5: Balanced compensation\n"
                "α = 1.0: OR logic (one criterion sufficient)"
            )
        )

        # Display interpretation label
        if alpha <= 0.15:
            alpha_label = "Strict AND — all criteria required"
        elif alpha <= 0.4:
            alpha_label = "Near AND — strong non-compensation"
        elif alpha <= 0.6:
            alpha_label = "Balanced"
        elif alpha <= 0.85:
            alpha_label = "Near OR — high compensation"
        else:
            alpha_label = "Permissive OR — one criterion sufficient"

        st.sidebar.caption(f"Current setting: {alpha_label}")

    st.sidebar.markdown("---")

    # ===================================================================
    # AHP weight injection — read values derived in the AHP tab
    # ===================================================================
    ahp_weights = st.session_state.get('ahp_weights', {})
    ahp_active  = st.session_state.get('ahp_source', False)
    _src_badge  = " `[AHP]`" if ahp_active else " `[Manual]`"

    # ===================================================================
    # Section 4: Inter-group weights (W_A, W_B, W_C)
    # ===================================================================
    st.sidebar.header("4. Inter-Group Weights" + _src_badge)
    st.sidebar.caption(
        "Relative importance of each functional group. "
        + ("Derived from AHP — go to **② Weight Calibration** to revise."
           if ahp_active else
           "Set manually or derive via **② Weight Calibration (AHP)**.")
    )

    defaults_inter = config.get('inter_group_weights', {})

    W_A = st.sidebar.slider(
        "W_A — Ecological integrity",
        min_value=0.0,
        max_value=1.0,
        value=float(ahp_weights.get('W_A', defaults_inter.get('W_A', 0.50))),
        step=0.05,
        help="Group A: ecosystem condition, regulating ES, low pressure"
    )

    W_B = st.sidebar.slider(
        "W_B — Co-benefits / social compatibility",
        min_value=0.0,
        max_value=1.0,
        value=float(ahp_weights.get('W_B', defaults_inter.get('W_B', 0.15))),
        step=0.05,
        help="Group B: cultural ecosystem services"
    )

    W_C = st.sidebar.slider(
        "W_C — Production / use function",
        min_value=0.0,
        max_value=1.0,
        value=float(ahp_weights.get('W_C', defaults_inter.get('W_C', 0.35))),
        step=0.05,
        help="Group C: provisioning ES, compatible land use"
    )

    # Display sum with warning if not equal to 1.0
    weight_sum = W_A + W_B + W_C

    if abs(weight_sum - 1.0) < 0.001:
        st.sidebar.success(f"Σ = {weight_sum:.3f} ✓")
    else:
        st.sidebar.error(f"Σ = {weight_sum:.3f} ≠ 1.0")
        st.sidebar.warning("Inter-group weights must sum to 1.0. Use the normalise button below.")

    # Normalise button (manual mode only — AHP weights are already normalised)
    if not ahp_active:
        if st.sidebar.button("Normalise inter-group weights"):
            if weight_sum > 0:
                st.session_state['W_A_normalised'] = W_A / weight_sum
                st.session_state['W_B_normalised'] = W_B / weight_sum
                st.session_state['W_C_normalised'] = W_C / weight_sum
                st.rerun()

    # Use normalised values if available
    if 'W_A_normalised' in st.session_state:
        W_A = st.session_state.pop('W_A_normalised')
        W_B = st.session_state.pop('W_B_normalised')
        W_C = st.session_state.pop('W_C_normalised')

    st.sidebar.markdown("---")

    # ===================================================================
    # Section 5: Intra-group weights (expandable)
    # ===================================================================
    with st.sidebar.expander("5. Intra-Group Weights (Advanced)" + _src_badge):
        st.markdown("### Group A — Ecological integrity")

        defaults_a = config.get('group_a_weights', {})

        w_condition = st.slider(
            "Ecosystem condition",
            min_value=0.0,
            max_value=1.0,
            value=float(ahp_weights.get('w_condition',
                        defaults_a.get('ecosystem_condition', 0.45))),
            step=0.05,
            key='w_condition'
        )

        w_regulating_es = st.slider(
            "Regulating ES capacity",
            min_value=0.0,
            max_value=1.0,
            value=float(ahp_weights.get('w_regulating_es',
                        defaults_a.get('regulating_es', 0.35))),
            step=0.05,
            key='w_regulating_es'
        )

        w_pressure = st.slider(
            "Low anthropogenic pressure",
            min_value=0.0,
            max_value=1.0,
            value=float(ahp_weights.get('w_pressure',
                        defaults_a.get('low_pressure', 0.20))),
            step=0.05,
            key='w_pressure'
        )

        sum_a = w_condition + w_regulating_es + w_pressure

        if abs(sum_a - 1.0) < 0.001:
            st.success(f"Group A: Σ = {sum_a:.3f} ✓")
        else:
            st.error(f"Group A: Σ = {sum_a:.3f} ≠ 1.0")

        if not ahp_active and st.button("Normalise Group A weights"):
            if sum_a > 0:
                st.session_state['group_a_normalised'] = {
                    'w_condition':     w_condition     / sum_a,
                    'w_regulating_es': w_regulating_es / sum_a,
                    'w_pressure':      w_pressure      / sum_a,
                }
                st.rerun()

        if 'group_a_normalised' in st.session_state:
            _n = st.session_state.pop('group_a_normalised')
            w_condition     = _n['w_condition']
            w_regulating_es = _n['w_regulating_es']
            w_pressure      = _n['w_pressure']

        st.markdown("---")
        st.markdown("### Group B — Co-benefits")

        w_cultural_es = 1.0
        st.info(f"Cultural ES capacity: {w_cultural_es:.2f} (single criterion)")

        st.markdown("---")
        st.markdown("### Group C — Production function")

        defaults_c = config.get('group_c_weights', {})

        w_provisioning_es = st.slider(
            "Provisioning ES capacity",
            min_value=0.0,
            max_value=1.0,
            value=float(ahp_weights.get('w_provisioning_es',
                        defaults_c.get('provisioning_es', 0.60))),
            step=0.05,
            key='w_provisioning_es'
        )

        w_landuse_compatible = st.slider(
            "Compatible land use",
            min_value=0.0,
            max_value=1.0,
            value=float(ahp_weights.get('w_landuse_compatible',
                        defaults_c.get('compatible_landuse', 0.40))),
            step=0.05,
            key='w_landuse_compatible'
        )

        sum_c = w_provisioning_es + w_landuse_compatible

        if abs(sum_c - 1.0) < 0.001:
            st.success(f"Group C: Σ = {sum_c:.3f} ✓")
        else:
            st.error(f"Group C: Σ = {sum_c:.3f} ≠ 1.0")

        if not ahp_active and st.button("Normalise Group C weights"):
            if sum_c > 0:
                st.session_state['group_c_normalised'] = {
                    'w_provisioning_es':    w_provisioning_es    / sum_c,
                    'w_landuse_compatible': w_landuse_compatible / sum_c,
                }
                st.rerun()

        if 'group_c_normalised' in st.session_state:
            _n = st.session_state.pop('group_c_normalised')
            w_provisioning_es    = _n['w_provisioning_es']
            w_landuse_compatible = _n['w_landuse_compatible']

    st.sidebar.markdown("---")

    # ===================================================================
    # Section 6: Gap analysis bonus
    # ===================================================================
    st.sidebar.header("6. Gap Analysis Bonus (Optional)")

    default_gap_bonus_max = config.get('eliminatory', {}).get('gap_bonus_max', 0.20)

    gap_bonus = st.sidebar.slider(
        "Gap priority bonus",
        min_value=0.0,
        max_value=default_gap_bonus_max,
        value=0.0,
        step=0.01,
        help=(
            "Optional positive weighting for areas identified as gaps "
            "in Module 1 analysis. 0.0 = no bonus."
        )
    )

    st.sidebar.markdown("---")

    # ===================================================================
    # Section 6b: PA Network Proximity Bonus
    # ===================================================================
    st.sidebar.header("6b. PA Proximity Bonus (Optional)")
    st.sidebar.caption(
        "Boosts scores for pixels close to existing protected areas — "
        "rewarding spatial complementarity with the PA network."
    )

    proximity_bonus = st.sidebar.slider(
        "Max proximity bonus",
        min_value=0.0,
        max_value=0.20,
        value=0.0,
        step=0.01,
        help=(
            "Maximum score multiplier applied to pixels adjacent to a PA. "
            "Bonus decays exponentially with distance. 0.0 = disabled."
        )
    )

    proximity_decay_km = st.sidebar.slider(
        "Decay distance (km)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        disabled=(proximity_bonus == 0.0),
        help=(
            "Distance at which the proximity bonus falls to 37% of its "
            "maximum value (1/e decay). "
            "E.g. 10 km → full bonus at 0 km, ~37% bonus at 10 km, "
            "~14% at 20 km."
        )
    )

    st.sidebar.markdown("---")

    # ===================================================================
    # Section 6c: Sensitivity Analysis Settings
    # ===================================================================
    with st.sidebar.expander("6c. Sensitivity Analysis Settings"):
        sensitivity_runs = st.slider(
            "Monte Carlo runs",
            min_value=50, max_value=500, value=200, step=50,
            key='sensitivity_runs',
            help="More runs = more stable estimate but slower computation."
        )
        sensitivity_concentration = st.slider(
            "Weight uncertainty (concentration)",
            min_value=5, max_value=100, value=20, step=5,
            key='sensitivity_concentration',
            help=(
                "Controls how widely weights are perturbed.\n"
                "5 = high uncertainty, 100 = low uncertainty."
            )
        )
        sensitivity_perturb_intra = st.toggle(
            "Also perturb intra-group weights",
            value=True,
            key='sensitivity_perturb_intra',
            help="When OFF, only inter-group weights (W_A/B/C) are perturbed."
        )

    st.sidebar.markdown("---")

    # ===================================================================
    # Section 6d: Patch Delineation Settings
    # ===================================================================
    with st.sidebar.expander("6d. Candidate Site Delineation"):
        mmu_ha = st.slider(
            "Minimum Mapping Unit (ha)",
            min_value=10, max_value=5000, value=100, step=10,
            key='mmu_ha',
            help=(
                "Candidate OECM patches smaller than this area are discarded. "
                "Set based on the minimum viable governance unit for OECMs "
                "in your jurisdiction."
            )
        )

    st.sidebar.markdown("---")

    # ===================================================================
    # Section 6e: PA Exclusion / Overlay
    # ===================================================================
    st.sidebar.header("6e. PA Exclusion from Favourability")
    st.sidebar.caption(
        "Exclude pixels already covered by existing PAs from the favourability "
        "result (sets their score to NaN so they appear as excluded). "
        "Useful to focus OECM search on areas **outside** the existing network."
    )
    exclude_pa_pixels = st.sidebar.checkbox(
        "Exclude pixels within existing PAs",
        value=st.session_state.get('exclude_pa_pixels', False),
        key='exclude_pa_pixels',
        help=(
            "When ON, pixels that fall inside any PA polygon (after applying "
            "the protection-class filter below) are masked out of the score map."
        )
    )

    _all_pa_classes = ['strict_core', 'regulatory', 'contractual', 'unassigned']
    _pa_class_labels = {
        'strict_core':  'Strict core (IUCN I–II)',
        'regulatory':   'Regulatory (IUCN III–VI)',
        'contractual':  'Contractual (national)',
        'unassigned':   'Unassigned / unknown',
    }
    exclude_pa_classes = st.sidebar.multiselect(
        "PA classes to exclude",
        options=_all_pa_classes,
        default=st.session_state.get('exclude_pa_classes', ['strict_core', 'regulatory']),
        format_func=lambda c: _pa_class_labels.get(c, c),
        key='exclude_pa_classes',
        disabled=not exclude_pa_pixels,
        help="Only pixels whose PA belongs to one of the selected classes are excluded.",
    )

    # PA overlay toggle lives inline in the map tab (avoids duplicate key conflict).
    # Read the session state value set by the inline checkbox.
    show_pa_overlay = st.session_state.get('show_pa_overlay', True)

    st.sidebar.markdown("---")

    # ===================================================================
    # Section 7: Module 1 weight suggestions
    # ===================================================================
    st.sidebar.header("7. Module 1 Weight Suggestions")

    # Check if Module 1 has been run and produced weight suggestions
    has_suggestions = 'proposed_group_a_weights' in st.session_state

    if has_suggestions:
        st.sidebar.info(
            "Module 1 gap analysis has proposed weights for Group A criteria "
            "based on ecosystem representativity deficits."
        )

        if st.sidebar.button(
            "Apply Module 1 weight suggestions",
            help=(
                "Applies gap-filling logic: criteria corresponding to under-represented "
                "ecosystem types receive higher weights. Only affects Group A intra-weights."
            )
        ):
            proposed = st.session_state['proposed_group_a_weights']

            # Validate handoff before applying
            try:
                from modules.module1_protected_areas.handoff import (
                    validate_weight_handoff,
                    format_weights_for_mce
                )

                config_path = Path(__file__).parent.parent / "config" / "criteria_defaults.yaml"
                validate_weight_handoff(proposed, str(config_path))
                formatted_weights = format_weights_for_mce(proposed)

                # Update Group A weights
                st.session_state['group_a_applied'] = {
                    'w_condition': formatted_weights.get('ecosystem_condition', w_condition),
                    'w_regulating_es': formatted_weights.get('regulating_es', w_regulating_es),
                    'w_pressure': formatted_weights.get('low_pressure', w_pressure)
                }

                st.sidebar.success("Module 1 weights validated and applied to Group A!")
                st.rerun()

            except ValueError as e:
                st.sidebar.error(f"Weight validation failed: {e}")
            except Exception as e:
                st.sidebar.error(f"Unexpected error applying weights: {e}")
    else:
        st.sidebar.info(
            "Run Module 1 gap analysis first to receive weight suggestions "
            "based on ecosystem representativity."
        )

    # Use applied weights if available
    if 'group_a_applied' in st.session_state:
        w_condition = st.session_state['group_a_applied']['w_condition']
        w_regulating_es = st.session_state['group_a_applied']['w_regulating_es']
        w_pressure = st.session_state['group_a_applied']['w_pressure']
        del st.session_state['group_a_applied']

    # ===================================================================
    # Return all parameters
    # ===================================================================
    return {
        'study_area_nuts_id': study_area_nuts_id,
        'study_area_name': study_area_name,
        'study_area_geometry': study_area_geometry,
        'threshold_pressure': threshold_pressure,
        'method': method,
        'alpha': alpha,
        'W_A': W_A,
        'W_B': W_B,
        'W_C': W_C,
        'w_condition': w_condition,
        'w_regulating_es': w_regulating_es,
        'w_pressure': w_pressure,
        'w_cultural_es': w_cultural_es,
        'w_provisioning_es': w_provisioning_es,
        'w_landuse_compatible': w_landuse_compatible,
        'percentile_norm': percentile_norm,
        'proximity_bonus': proximity_bonus,
        'proximity_decay_km': proximity_decay_km,
        'sensitivity_runs': st.session_state.get('sensitivity_runs', 200),
        'sensitivity_concentration': st.session_state.get('sensitivity_concentration', 20),
        'sensitivity_perturb_intra': st.session_state.get('sensitivity_perturb_intra', True),
        'mmu_ha': st.session_state.get('mmu_ha', 100),
        'gap_bonus': gap_bonus,
        'exclude_pa_pixels': exclude_pa_pixels,
        'exclude_pa_classes': exclude_pa_classes,
        'show_pa_overlay': show_pa_overlay,
    }
