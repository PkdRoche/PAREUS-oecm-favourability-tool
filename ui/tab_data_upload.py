"""Streamlit UI for Data Upload Tab — centralized input layer management."""
import streamlit as st
import tempfile
import configparser
from pathlib import Path
import logging
import subprocess
import sys

# Import validation function with graceful fallback
try:
    from modules.module2_favourability.raster_preprocessing import (
        validate_and_rescale_layer, load_raster
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

logger = logging.getLogger(__name__)

# True when running locally on Windows (native file dialog available).
# On Streamlit Cloud (Linux) we fall back to st.file_uploader.
_IS_LOCAL = sys.platform == 'win32'

# Sidebar analysis parameters saved/restored via .ini project files, with
# their expected type for round-tripping through configparser (which only
# stores strings). Keys match ui/sidebar.py's returned `parameters` dict —
# see the `_pending_sidebar_params` restore logic in render_sidebar().
_PARAM_TYPES = {
    'threshold_pressure': float,
    'method': str,
    'alpha': float,
    'W_A': float, 'W_B': float, 'W_C': float,
    'w_condition': float, 'w_regulating_es': float, 'w_pressure': float,
    'w_provisioning_es': float, 'w_landuse_compatible': float,
    'percentile_norm': bool,
    'proximity_bonus': float, 'proximity_decay_km': float,
    'sensitivity_runs': int, 'sensitivity_concentration': int,
    'sensitivity_perturb_intra': bool,
    'mmu_ha': int,
    'gap_bonus': float,
    'exclude_pa_pixels': bool,
    'exclude_pa_classes': list,
    'show_pa_overlay': bool,
    'provisioning_calibration_mode': str,
    'provisioning_mean': float,
    'provisioning_std': float,
}


def _native_browse(filetypes: list[tuple]) -> str:
    """Open a native Windows file dialog in a subprocess and return the selected path (or '')."""
    script = (
        "import tkinter as tk; from tkinter import filedialog; "
        "root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', True); "
        f"path = filedialog.askopenfilename(filetypes={filetypes!r}); "
        "root.destroy(); print(path)"
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Native file dialog failed: {e}")
        st.error(
            "Could not open the native file browser. "
            "Please paste the file path directly into the text field below."
        )
        return ''


def _native_browse_folder() -> str:
    """Open a native Windows folder-picker dialog and return the selected path (or '')."""
    script = (
        "import tkinter as tk; from tkinter import filedialog; "
        "root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', True); "
        "path = filedialog.askdirectory(); "
        "root.destroy(); print(path)"
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Native folder dialog failed: {e}")
        st.error(
            "Could not open the native folder browser. "
            "Please paste the folder path directly into the text field below."
        )
        return ''


def _save_upload_to_tmp(uploaded_file, suffix: str) -> str:
    """Save a Streamlit UploadedFile to a temp file and return the path.

    Uses a deduplication signature stored in session state so the file is
    only written once per unique upload, not on every Streamlit rerun.
    """
    sig_key  = f'_upload_sig_{uploaded_file.name}_{uploaded_file.size}'
    path_key = f'_upload_path_{uploaded_file.name}_{uploaded_file.size}'

    if st.session_state.get(sig_key):
        return st.session_state[path_key]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    st.session_state[sig_key]  = True
    st.session_state[path_key] = tmp.name
    return tmp.name


def _validate_layer(criterion_key: str, file_path: str) -> None:
    """
    Validate and rescale a raster layer, storing results in session state.

    Parameters
    ----------
    criterion_key : str
        Criterion key (e.g., 'ecosystem_condition', 'anthropogenic_pressure').
    file_path : str
        Path to the uploaded raster file.

    Notes
    -----
    This function:
    - Calls validate_and_rescale_layer() if available
    - Stores the validation report in st.session_state['validation_reports']
    - If rescaling occurred, saves the rescaled array to a new temp file
      and updates st.session_state['criterion_raster_paths']
    - Uses a session state flag to avoid re-validation on every rerender
    """
    if not VALIDATION_AVAILABLE:
        logger.warning("Validation function not available — skipping validation")
        return

    # Check the source file's CRS directly (independent of rescaling/validation
    # below) so a missing or non-EPSG:3035 CRS is surfaced visibly in the UI
    # instead of being silently assumed/left uncorrected.
    crs_warning = None
    try:
        import rasterio as _rasterio_crs_check
        with _rasterio_crs_check.open(file_path) as _src:
            _orig_crs = _src.crs
        if _orig_crs is None:
            crs_warning = "No CRS embedded in file — assumed EPSG:3035. Please verify this is correct."
        elif _orig_crs.to_epsg() != 3035:
            crs_warning = (
                f"File CRS is EPSG:{_orig_crs.to_epsg()}, not EPSG:3035 — "
                "results may be spatially incorrect."
            )
    except Exception as e:
        logger.warning(f"Could not read CRS for {criterion_key}: {e}")

    # CLC land use values are categorical integer codes — no rescaling needed
    if criterion_key == 'landuse':
        logger.info("Skipping validation for CLC land use layer (categorical codes, no rescaling)")
        st.session_state['validation_reports'][criterion_key] = {
            'criterion': criterion_key,
            'original_min': None,
            'original_max': None,
            'expected_min': 111,
            'expected_max': 523,
            'rescaled': False,
            'method': 'none',
            'warning': None,
            'crs_warning': crs_warning
        }
        st.session_state[f"{criterion_key}_validated_path"] = file_path
        return

    # Check if we've already validated this specific file path
    validation_key = f"{criterion_key}_validated_path"
    if st.session_state.get(validation_key) == file_path:
        # Already validated this exact file — skip
        return

    # Validate and rescale
    with st.spinner(f"Validating {criterion_key.replace('_', ' ')}..."):
        try:
            # Load the raster
            array, profile = load_raster(file_path)

            # Validate and rescale
            rescaled_array, _, report = validate_and_rescale_layer(
                array=array,
                criterion_key=criterion_key,
                profile=profile
            )

            # Store validation report (augmented with the CRS check from above)
            report['crs_warning'] = crs_warning
            st.session_state['validation_reports'][criterion_key] = report

            # If rescaling occurred, save to new temp file and update path
            if report['rescaled']:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                    temp_path = tmp.name

                # Write rescaled array to the temp file
                import rasterio as _rasterio
                with _rasterio.open(temp_path, 'w', **profile) as dst:
                    dst.write(rescaled_array, 1)

                # Update path in session state
                st.session_state['criterion_raster_paths'][criterion_key] = temp_path
                logger.info(f"Rescaled {criterion_key} saved to {temp_path}")

            # Mark this path as validated
            st.session_state[validation_key] = file_path

            logger.info(f"Validation complete for {criterion_key}: {report}")

        except Exception as e:
            logger.error(f"Validation failed for {criterion_key}: {e}")
            # Store error report
            st.session_state['validation_reports'][criterion_key] = {
                'criterion': criterion_key,
                'error': str(e),
                'rescaled': False
            }


def _save_project_ini(filepath: str) -> None:
    """Save current layer paths to a .ini project file.

    Parameters
    ----------
    filepath : str
        Output .ini file path.
    """
    config = configparser.ConfigParser()

    # NUTS section
    config['nuts'] = {}
    nuts_path = st.session_state.get('nuts_file')
    if nuts_path:
        config['nuts']['path'] = str(nuts_path)

    # Study area section — Eurostat-selected NUTS2 region (Country/NUTS2 Region
    # dropdowns in the sidebar). Only meaningful when not using a local NUTS
    # file override above, and not a manual bounding box ('CUSTOM').
    study_area_nuts_id = st.session_state.get('parameters', {}).get('study_area_nuts_id')
    if study_area_nuts_id and study_area_nuts_id != 'CUSTOM':
        config['study_area'] = {'nuts_id': str(study_area_nuts_id)}

    # WDPA section
    config['wdpa'] = {}
    wdpa_path = st.session_state.get('wdpa_file')
    original_wdpa = st.session_state.get('_original_wdpa_path', wdpa_path)
    if original_wdpa:
        config['wdpa']['path'] = str(original_wdpa)

    # Criterion rasters section
    config['rasters'] = {}
    raster_paths = st.session_state.get('criterion_raster_paths', {})
    original_paths = st.session_state.get('_original_raster_paths', {})
    for key, path in raster_paths.items():
        # Prefer the original on-disk path over temp copies
        config['rasters'][key] = str(original_paths.get(key, path))

    # Settings section
    config['settings'] = {
        'exclude_marine_pa': str(st.session_state.get('exclude_marine_pa', True))
    }

    # Analysis parameters — all sidebar settings (weights, thresholds,
    # aggregation method, gap/proximity bonuses, sensitivity & delineation
    # settings, PA exclusion). See _PARAM_TYPES for the full list.
    analysis_params = st.session_state.get('parameters', {})
    if analysis_params:
        config['parameters'] = {}
        for key, ptype in _PARAM_TYPES.items():
            if key not in analysis_params or analysis_params[key] is None:
                continue
            value = analysis_params[key]
            if ptype is list:
                config['parameters'][key] = ','.join(str(v) for v in value)
            else:
                config['parameters'][key] = str(value)

    with open(filepath, 'w') as f:
        config.write(f)
    logger.info(f"Project saved to {filepath}")


def _load_project_ini(filepath: str) -> dict:
    """Load layer paths from a .ini project file.

    Parameters
    ----------
    filepath : str
        Input .ini file path.

    Returns
    -------
    dict
        Dictionary with keys: 'wdpa_path', 'raster_paths', 'settings'.
        Missing files are reported but non-fatal.
    """
    config = configparser.ConfigParser()
    config.read(filepath)

    result = {
        'nuts_path': None, 'nuts_id': None, 'wdpa_path': None,
        'raster_paths': {}, 'settings': {}, 'parameters': {}, 'errors': []
    }

    # NUTS
    nuts_path = config.get('nuts', 'path', fallback=None)
    if nuts_path and Path(nuts_path).exists():
        result['nuts_path'] = nuts_path
    elif nuts_path:
        result['errors'].append(f"NUTS file not found: {nuts_path}")

    # Study area — Eurostat-selected NUTS2 region ID (e.g. "FRF1")
    result['nuts_id'] = config.get('study_area', 'nuts_id', fallback=None)

    # WDPA
    wdpa_path = config.get('wdpa', 'path', fallback=None)
    if wdpa_path and Path(wdpa_path).exists():
        result['wdpa_path'] = wdpa_path
    elif wdpa_path:
        result['errors'].append(f"WDPA file not found: {wdpa_path}")

    # Rasters
    if config.has_section('rasters'):
        for key, path in config.items('rasters'):
            if Path(path).exists():
                result['raster_paths'][key] = path
            else:
                result['errors'].append(f"{key}: file not found — {path}")

    # Settings
    if config.has_section('settings'):
        for key, value in config.items('settings'):
            result['settings'][key] = value

    # Analysis parameters — sidebar settings (weights, thresholds, method, etc.)
    if config.has_section('parameters'):
        for key, ptype in _PARAM_TYPES.items():
            if not config.has_option('parameters', key):
                continue
            raw = config.get('parameters', key)
            try:
                if ptype is bool:
                    result['parameters'][key] = raw.strip().lower() == 'true'
                elif ptype is list:
                    result['parameters'][key] = [v for v in raw.split(',') if v]
                else:
                    result['parameters'][key] = ptype(raw)
            except ValueError:
                result['errors'].append(f"Invalid value for parameter '{key}': {raw!r}")

    logger.info(f"Project loaded from {filepath}: {len(result['raster_paths'])} rasters")
    return result


def render():
    """
    Render data upload interface with validation summary.

    All uploaded files are stored in session state for consumption by
    the ④ and ⑤ analysis tabs. No analytical code is executed here.

    Session state keys set:
    -----------------------
    wdpa_file : str or None
        Path to uploaded WDPA file (GeoPackage, Shapefile, or GeoJSON).
    criterion_raster_paths : dict
        Dictionary mapping criterion names to file paths:
        - 'ecosystem_condition': path to GeoTIFF
        - 'regulating_es': path to GeoTIFF
        - 'anthropogenic_pressure': path to GeoTIFF
        - 'cultural_es': path to GeoTIFF
        - 'provisioning_es': path to GeoTIFF
        - 'landuse': path to GeoTIFF
    data_ready : bool
        True if all required layers are uploaded, False otherwise.
    """
    st.header("Input Data Upload")
    st.markdown(
        """
        Upload all required layers before running the analysis.
        **All rasters must be in EPSG:3035** (LAEA Europe) with consistent resolution.
        """
    )

    # ===================================================================
    # Output Folder — where analysis exports (reports, GeoTIFF, shapefiles,
    # CSV, PDF, etc.) are saved in addition to the browser download. Set
    # first, since the .ini project file below defaults to being saved here.
    # ===================================================================
    st.subheader("Output Folder")
    st.caption(
        "Optional. When set, every export button across the app (④ Protection "
        "Network Diagnostic and ⑤ OECM Favourability Analysis) also saves a "
        "copy directly to this folder — no need "
        "to pick a save location in the browser each time. The project "
        "(.ini) file below also defaults to being saved here."
    )

    if _IS_LOCAL:
        col_out_browse, col_out_clear = st.columns(2)
        with col_out_browse:
            if st.button("Browse…", key='output_dir_browse'):
                chosen = _native_browse_folder()
                if chosen:
                    st.session_state['output_dir'] = chosen
                    st.rerun()
        with col_out_clear:
            if st.button("✕ Clear", key='output_dir_clear'):
                st.session_state['output_dir'] = ''
                st.rerun()

        # Single source of truth: the text input's own key IS 'output_dir' —
        # the same key read everywhere else in the app (save_and_download(),
        # the .ini default path, etc.). The Browse/Clear buttons above just
        # set this same key before rerunning. Using a *separate* key here
        # (as an earlier version of this code did) is a classic Streamlit
        # trap: once a keyed widget exists, its `value=` argument is only
        # used on first creation — writing to a *different* session_state
        # key from a button click never reaches an already-initialised
        # widget, so Browse would appear to work but get silently reverted
        # by this text input on the very next rerun.
        output_dir_input = st.text_input(
            "Output folder path",
            key='output_dir',
            help="Or paste a folder path directly.",
        )
        if output_dir_input and not (Path(output_dir_input).exists() and Path(output_dir_input).is_dir()):
            st.caption("⚠️ Folder not found.")
    else:
        # Cloud: no local filesystem to save into — browser download only.
        st.caption(
            "Not available in this deployment — exports are offered as browser "
            "downloads only."
        )

    if st.session_state.get('output_dir'):
        st.caption(f"📁 Exports will also be saved to `{st.session_state['output_dir']}`")

    st.markdown("---")

    # ===================================================================
    # Project Save / Load
    # ===================================================================
    st.subheader("Project Configuration")
    st.caption(
        "Save or load layer paths and analysis settings (weights, thresholds, "
        "aggregation method, etc.) to avoid reconfiguring everything on each "
        "restart. The .ini file stores **file paths and parameter values "
        "only** — no data is copied."
    )

    col_load, col_save = st.columns(2)

    with col_load:
        ini_file = st.file_uploader(
            "Load project (.ini)",
            type=['ini'],
            key='project_ini_upload',
            help="Load a previously saved .ini project file to restore all layer paths."
        )
        if ini_file is not None:
            # Deduplicate: only process if this is a different .ini than last time.
            # Using getvalue() (not read()) so the content is always available even
            # when Streamlit reruns with the same uploaded file still in the widget.
            _ini_sig = f"{ini_file.name}:{ini_file.size}"
            if st.session_state.get('_loaded_ini_sig') != _ini_sig:
                # Write to temp so configparser can read it
                ini_content = ini_file.getvalue().decode('utf-8')
                tmp_ini = Path(tempfile.gettempdir()) / 'oecm_project_load.ini'
                tmp_ini.write_text(ini_content)

                project = _load_project_ini(str(tmp_ini))

                # Apply loaded paths to session state
                if 'criterion_raster_paths' not in st.session_state:
                    st.session_state['criterion_raster_paths'] = {}
                if 'validation_reports' not in st.session_state:
                    st.session_state['validation_reports'] = {}
                if '_original_raster_paths' not in st.session_state:
                    st.session_state['_original_raster_paths'] = {}

                # NUTS — set widget key so the text input renders with the path
                if project.get('nuts_path'):
                    st.session_state['nuts_file'] = project['nuts_path']
                    st.session_state['nuts_direct_path'] = project['nuts_path']
                    st.session_state.pop('nuts_gdf_cache', None)

                # Study area — Eurostat-selected NUTS2 region (Country/NUTS2
                # Region dropdowns in the sidebar). Picked up once by
                # render_sidebar() on its next run to restore the selection.
                if project.get('nuts_id'):
                    st.session_state['_pending_nuts_id'] = project['nuts_id']

                # WDPA — set widget key so the text input renders with the path
                if project['wdpa_path']:
                    st.session_state['wdpa_file'] = project['wdpa_path']
                    st.session_state['_original_wdpa_path'] = project['wdpa_path']
                    st.session_state['wdpa_direct_path'] = project['wdpa_path']

                # Rasters — set widget keys so text inputs don't clear paths
                for key, path in project['raster_paths'].items():
                    st.session_state['criterion_raster_paths'][key] = path
                    st.session_state['_original_raster_paths'][key] = path
                    st.session_state[f'{key}_direct_path'] = path
                    _validate_layer(key, path)

                # Settings
                if project['settings'].get('exclude_marine_pa'):
                    st.session_state['exclude_marine_pa'] = (
                        project['settings']['exclude_marine_pa'].lower() == 'true'
                    )

                # Analysis parameters (weights, thresholds, aggregation method,
                # gap/proximity bonuses, sensitivity & delineation settings,
                # PA exclusion). Picked up once by render_sidebar() on its
                # next run to pre-seed each widget's session state.
                if project.get('parameters'):
                    st.session_state['_pending_sidebar_params'] = project['parameters']

                # Mark this .ini as processed so we don't re-run on every rerender
                st.session_state['_loaded_ini_sig'] = _ini_sig

                # Report
                n_loaded = len(project['raster_paths'])
                wdpa_ok = project['wdpa_path'] is not None
                n_params = len(project.get('parameters', {}))
                st.success(
                    f"Project loaded: {n_loaded} raster(s)"
                    f"{', WDPA' if wdpa_ok else ''}"
                    f"{f', {n_params} analysis parameter(s)' if n_params else ''}"
                )
                if project['errors']:
                    for err in project['errors']:
                        st.warning(f"Missing: {err}")

                # One rerun to let widgets pick up the new session-state values.
                # The deduplication flag above prevents this from looping.
                st.rerun()
            else:
                # Already processed — just show confirmation
                n_loaded = len(st.session_state.get('criterion_raster_paths', {}))
                wdpa_ok  = st.session_state.get('wdpa_file') is not None
                st.success(
                    f"Project active: {n_loaded} raster(s)"
                    f"{', WDPA' if wdpa_ok else ''}"
                )

    with col_save:
        _default_ini_dir = st.session_state.get('output_dir') or str(Path.home())
        save_path = st.text_input(
            "Save project as (.ini)",
            value=str(Path(_default_ini_dir) / 'oecm_project.ini'),
            help="Choose a path to save the current layer configuration. "
                 "Defaults to the Output Folder above, when set.",
        )
        if st.button("Save Project"):
            try:
                _save_project_ini(save_path)
                st.success(f"Project saved to `{save_path}`")
            except Exception as e:
                st.error(f"Failed to save project: {e}")

    # Define required layers (used throughout the render function)
    required_rasters = [
        ('ecosystem_condition', 'Ecosystem condition'),
        ('regulating_es', 'Regulating ES capacity'),
        ('anthropogenic_pressure', 'Anthropogenic pressure'),
        ('cultural_es', 'Cultural ES capacity'),
        ('provisioning_es', 'Provisioning ES capacity'),
        ('landuse', 'Land use / land cover')
    ]

    st.markdown("---")

    # ===================================================================
    # NUTS boundaries (study area selector)
    # ===================================================================
    st.subheader("NUTS Boundaries (Study Area)")
    st.caption(
        "Provide a local NUTS file (NUTS1/2/3). Accepted formats: GeoPackage (.gpkg), GeoJSON, "
        "or a **ZIP archive** containing the Shapefile components (.shp + .dbf + .shx + .prj). "
        "If left empty, the tool falls back to the Eurostat online service."
    )

    if _IS_LOCAL:
        col_nuts_browse, col_nuts_clear = st.columns(2)
        with col_nuts_browse:
            if st.button("Browse…", key='nuts_browse'):
                chosen = _native_browse([
                    ("Vector layers", "*.gpkg *.geojson *.zip"),
                    ("All files", "*.*"),
                ])
                if chosen:
                    st.session_state['nuts_direct_path'] = chosen
                    st.session_state['nuts_file'] = chosen
                    st.session_state.pop('nuts_gdf_cache', None)
                    st.rerun()
        with col_nuts_clear:
            if st.button("✕ Clear", key='nuts_clear'):
                st.session_state['nuts_direct_path'] = ''
                st.session_state['nuts_file'] = None
                st.session_state.pop('nuts_gdf_cache', None)
                st.rerun()

        with st.expander("Enter path manually"):
            st.text_input(
                "NUTS file path",
                key='nuts_direct_path',
                placeholder=r"C:\data\NUTS_RG_01M_2021_3035.zip",
                help="Path to a local NUTS GeoPackage (.gpkg), GeoJSON, or ZIP archive.",
                label_visibility="collapsed",
            )

        nuts_path_input = st.session_state.get('nuts_direct_path', '')
        if nuts_path_input:
            p = Path(nuts_path_input)
            if p.exists():
                st.session_state['nuts_file'] = str(p)
                st.caption(f"✅ {p.name}")
            else:
                st.caption("⚠️ File not found — check the path in the expander above.")
                st.session_state['nuts_file'] = None
        else:
            st.caption("No file selected.")
            st.session_state['nuts_file'] = None

    else:
        # Cloud: use Streamlit file uploader
        nuts_upload = st.file_uploader(
            "Upload NUTS file",
            type=['gpkg', 'geojson', 'zip'],
            key='nuts_upload_widget',
            help="GeoPackage, GeoJSON, or ZIP archive of a Shapefile. Must contain NUTS_ID, LEVL_CODE, NUTS_NAME columns.",
            label_visibility="collapsed",
        )
        if nuts_upload is not None:
            suffix = Path(nuts_upload.name).suffix or '.gpkg'
            tmp_path = _save_upload_to_tmp(nuts_upload, suffix)
            st.session_state['nuts_file'] = tmp_path
            st.session_state.pop('nuts_gdf_cache', None)
            st.caption(f"✅ {nuts_upload.name}")
        else:
            st.caption("No file selected.")
            st.session_state['nuts_file'] = None

    st.markdown("---")

    # ===================================================================
    # WDPA section (for ④ Protection Network Diagnostic)
    # ===================================================================
    st.subheader("Protected Areas Network (WDPA)")

    st.caption(
        "Supported formats: GeoPackage (.gpkg), GeoJSON (.geojson), Shapefile (.shp), ZIP archive (.zip). "
        "Click **Browse…** or paste a path directly. Use **✕ Clear** to remove and reload a different file."
    )

    if _IS_LOCAL:
        col_wdpa_browse, col_wdpa_clear = st.columns(2)
        with col_wdpa_browse:
            if st.button("Browse…", key='wdpa_browse'):
                chosen = _native_browse([
                    ("Vector layers", "*.gpkg *.geojson *.shp *.zip"),
                    ("All files", "*.*"),
                ])
                if chosen:
                    st.session_state['wdpa_direct_path'] = chosen
                    st.session_state.pop('pa_gdf', None)  # force reload
        with col_wdpa_clear:
            if st.button("✕ Clear", key='wdpa_clear'):
                st.session_state['wdpa_direct_path'] = ''
                st.session_state['wdpa_file'] = None
                st.session_state.pop('_original_wdpa_path', None)
                st.session_state.pop('pa_gdf', None)

        with st.expander("Enter path manually"):
            st.text_input(
                "WDPA file path",
                key='wdpa_direct_path',
                placeholder=r"C:\data\wdpa.gpkg",
                help="Full path to the WDPA GeoPackage (.gpkg), GeoJSON, Shapefile, or ZIP on disk.",
                label_visibility="collapsed",
            )

        prev_wdpa = st.session_state.get('wdpa_file')
        wdpa_path_input = st.session_state.get('wdpa_direct_path', '')
        if wdpa_path_input:
            p = Path(wdpa_path_input)
            if p.exists():
                if prev_wdpa != str(p):
                    st.session_state.pop('pa_gdf', None)
                st.session_state['wdpa_file'] = str(p)
                st.session_state['_original_wdpa_path'] = str(p)
                st.caption(f"✅ {p.name}")
            else:
                st.caption("⚠️ File not found — check the path in the expander above.")
                st.session_state['wdpa_file'] = None
                st.session_state.pop('_original_wdpa_path', None)
        else:
            st.caption("No file selected.")
            st.session_state['wdpa_file'] = None
            st.session_state.pop('_original_wdpa_path', None)

    else:
        # Cloud: use Streamlit file uploader
        st.caption("Upload a GeoPackage (.gpkg), GeoJSON, or ZIP of a Shapefile. "
                   "For large WDPA exports, pre-clip to your study area before uploading.")
        wdpa_upload = st.file_uploader(
            "Upload WDPA file",
            type=['gpkg', 'geojson', 'shp', 'zip'],
            key='wdpa_upload_widget',
            label_visibility="collapsed",
        )
        prev_wdpa = st.session_state.get('wdpa_file')
        if wdpa_upload is not None:
            suffix = Path(wdpa_upload.name).suffix or '.gpkg'
            tmp_path = _save_upload_to_tmp(wdpa_upload, suffix)
            if prev_wdpa != tmp_path:
                st.session_state.pop('pa_gdf', None)
            st.session_state['wdpa_file'] = tmp_path
            st.session_state['_original_wdpa_path'] = tmp_path
            st.caption(f"✅ {wdpa_upload.name}")
        else:
            st.caption("No file selected.")
            st.session_state['wdpa_file'] = None
            st.session_state.pop('_original_wdpa_path', None)

    def _clear_pa_gdf():
        """Reset cached PA GeoDataFrame when marine filter changes."""
        st.session_state.pop('pa_gdf', None)

    st.checkbox(
        "Exclude marine protected areas (REALM = 'Marine')",
        value=st.session_state.get('exclude_marine_pa', True),
        key='exclude_marine_pa',
        on_change=_clear_pa_gdf,
        help=(
            "Removes fully marine PAs (WDPA REALM column = 'Marine') from the analysis. "
            "Coastal and terrestrial PAs are kept. Change requires re-running the diagnostic."
        )
    )

    st.markdown("---")

    # ===================================================================
    # Criterion rasters (for ⑤ OECM Favourability Analysis)
    # ===================================================================
    st.subheader("Multi-Criteria Evaluation Layers")
    if _IS_LOCAL:
        st.caption(
            "Click **Browse…** or paste a path for each GeoTIFF (EPSG:3035). "
            "Paths are saved as-is in the .ini project file — no temporary copies."
        )
    else:
        st.caption(
            "Upload each GeoTIFF (EPSG:3035). "
            "Large files (> 200 MB) should be pre-clipped to the study area before uploading."
        )

    # Initialize session state dicts
    if 'criterion_raster_paths' not in st.session_state:
        st.session_state['criterion_raster_paths'] = {}
    if 'validation_reports' not in st.session_state:
        st.session_state['validation_reports'] = {}

    # Initialize original paths dict
    if '_original_raster_paths' not in st.session_state:
        st.session_state['_original_raster_paths'] = {}

    def _clear_raster(criterion_key, widget_key):
        """Remove a raster from all session state dicts."""
        st.session_state[widget_key] = ''
        st.session_state['criterion_raster_paths'].pop(criterion_key, None)
        st.session_state['_original_raster_paths'].pop(criterion_key, None)
        st.session_state['validation_reports'].pop(criterion_key, None)
        st.session_state.pop(f"{criterion_key}_validated_path", None)

    def _register_raster(criterion_key, file_path):
        """Store a valid raster path and trigger validation if the path changed."""
        prev = st.session_state['criterion_raster_paths'].get(criterion_key)
        st.session_state['criterion_raster_paths'][criterion_key] = file_path
        st.session_state['_original_raster_paths'][criterion_key] = file_path
        if prev != file_path:
            _validate_layer(criterion_key, file_path)

    # Helper: layer name as heading → file input (Browse on local, uploader on Cloud)
    def _layer_uploader(criterion_key, label, help_text):
        widget_key = f'{criterion_key}_direct_path'

        # ① Layer name as a prominent heading above the controls
        st.markdown(f"**{label}**")

        if _IS_LOCAL:
            # ── Local Windows: native Browse button + collapsible path input ──
            col_browse, col_clear = st.columns(2)
            with col_browse:
                if st.button("Browse…", key=f'{criterion_key}_browse'):
                    chosen = _native_browse([
                        ("GeoTIFF", "*.tif *.tiff"),
                        ("All files", "*.*"),
                    ])
                    if chosen:
                        st.session_state[widget_key] = chosen
            with col_clear:
                if st.button("✕ Clear", key=f'{criterion_key}_clear'):
                    _clear_raster(criterion_key, widget_key)

            with st.expander("Enter path manually"):
                st.text_input(
                    label,
                    key=widget_key,
                    placeholder=r"C:\data\file.tif",
                    help=help_text,
                    label_visibility="collapsed",
                )

            current_path = st.session_state.get(widget_key, '')
            if current_path:
                p = Path(current_path)
                if p.exists():
                    _register_raster(criterion_key, str(p))
                    st.caption(f"✅ {p.name}")
                else:
                    st.caption("⚠️ File not found — check the path in the expander above.")
                    _clear_raster(criterion_key, widget_key)
            else:
                st.caption("No file selected.")
                st.session_state['criterion_raster_paths'].pop(criterion_key, None)
                st.session_state['_original_raster_paths'].pop(criterion_key, None)
                st.session_state['validation_reports'].pop(criterion_key, None)

        else:
            # ── Cloud: Streamlit file uploader ──
            uploaded = st.file_uploader(
                label,
                type=['tif', 'tiff'],
                key=f'{criterion_key}_upload_widget',
                help=help_text,
                label_visibility="collapsed",
            )
            if uploaded is not None:
                tmp_path = _save_upload_to_tmp(uploaded, '.tif')
                _register_raster(criterion_key, tmp_path)
                st.caption(f"✅ {uploaded.name}")
            else:
                st.caption("No file selected.")
                st.session_state['criterion_raster_paths'].pop(criterion_key, None)
                st.session_state['_original_raster_paths'].pop(criterion_key, None)
                st.session_state['validation_reports'].pop(criterion_key, None)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Group A — Ecological Integrity")
        _layer_uploader(
            'ecosystem_condition', "Ecosystem condition [0–1]",
            "Normalised ecosystem condition index (0 = degraded, 1 = pristine)."
        )
        _layer_uploader(
            'regulating_es', "Regulating ES capacity [0–1]",
            "Regulating ecosystem services capacity (e.g. carbon storage, water regulation)."
        )
        _layer_uploader(
            'anthropogenic_pressure', "Anthropogenic pressure (hab/km²)",
            "Human population density or composite anthropogenic pressure index. Raw values accepted."
        )

    with col_right:
        st.markdown("#### Group B — Co-benefits")
        _layer_uploader(
            'cultural_es', "Cultural ES capacity [0–1]",
            "Cultural ecosystem services capacity (e.g. recreation, landscape aesthetics)."
        )
        st.markdown("#### Group C — Production Function")
        _layer_uploader(
            'provisioning_es', "Provisioning ES capacity [0–1]",
            "Provisioning ecosystem services capacity (e.g. food, timber, water supply)."
        )
        _layer_uploader(
            'landuse', "Land use / land cover — CLC 2018 (categorical)",
            "Corine Land Cover 2018 GeoTIFF (up to 2 GB supported). "
            "Download free from https://land.copernicus.eu/pan-european/corine-land-cover"
        )

    st.markdown("---")

    # ===================================================================
    # Layer Validation section
    # ===================================================================
    if VALIDATION_AVAILABLE:
        st.subheader("Layer Validation")

        validation_reports = st.session_state.get('validation_reports', {})
        uploaded_rasters = st.session_state.get('criterion_raster_paths', {})

        # Check if any layers have been uploaded
        if uploaded_rasters:
            # Display validation status for each layer in required_rasters
            for key, label in required_rasters:
                col_layer, col_status = st.columns([3, 1])

                with col_layer:
                    st.markdown(f"**{label}**")

                    if key in validation_reports:
                        report = validation_reports[key]

                        # Check for validation error
                        if 'error' in report:
                            st.error(f"Validation failed: {report['error']}")
                        else:
                            # Build display message
                            orig_min = report.get('original_min', 'N/A')
                            orig_max = report.get('original_max', 'N/A')
                            exp_min = report.get('expected_min', 'N/A')
                            exp_max = report.get('expected_max', 'N/A')

                            # Format ranges
                            if isinstance(orig_min, (int, float)) and isinstance(orig_max, (int, float)):
                                orig_range = f"[{orig_min:.3f}, {orig_max:.3f}]"
                            else:
                                orig_range = f"[{orig_min}, {orig_max}]"

                            if isinstance(exp_min, (int, float)) and isinstance(exp_max, (int, float)):
                                if exp_min == 0 and exp_max == 1:
                                    exp_range = "[0, 1]"
                                else:
                                    exp_range = f"[{exp_min:.3f}, {exp_max:.3f}]"
                            else:
                                exp_range = str(exp_max) if exp_max == exp_min else f"[{exp_min}, {exp_max}]"

                            st.caption(f"Original: {orig_range} → Expected: {exp_range}")

                            if report.get('warning'):
                                st.warning(f"⚠️ {report['warning']}")

                            if report.get('crs_warning'):
                                st.warning(f"⚠️ {report['crs_warning']}")

                    elif key in uploaded_rasters:
                        st.caption("Uploaded — validation pending")
                    else:
                        st.caption("Not uploaded")

                with col_status:
                    if key in validation_reports:
                        report = validation_reports[key]

                        if 'error' in report:
                            st.error("⚪ Error")
                        elif report.get('rescaled'):
                            st.info("🔵 Auto-rescaled")
                        elif report.get('warning') or report.get('crs_warning'):
                            st.warning("🟡 Warning")
                        else:
                            st.success("🟢 OK")
                    elif key in uploaded_rasters:
                        st.info("⏳ Pending")
                    else:
                        st.error("⚪ Not uploaded")

                st.markdown("---")

        else:
            st.info("Upload raster layers above to see validation results.")

    st.markdown("---")

    # ===================================================================
    # Validation summary
    # ===================================================================
    st.subheader("Upload Status")

    # Check upload status
    uploaded_rasters = st.session_state.get('criterion_raster_paths', {})
    wdpa_uploaded = st.session_state.get('wdpa_file') is not None

    col_status1, col_status2 = st.columns(2)

    with col_status1:
        st.markdown("**Protected Areas:**")
        if wdpa_uploaded:
            st.success("✓ WDPA layer uploaded")
        else:
            st.error("✗ WDPA layer missing")

    with col_status2:
        st.markdown("**MCE Criterion Rasters:**")
        missing_rasters = []
        for key, label in required_rasters:
            if key in uploaded_rasters:
                st.success(f"✓ {label}")
            else:
                st.error(f"✗ {label}")
                missing_rasters.append(label)

    # Overall validation status
    all_rasters_uploaded = len(missing_rasters) == 0
    data_ready_module1 = wdpa_uploaded
    data_ready_module2 = all_rasters_uploaded

    # Store validation flags in session state
    st.session_state['data_ready_module1'] = data_ready_module1
    st.session_state['data_ready_module2'] = data_ready_module2

    st.markdown("---")

    # ===================================================================
    # Validation button and status message
    # ===================================================================
    st.subheader("Ready to Proceed")

    col_ready1, col_ready2 = st.columns([2, 1])

    with col_ready1:
        if data_ready_module1 and data_ready_module2:
            st.success(
                "All required layers uploaded successfully! "
                "You may now proceed to the **④ Protection Network Diagnostic** and "
                "**⑤ OECM Favourability Analysis** tabs."
            )
            st.session_state['data_ready'] = True
        elif data_ready_module1 and not data_ready_module2:
            st.info(
                "WDPA layer uploaded — **④ Protection Network Diagnostic** is ready. "
                f"Upload {len(missing_rasters)} remaining raster(s) to enable **⑤ OECM Favourability Analysis**."
            )
            st.session_state['data_ready'] = False
        elif not data_ready_module1 and data_ready_module2:
            st.info(
                "All MCE rasters uploaded — **⑤ OECM Favourability Analysis** is ready. "
                "Upload WDPA layer to enable **④ Protection Network Diagnostic**."
            )
            st.session_state['data_ready'] = False
        else:
            st.warning(
                "Upload all required layers to proceed. "
                "At minimum, upload WDPA for **④ Protection Network Diagnostic** or "
                "all 6 rasters for **⑤ OECM Favourability Analysis**."
            )
            st.session_state['data_ready'] = False

    # ===================================================================
    # Data validation notes
    # ===================================================================
    with st.expander("Data Requirements & Validation"):
        st.markdown(
            """
            ### Raster Requirements
            All criterion rasters must satisfy:
            - **CRS:** EPSG:3035 (Lambert Azimuthal Equal Area — Europe)
            - **Resolution:** Consistent across all layers (recommended: 100m)
            - **Extent:** Identical bounding box and dimensions
            - **Format:** GeoTIFF (.tif or .tiff)
            - **Value range:**
              - Normalized criteria [0-1]: ecosystem condition, regulating ES, cultural ES, provisioning ES
              - Anthropogenic pressure: raw values (hab/km²)
              - Land use: categorical integer codes (e.g., CLC 2018)

            ### Vector Requirements (WDPA)
            - **Formats:** GeoPackage (.gpkg), Shapefile (.shp), or GeoJSON (.geojson)
            - **Required columns:**
              - `IUCN_MAX`: IUCN category (Ia, Ib, II, III, IV, V, VI, Not Reported)
              - `DESIG_TYPE`: Designation type (National, Regional, International)
              - `WDPA_NAME` or `name`: Protected area name
            - **Geometry:** Polygon or MultiPolygon in any CRS (will be reprojected to EPSG:3035)

            ### Pre-processing Notes
            - Raster alignment is performed automatically during MCE execution
            - WDPA geometries are classified into 5 protection classes based on IUCN category
            - Missing or invalid geometries are automatically filtered
            """
        )

    # ===================================================================
    # Session state debug info (collapsible)
    # ===================================================================
    with st.expander("Debug: Session State"):
        st.json({
            'wdpa_file': st.session_state.get('wdpa_file', 'Not uploaded'),
            'criterion_raster_paths': list(st.session_state.get('criterion_raster_paths', {}).keys()),
            'data_ready': st.session_state.get('data_ready', False),
            'data_ready_module1': st.session_state.get('data_ready_module1', False),
            'data_ready_module2': st.session_state.get('data_ready_module2', False)
        })
