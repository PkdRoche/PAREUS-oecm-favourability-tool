"""
Protected Area Optimization — Streamlit App
=============================================

Re-implements, in an interactive Streamlit app, the workflow from the R script that:
  1. reads a planning-unit (PU) grid (GeoJSON/JSON with polygon geometries + attribute columns),
  2. derives cost/distance features and rescales them 0-1,
  3. filters out partial/too-small PUs,
  4. runs a reserve-design optimization (locked-in existing PAs, optional lockout classes,
     optional boundary length penalty), and
  5. maps + classifies the resulting solution (existing core PA / new core PA / proposed
     upgrade / other protected / not protected).

Because `prioritizr` (R) has no direct Python equivalent, the optimization is solved here as
an Integer Linear Program with PuLP/CBC (open source, no license needed):

  * "Maximize utility within a budget"  ~= prioritizr::add_max_utility_objective()
  * "Minimize cost to meet feature targets" ~= prioritizr::add_min_set_objective()

Boundary penalties (~= add_boundary_penalties()) are supported via a linearized adjacency
term. This is O(n_pairs) and can get slow for very large / high-resolution grids — a warning
and a PU-count guard are included; raise `MAX_PU_FOR_BOUNDARY` if you have the patience/CPU.

Run with:
    pip install -r requirements.txt
    streamlit run pa_optimization_app.py
"""

import json
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pulp
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="PA Optimization", layout="wide")

MAX_PU_FOR_BOUNDARY = 1500  # guard rail: pairwise adjacency gets expensive above this


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def zero_one_scale(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Min-max scale the given columns to [0, 1]. Mirrors the R `zero_one_scale()` helper."""
    df = df.copy()
    for c in cols:
        lo, hi = df[c].min(), df[c].max()
        df[c] = 0.0 if hi == lo else (df[c] - lo) / (hi - lo)
    return df


@st.cache_data(show_spinner=False)
def load_grid(file_bytes: bytes) -> gpd.GeoDataFrame:
    """Load an uploaded GeoJSON/JSON planning-unit grid into a GeoDataFrame."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    gdf = gpd.read_file(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    if gdf.crs is None:
        st.warning("Input has no CRS defined — assuming EPSG:4326 (lon/lat).")
        gdf = gdf.set_crs(4326)
    return gdf


def ensure_projected_area_km2(gdf: gpd.GeoDataFrame, area_col: str | None) -> gpd.GeoDataFrame:
    """Return a copy with an `area_km2` column, reusing an existing area field (assumed m^2)
    if one is supplied, otherwise computing it in an equal-area projection."""
    gdf = gdf.copy()
    if area_col and area_col in gdf.columns:
        gdf["area_km2"] = pd.to_numeric(gdf[area_col], errors="coerce") / 1e6
    else:
        gdf_ea = gdf.to_crs(6933)  # world equal-area, meters
        gdf["area_km2"] = gdf_ea.geometry.area / 1e6
    return gdf


def compute_adjacency(gdf: gpd.GeoDataFrame):
    """Return list of (i, j, shared_boundary_length_m) for touching PUs, and each PU's own
    perimeter (m). Uses an equal-area/metric CRS so lengths are meaningful."""
    gdf_m = gdf.to_crs(6933) if gdf.crs and gdf.crs.to_epsg() != 6933 else gdf
    perim = gdf_m.geometry.length.to_numpy()

    # candidate touching pairs via spatial join
    left = gdf_m.reset_index(drop=True)
    joined = gpd.sjoin(left, left, predicate="touches", how="inner")
    pairs = []
    seen = set()
    for i, j in zip(joined.index, joined["index_right"]):
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        shared = left.geometry.iloc[i].intersection(left.geometry.iloc[j]).length
        if shared > 0:
            pairs.append((key[0], key[1], shared))
    return pairs, perim


def solve_optimization(
    pu: gpd.GeoDataFrame,
    features: list[str],
    cost_col: str,
    objective: str,
    budget: float | None,
    targets_pct: dict | None,
    lockin_col: str | None,
    boundary_penalty: float,
):
    """Build and solve the ILP. Returns pu with an added 'solution' 0/1 column
    (only for the rows passed in — caller is responsible for filtering lockout classes first)."""

    n = len(pu)
    idx = list(pu.index)
    prob = pulp.LpProblem("pa_optimization", pulp.LpMaximize if objective == "max_utility" else pulp.LpMinimize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in idx}

    # locked-in constraints
    if lockin_col and lockin_col in pu.columns:
        for i in idx:
            val = pu.loc[i, lockin_col]
            if bool(val) if not pd.isna(val) else False:
                prob += x[i] == 1

    use_boundary = boundary_penalty > 0
    y = {}
    pairs, perim = [], np.zeros(n)
    if use_boundary:
        if n > MAX_PU_FOR_BOUNDARY:
            st.warning(
                f"Boundary penalty skipped: {n} planning units exceeds the "
                f"{MAX_PU_FOR_BOUNDARY}-unit guard rail for pairwise adjacency. "
                "Raise MAX_PU_FOR_BOUNDARY in the script if you want to force it."
            )
            use_boundary = False
        else:
            pairs, perim = compute_adjacency(pu)
            pos = {ix: k for k, ix in enumerate(idx)}
            for a, b, shared in pairs:
                key = (idx[a], idx[b])
                y[key] = pulp.LpVariable(f"y_{a}_{b}", lowBound=0, upBound=1)
                prob += y[key] <= x[key[0]]
                prob += y[key] <= x[key[1]]

    utility = {i: sum(pu.loc[i, f] for f in features) for i in idx}

    if objective == "max_utility":
        obj = pulp.lpSum(utility[i] * x[i] for i in idx)
        if use_boundary:
            perim_map = {idx[k]: perim[k] for k in range(n)}
            obj -= boundary_penalty * pulp.lpSum(perim_map[i] * x[i] for i in idx)
            for a, b, shared in pairs:
                obj += 2 * boundary_penalty * shared * y[(idx[a], idx[b])]
        prob += obj
        prob += pulp.lpSum(pu.loc[i, cost_col] * x[i] for i in idx) <= budget

    else:  # min_set: minimize cost subject to feature targets
        obj = pulp.lpSum(pu.loc[i, cost_col] * x[i] for i in idx)
        if use_boundary:
            perim_map = {idx[k]: perim[k] for k in range(n)}
            obj += boundary_penalty * pulp.lpSum(perim_map[i] * x[i] for i in idx)
            for a, b, shared in pairs:
                obj -= 2 * boundary_penalty * shared * y[(idx[a], idx[b])]
        prob += obj
        for f in features:
            total = pu[f].sum()
            target = (targets_pct.get(f, 0) / 100.0) * total
            prob += pulp.lpSum(pu.loc[i, f] * x[i] for i in idx) >= target

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    pu = pu.copy()
    pu["solution"] = [int(round(pulp.value(x[i]) or 0)) for i in idx]
    return pu, pulp.LpStatus[status]


def classify(pu: gpd.GeoDataFrame, lockin_col: str, n_pa_col: str | None) -> gpd.GeoDataFrame:
    pu = pu.copy()

    def _row(r):
        existing = bool(r[lockin_col]) if lockin_col in pu.columns and not pd.isna(r[lockin_col]) else False
        n_pa = r[n_pa_col] if n_pa_col and n_pa_col in pu.columns else 0
        n_pa = 0 if pd.isna(n_pa) else n_pa
        sol = r["solution"]
        if existing:
            return "existing core PA"
        if sol == 1 and n_pa == 0:
            return "new core PA"
        if sol == 1 and n_pa > 0:
            return "proposed upgrade existing PA"
        if sol == 0 and n_pa == 0:
            return "not protected"
        return "other protected areas"

    pu["core_pa_class"] = pu.apply(_row, axis=1)
    return pu


CLASS_COLORS = {
    "existing core PA": "#1a9850",
    "new core PA": "#66bd63",
    "proposed upgrade existing PA": "#fdae61",
    "other protected areas": "#a6d96a",
    "not protected": "#d9d9d9",
}


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

st.title("Protected Area Optimization")
st.caption(
    "Upload a planning-unit grid (GeoJSON/JSON), configure the optimization, "
    "solve, and map the resulting protected-area solution."
)

uploaded = st.file_uploader("Planning-unit grid (.json / .geojson)", type=["json", "geojson"])

if not uploaded:
    st.info("Upload a planning-unit grid to get started.")
    st.stop()

raw_gdf = load_grid(uploaded.getvalue())
all_cols = [c for c in raw_gdf.columns if c != raw_gdf.geometry.name]
numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(raw_gdf[c])]
categorical_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(raw_gdf[c])]
bool_like_cols = [
    c for c in all_cols
    if raw_gdf[c].dropna().isin([True, False, 0, 1]).all() and raw_gdf[c].notna().any()
]

st.success(f"Loaded {len(raw_gdf)} planning units, {len(all_cols)} attribute columns.")

with st.sidebar:
    st.header("1. Columns")
    class_col = st.selectbox("Land-cover / class column (e.g. lulc_name)", options=["(none)"] + categorical_cols)
    class_col = None if class_col == "(none)" else class_col

    cost_col = st.selectbox("Cost column", options=numeric_cols)
    dist_col = st.selectbox("Distance column (optional, inverted)", options=["(none)"] + numeric_cols)
    dist_col = None if dist_col == "(none)" else dist_col

    default_feats = [c for c in numeric_cols if c not in {cost_col, dist_col}][:2]
    feature_cols = st.multiselect("Feature columns to optimize for", options=numeric_cols, default=default_feats)

    lockin_col = st.selectbox("Locked-in column (existing core PA, boolean)", options=["(none)"] + bool_like_cols)
    lockin_col = None if lockin_col == "(none)" else lockin_col

    n_pa_col = st.selectbox("Existing PA count column (for classification, optional)", options=["(none)"] + numeric_cols)
    n_pa_col = None if n_pa_col == "(none)" else n_pa_col

    area_col = st.selectbox("Area column in m² (optional, else computed)", options=["(none)"] + numeric_cols)
    area_col = None if area_col == "(none)" else area_col

    st.header("2. Filters")
    lockout_vals = []
    target_val = "(all)"
    if class_col:
        classes = sorted(raw_gdf[class_col].dropna().unique().tolist())
        lockout_vals = st.multiselect("Lockout classes (excluded entirely)", options=classes)
        target_val = st.selectbox("Restrict to a single class (optional)", options=["(all)"] + classes)

    min_area_km2 = st.number_input("Minimum PU area (km²) — drops partial PUs", value=1.2, min_value=0.0, step=0.1)

    st.header("3. Objective")
    objective_label = st.radio(
        "Objective",
        options=["Maximize utility within an area budget", "Minimize cost to meet feature targets"],
    )
    objective = "max_utility" if objective_label.startswith("Maximize") else "min_set"

    budget = None
    targets_pct = None
    if objective == "max_utility":
        budget = st.number_input("Area budget (km²)", min_value=0.0, value=50.0, step=1.0)
    else:
        st.caption("Target = % of the grid-wide total for each feature that must be captured.")
        targets_pct = {f: st.slider(f"Target for {f} (%)", 0, 100, 30) for f in feature_cols}

    boundary_penalty = st.number_input(
        "Boundary penalty (0 = off)", min_value=0.0, value=0.0, step=0.0001, format="%.4f",
        help="Higher values favor spatially compact solutions. Uses shared-boundary length; "
             "skipped automatically above the PU-count guard rail.",
    )

    run = st.button("Run optimization", type="primary")

# --------------------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------------------

if run:
    if not feature_cols:
        st.error("Select at least one feature column.")
        st.stop()
    if objective == "max_utility" and (budget is None or budget <= 0):
        st.error("Set a positive area budget.")
        st.stop()

    pu = raw_gdf.copy()

    # derive inverse cost / inverse distance, then rescale 0-1 (mirrors the R script)
    scale_cols = []
    if dist_col:
        pu["inv_dist"] = np.where(pu[dist_col] == 0, 0, 1 / pu[dist_col])
        scale_cols.append("inv_dist")
    pu["inv_cost"] = np.where(pu[cost_col] > 0, 1 / pu[cost_col], 0)
    scale_cols.append("inv_cost")
    pu = zero_one_scale(pu, scale_cols)

    pu = ensure_projected_area_km2(pu, area_col)
    pu = pu[pu["area_km2"] > min_area_km2].copy()

    if class_col and lockout_vals:
        pu = pu[~pu[class_col].isin(lockout_vals)].copy()
    if class_col and target_val != "(all)":
        pu = pu[pu[class_col] == target_val].copy()

    if pu.empty:
        st.error("No planning units left after filtering — relax the filters.")
        st.stop()

    pu = pu.reset_index(drop=True)

    with st.spinner(f"Solving ({len(pu)} planning units)..."):
        solved, status = solve_optimization(
            pu=pu,
            features=feature_cols,
            cost_col="area_km2",
            objective=objective,
            budget=budget,
            targets_pct=targets_pct,
            lockin_col=lockin_col,
            boundary_penalty=boundary_penalty,
        )

    st.session_state["solved"] = solved
    st.session_state["status"] = status
    st.session_state["lockin_col"] = lockin_col
    st.session_state["n_pa_col"] = n_pa_col

if "solved" in st.session_state:
    solved = st.session_state["solved"]
    status = st.session_state["status"]
    st.subheader(f"Solver status: {status}")

    if status != "Optimal":
        st.warning("Solver did not reach an optimal solution — check budget/targets feasibility.")

    classified = classify(solved, st.session_state.get("lockin_col") or "", st.session_state.get("n_pa_col"))

    selected_area = classified.loc[classified["solution"] == 1, "area_km2"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Planning units selected", int(classified["solution"].sum()))
    c2.metric("Area protected (km²)", f"{selected_area:,.1f}")
    c3.metric("Total PUs considered", len(classified))

    st.markdown("**Classification breakdown**")
    st.dataframe(classified["core_pa_class"].value_counts().rename_axis("class").reset_index(name="count"))

    # Map
    geo_wgs84 = classified.to_crs(4326)
    center = geo_wgs84.geometry.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="CartoDB positron")

    def style_fn(feat):
        cls = feat["properties"]["core_pa_class"]
        return {"fillColor": CLASS_COLORS.get(cls, "#999999"), "color": "#555555", "weight": 0.5, "fillOpacity": 0.75}

    folium.GeoJson(
        geo_wgs84[["core_pa_class", "solution", "area_km2", geo_wgs84.geometry.name]],
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=["core_pa_class", "area_km2"]),
    ).add_to(m)

    legend_html = "".join(
        f'<div><span style="background:{color};width:12px;height:12px;display:inline-block;'
        f'margin-right:6px;"></span>{cls}</div>'
        for cls, color in CLASS_COLORS.items()
    )
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;'
        f'padding:10px;border-radius:6px;font-size:12px;box-shadow:0 0 6px rgba(0,0,0,0.3);">{legend_html}</div>'
    ))

    st_folium(m, width=None, height=600)

    st.download_button(
        "Download solution (GeoJSON)",
        data=classified.to_json(),
        file_name="pa_optimization_solution.geojson",
        mime="application/geo+json",
    )
    st.download_button(
        "Download solution (CSV, no geometry)",
        data=classified.drop(columns=[classified.geometry.name]).to_csv(index=False),
        file_name="pa_optimization_solution.csv",
        mime="text/csv",
    )
