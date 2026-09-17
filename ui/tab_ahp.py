"""AHP (Analytic Hierarchy Process) weight calibration tab.

Provides pairwise comparison matrices for:
  - Inter-group weights  (W_A, W_B, W_C)         — 3×3
  - Group A intra-weights (condition, reg, press) — 3×3
  - Group C intra-weights (provisioning, landuse) — 2×2

Derived weights are written into session state and reflected in the
Parameters tab sliders.  CR < 0.10 is required before weights can be applied.
"""

import numpy as np
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Saaty scale
# ---------------------------------------------------------------------------
_SAATY_OPTIONS = [
    (1/9, "1/9 — Extremely less important"),
    (1/8, "1/8"),
    (1/7, "1/7 — Very strongly less important"),
    (1/6, "1/6"),
    (1/5, "1/5 — Strongly less important"),
    (1/4, "1/4"),
    (1/3, "1/3 — Moderately less important"),
    (1/2, "1/2"),
    (1,   "1   — Equally important"),
    (2,   "2"),
    (3,   "3   — Moderately more important"),
    (4,   "4"),
    (5,   "5   — Strongly more important"),
    (6,   "6"),
    (7,   "7   — Very strongly more important"),
    (8,   "8"),
    (9,   "9   — Extremely more important"),
]
_SAATY_VALUES  = [v for v, _ in _SAATY_OPTIONS]
_SAATY_LABELS  = [l for _, l in _SAATY_OPTIONS]
_DEFAULT_IDX   = _SAATY_VALUES.index(1)   # "Equally important"

# Random consistency index for n = 1..10 (Saaty 1980)
_RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
       6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


# ---------------------------------------------------------------------------
# AHP maths
# ---------------------------------------------------------------------------

def _build_matrix(values: list[float], n: int) -> np.ndarray:
    """Build full n×n AHP comparison matrix from upper-triangle values."""
    A = np.ones((n, n), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            v = values[idx]
            A[i, j] = v
            A[j, i] = 1.0 / v
            idx += 1
    return A


def _ahp_weights(A: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Compute AHP priority vector and consistency metrics.

    Returns
    -------
    weights : np.ndarray  normalised priority vector (sums to 1)
    CR      : float       consistency ratio  (< 0.10 = acceptable)
    lambda_max : float    principal eigenvalue
    """
    n = A.shape[0]
    # Column-normalise then row-average (geometric mean approximation)
    col_sums = A.sum(axis=0)
    A_norm = A / col_sums
    w = A_norm.mean(axis=1)
    w /= w.sum()

    # λ_max
    Aw = A @ w
    lambda_max = float(np.mean(Aw / w))

    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = _RI.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0

    return w, CR, lambda_max


# ---------------------------------------------------------------------------
# Widget helpers
# ---------------------------------------------------------------------------

def _pairwise_selector(label_i: str, label_j: str, key: str) -> float:
    """Single selectbox for one pairwise comparison (i vs j)."""
    saved = st.session_state.get(key, 1)
    try:
        current_idx = _SAATY_VALUES.index(saved)
    except ValueError:
        current_idx = _DEFAULT_IDX

    chosen_label = st.selectbox(
        f"**{label_i}** vs **{label_j}**",
        options=_SAATY_LABELS,
        index=current_idx,
        key=key,
    )
    value = _SAATY_VALUES[_SAATY_LABELS.index(chosen_label)]
    return value


def _matrix_block(title: str, criteria: list[str], key_prefix: str
                  ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Render pairwise comparison block and return (A, weights, CR, lambda_max).
    """
    st.markdown(f"#### {title}")
    n = len(criteria)
    values = []
    for i in range(n):
        for j in range(i + 1, n):
            v = _pairwise_selector(
                criteria[i], criteria[j],
                key=f"{key_prefix}_{i}_{j}"
            )
            values.append(v)

    A = _build_matrix(values, n)
    w, CR, lmax = _ahp_weights(A)
    return A, w, CR, lmax


def _cr_badge(CR: float) -> None:
    """Display CR with colour-coded feedback."""
    if CR < 0.10:
        st.success(f"Consistency Ratio CR = {CR:.3f} — acceptable (< 0.10)")
    elif CR < 0.20:
        st.warning(f"Consistency Ratio CR = {CR:.3f} — marginal (< 0.20 tolerated)")
    else:
        st.error(f"Consistency Ratio CR = {CR:.3f} — inconsistent, please revise judgements")


def _weight_table(criteria: list[str], weights: np.ndarray) -> None:
    """Display derived weights as a small dataframe."""
    df = pd.DataFrame({
        "Criterion": criteria,
        "AHP Weight": [f"{w:.4f}" for w in weights],
        "% share":    [f"{w*100:.1f}%" for w in weights],
    })
    st.dataframe(df, hide_index=True, use_container_width=False)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_tab_ahp() -> None:
    """Render the AHP Weight Calibration tab."""

    st.header("③ Weight Calibration — Analytic Hierarchy Process (AHP)")

    st.markdown(
        """
        AHP derives weights from **pairwise expert judgements** rather than
        direct slider assignment.  For each pair of criteria, select how much
        more (or less) important one criterion is relative to the other using
        Saaty's 1–9 scale.  The tool computes normalised weights and a
        **Consistency Ratio (CR)**; CR < 0.10 indicates coherent judgements.

        Clicking **Apply to MCE** transfers the derived weights to the
        **② Parameters** sliders and marks them as `[AHP]` source.
        """
    )

    st.info(
        "AHP calibration is optional.  If you skip this tab, **② Parameters** "
        "sliders retain their manual values."
    )

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Matrix 1 — Inter-group weights (3×3)
    # -----------------------------------------------------------------------
    with st.expander("Matrix 1 — Inter-group weights  (W_A, W_B, W_C)", expanded=True):
        st.caption(
            "Compare the three functional groups against each other.  "
            "W_A = ecological integrity, W_B = co-benefits, W_C = production function."
        )
        inter_criteria = [
            "W_A — Ecological integrity",
            "W_B — Co-benefits",
            "W_C — Production function",
        ]
        _, w_inter, cr_inter, lmax_inter = _matrix_block(
            "Inter-group comparison", inter_criteria, "ahp_inter"
        )
        _cr_badge(cr_inter)
        _weight_table(inter_criteria, w_inter)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Matrix 2 — Group A intra-group (3×3)
    # -----------------------------------------------------------------------
    with st.expander("Matrix 2 — Group A intra-weights  (ecosystem condition, regulating ES, low pressure)", expanded=True):
        st.caption(
            "Compare the three ecological integrity criteria within Group A."
        )
        group_a_criteria = [
            "Ecosystem condition",
            "Regulating ES capacity",
            "Low anthropogenic pressure",
        ]
        _, w_a, cr_a, lmax_a = _matrix_block(
            "Group A intra-group comparison", group_a_criteria, "ahp_group_a"
        )
        _cr_badge(cr_a)
        _weight_table(group_a_criteria, w_a)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Matrix 3 — Group C intra-group (2×2)
    # -----------------------------------------------------------------------
    with st.expander("Matrix 3 — Group C intra-weights  (provisioning ES, compatible land use)", expanded=True):
        st.caption(
            "Compare the two production-function criteria within Group C.  "
            "A 2×2 matrix always yields CR = 0 (perfectly consistent by definition)."
        )
        group_c_criteria = [
            "Provisioning ES capacity",
            "Compatible land use",
        ]
        _, w_c, cr_c, lmax_c = _matrix_block(
            "Group C intra-group comparison", group_c_criteria, "ahp_group_c"
        )
        _cr_badge(cr_c)
        _weight_table(group_c_criteria, w_c)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Apply button — only enabled when all CRs are acceptable
    # -----------------------------------------------------------------------
    st.subheader("Apply AHP Weights to MCE")

    all_cr_ok = (cr_inter < 0.10) and (cr_a < 0.10) and (cr_c < 0.10)
    any_cr_marginal = (
        (0.10 <= cr_inter < 0.20) or
        (0.10 <= cr_a    < 0.20) or
        (0.10 <= cr_c    < 0.20)
    )

    if all_cr_ok or any_cr_marginal:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if any_cr_marginal and not all_cr_ok:
                st.warning(
                    "One or more matrices have CR between 0.10 and 0.20.  "
                    "You may still apply, but consider revising judgements."
                )
            apply = st.button(
                "Apply AHP weights to Parameters",
                type="primary",
                use_container_width=True,
                disabled=not (all_cr_ok or any_cr_marginal),
            )
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.error(
                "One or more matrices are inconsistent (CR ≥ 0.20).  "
                "Please revise your pairwise judgements before applying."
            )
            apply = st.button(
                "Apply AHP weights to Parameters",
                type="primary",
                use_container_width=True,
                disabled=True,
            )

    if apply:
        # Write derived weights into session state keys that tab_parameters.py reads
        st.session_state['ahp_weights'] = {
            # Inter-group
            'W_A': float(w_inter[0]),
            'W_B': float(w_inter[1]),
            'W_C': float(w_inter[2]),
            # Group A
            'w_condition':     float(w_a[0]),
            'w_regulating_es': float(w_a[1]),
            'w_pressure':      float(w_a[2]),
            # Group C
            'w_provisioning_es':   float(w_c[0]),
            'w_landuse_compatible': float(w_c[1]),
        }
        st.session_state['ahp_source'] = True   # badge flag for the Parameters tab
        st.success(
            "AHP weights applied!  Switch to **② Parameters** — all weight sliders "
            "now reflect the AHP-derived values and are labelled **[AHP]**."
        )
        st.rerun()

    # -----------------------------------------------------------------------
    # Reset to manual
    # -----------------------------------------------------------------------
    if st.session_state.get('ahp_source', False):
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Reset to manual weights", use_container_width=True):
                st.session_state.pop('ahp_weights', None)
                st.session_state['ahp_source'] = False
                st.info("Weights reset to manual Parameters tab values.")
                st.rerun()

    # -----------------------------------------------------------------------
    # AHP methodology note
    # -----------------------------------------------------------------------
    with st.expander("Methodology — how AHP weights are computed"):
        st.markdown(
            """
            **Saaty's Analytic Hierarchy Process (1980)**

            1. For each pair of criteria *(i, j)*, the user assigns a value
               *a_ij* ∈ {1/9, …, 1, …, 9} indicating how many times more
               important criterion *i* is than criterion *j*.
               By construction *a_ji = 1 / a_ij*.

            2. The pairwise comparison matrix **A** is column-normalised.
               The row averages of the normalised matrix give the
               **priority vector** (weights).

            3. **Consistency Ratio** CR = CI / RI where:
               - CI = (λ_max − n) / (n − 1)  (consistency index)
               - λ_max = principal eigenvalue of **A**
               - RI = random consistency index for matrix size *n*
                 (0.58 for n=3, 0.90 for n=4 …)

            4. CR < **0.10** is the standard acceptance threshold.
               CR < 0.20 is sometimes tolerated in practice.

            **References:**
            - Saaty, T.L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill.
            - Malczewski, J. (1999). *GIS and Multicriteria Decision Analysis*.
              John Wiley & Sons.
            """
        )
