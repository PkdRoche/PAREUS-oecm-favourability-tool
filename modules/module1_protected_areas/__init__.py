"""Module 1 — Protection Network Diagnostic and Gap Analysis."""

from .wdpa_loader import (
    fetch_wdpa_api,
    load_wdpa_local,
    filter_to_extent,
    classify_iucn
)

from .coverage_stats import (
    compute_net_area,
    coverage_by_class,
    fragmentation_index,
    kmgbf_indicator
)

from .representativity import (
    cross_with_ecosystem_types,
    representativity_index,
    propose_group_a_weights
)

from .gap_analysis import (
    strict_gaps,
    qualitative_gaps,
    potential_corridors,
    export_gap_masks_as_raster
)

from .zonal_stats import (
    zonal_stats_by_pa_class,
    criterion_coverage_summary
)

__all__ = [
    'fetch_wdpa_api',
    'load_wdpa_local',
    'filter_to_extent',
    'classify_iucn',
    'compute_net_area',
    'coverage_by_class',
    'fragmentation_index',
    'kmgbf_indicator',
    'cross_with_ecosystem_types',
    'representativity_index',
    'propose_group_a_weights',
    'strict_gaps',
    'qualitative_gaps',
    'potential_corridors',
    'export_gap_masks_as_raster',
    'zonal_stats_by_pa_class',
    'criterion_coverage_summary'
]
