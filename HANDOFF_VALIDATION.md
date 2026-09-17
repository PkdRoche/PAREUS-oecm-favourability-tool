# Module 1 → Module 2 Weight Handoff Validation

## Overview

This document describes the weight handoff contract between Module 1 (Protected Area Network Diagnostic) and Module 2 (MCE Favourability Analysis).

## The Handoff Contract

### Module 1 Output

`propose_group_a_weights()` returns a dictionary with:
- **Keys**: Group A criterion IDs from `config/criteria_defaults.yaml`
  - `'ecosystem_condition'`
  - `'regulating_es'`
  - `'low_pressure'`
- **Values**: Weights derived from representativity deficits
- **Constraints**:
  - All values must be non-negative floats
  - Sum of values must equal 1.0 ± 1e-6
  - Zero-weight criteria should be excluded (optional, handled by formatter)

### Module 2 Input

`compute_favourability()` in `mce_engine.py` expects a `weights` dict with:
```python
weights = {
    'group_a_weights': {
        'ecosystem_condition': float,
        'regulating_es': float,
        'low_pressure': float
    },
    # ... other groups
}
```

The `group_a_weights` dict is passed to `compute_group_score()` which validates:
1. Keys match the criterion names in `criteria_arrays`
2. Weights sum to 1.0 (tolerance 1e-6)

## Data Flow

```
Module 1
  └─> propose_group_a_weights(ri_df, criterion_mapping)
       └─> Returns: {'ecosystem_condition': 0.35, 'regulating_es': 0.45, 'low_pressure': 0.20}

Handoff Validation (NEW)
  └─> validate_weight_handoff(weights, config_path)
       └─> Raises ValueError if contract violated
  └─> format_weights_for_mce(weights)
       └─> Returns normalised, filtered weights

UI Sidebar
  └─> Stores in st.session_state['proposed_group_a_weights']
  └─> User clicks "Apply Module 1 weight suggestions"
  └─> Maps to sidebar keys: w_condition, w_regulating_es, w_pressure

app.py
  └─> Reads sidebar parameters
  └─> Rebuilds weight dict:
       weights['group_a_weights'] = {
           'ecosystem_condition': parameters['w_condition'],
           'regulating_es': parameters['w_regulating_es'],
           'low_pressure': parameters['w_pressure']
       }

Module 2
  └─> mce_engine.compute_favourability(weights=weights, ...)
       └─> compute_group_score(group_a_arrays, group_a_weights, ...)
            └─> Validates keys match and sum to 1.0
```

## Validation Functions

### `validate_weight_handoff(weight_dict, config_path)`

Validates that a weight dictionary conforms to the handoff contract.

**Checks:**
1. Keys match expected Group A criteria from config
2. All weights are non-negative
3. Weights sum to 1.0 ± 1e-6
4. (Warning) No zero-weight criteria included

**Raises:**
- `FileNotFoundError`: Config file not found
- `ValueError`: Contract violation with detailed error message

### `format_weights_for_mce(weight_dict)`

Formats weights for Module 2 consumption.

**Actions:**
1. Filters out zero-weight criteria
2. Normalises to exactly sum = 1.0 (corrects floating point errors)
3. Returns clean dictionary

**Returns:**
- `dict[str, float]`: Formatted weights ready for MCE engine

## Usage Examples

### Valid Handoff

```python
from modules.module1_protected_areas.handoff import (
    validate_weight_handoff,
    format_weights_for_mce
)

# Module 1 produces weights
weights = {
    'ecosystem_condition': 0.35,
    'regulating_es': 0.45,
    'low_pressure': 0.20
}

# Validate handoff
config_path = "config/criteria_defaults.yaml"
validate_weight_handoff(weights, config_path)  # Passes

# Format for MCE
mce_weights = format_weights_for_mce(weights)

# Use in Module 2
results = mce_engine.compute_favourability(
    ...,
    weights={
        'group_a_weights': mce_weights,
        ...
    }
)
```

### Handling Errors

```python
# Mismatched keys
bad_weights = {
    'ecosystem_condition': 0.5,
    'unknown_criterion': 0.5
}

try:
    validate_weight_handoff(bad_weights, config_path)
except ValueError as e:
    print(f"Handoff validation failed: {e}")
    # Output:
    # Weight keys do not match expected Group A criteria.
    #   Missing: ['low_pressure', 'regulating_es']
    #   Unexpected: ['unknown_criterion']
    #   Expected: ['ecosystem_condition', 'low_pressure', 'regulating_es']
```

```python
# Weights don't sum to 1.0
bad_sum = {
    'ecosystem_condition': 0.35,
    'regulating_es': 0.40,
    'low_pressure': 0.20
}

try:
    validate_weight_handoff(bad_sum, config_path)
except ValueError as e:
    print(f"Handoff validation failed: {e}")
    # Output:
    # Weights must sum to 1.0 (tolerance 1e-06).
    # Got sum = 0.95000000. Difference: 5.00e-02
```

## Integration Points

### In UI (tab_module1.py)

After proposing weights, validate before storing:

```python
from modules.module1_protected_areas.handoff import (
    validate_weight_handoff,
    format_weights_for_mce
)

# Propose weights
proposed_weights = propose_group_a_weights(ri_df, criterion_mapping)

# Validate handoff
try:
    config_path = Path(__file__).parent.parent / "config" / "criteria_defaults.yaml"
    validate_weight_handoff(proposed_weights, str(config_path))
    formatted_weights = format_weights_for_mce(proposed_weights)

    # Store in session state
    st.session_state['proposed_group_a_weights'] = formatted_weights
    st.success("Weight suggestions validated and applied!")

except ValueError as e:
    st.error(f"Weight validation failed: {e}")
```

### In UI (sidebar.py)

When applying Module 1 suggestions, validate:

```python
if st.sidebar.button("Apply Module 1 weight suggestions"):
    proposed = st.session_state['proposed_group_a_weights']

    # Optional: Re-validate at application point
    try:
        config_path = Path(__file__).parent.parent / "config" / "criteria_defaults.yaml"
        validate_weight_handoff(proposed, str(config_path))

        # Map to sidebar parameter names
        st.session_state['group_a_applied'] = {
            'w_condition': proposed['ecosystem_condition'],
            'w_regulating_es': proposed['regulating_es'],
            'w_pressure': proposed['low_pressure']
        }

        st.sidebar.success("Module 1 weights applied to Group A!")
        st.rerun()

    except ValueError as e:
        st.sidebar.error(f"Invalid weights: {e}")
```

## Testing

Run validation tests:
```bash
cd tests
python -m pytest test_handoff.py -v
```

Or use the quick verification script:
```bash
python verify_handoff.py
```

## Files

- **Implementation**: `modules/module1_protected_areas/handoff.py`
- **Tests**: `tests/test_handoff.py`
- **Verification**: `verify_handoff.py`
- **This document**: `HANDOFF_VALIDATION.md`

## Notes

### Why This Matters

1. **Type Safety**: Ensures weight dicts have correct structure
2. **Early Error Detection**: Catches mismatches before MCE execution
3. **Clear Error Messages**: Helps users diagnose weight configuration issues
4. **Numerical Stability**: Handles floating point rounding errors
5. **Documentation**: Makes the contract explicit and testable

### Limitations

This validation only covers **Group A intra-weights**. It does NOT:
- Validate inter-group weights (W_A, W_B, W_C)
- Validate Group B or Group C weights
- Propose inter-group weight values (requires user judgement)

### Future Extensions

Consider adding:
- Validation for complete weight structure (all groups)
- Weight range constraints (e.g., min/max per criterion)
- Weight dependency checks (e.g., if X > threshold then Y must be < threshold)
- Serialisation/deserialisation for weight presets
