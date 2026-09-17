"""Config loaders shared across the app (used by ui/tab_parameters.py).

The parameter widgets themselves used to live here, rendered into the
Streamlit sidebar — they now live in ui/tab_parameters.py as a regular tab.
This module only keeps the two cached YAML loaders, which tab_parameters.py
imports.
"""
import streamlit as st
import yaml
from pathlib import Path


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
