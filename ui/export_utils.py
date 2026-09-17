"""Shared export helper — browser download plus an optional direct save to a
user-chosen output folder (set once in ① Data Upload, session-wide)."""
import logging
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)


def save_and_download(
    label: str,
    data,
    file_name: str,
    mime: str,
    key: str = None,
    use_container_width: bool = False,
    disabled: bool = False,
) -> None:
    """
    Render a download button, and — if an output folder has been configured
    in ① Data Upload (``st.session_state['output_dir']``) — also write the
    file directly to that folder on disk.

    The download button always works regardless of the output folder, so
    this is purely additive: nothing changes for users who haven't set one.

    Parameters
    ----------
    data : bytes or str
        File content, exactly as you'd pass to st.download_button.
    """
    st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        key=key,
        use_container_width=use_container_width,
        disabled=disabled,
    )

    output_dir = st.session_state.get('output_dir')
    if output_dir and not disabled:
        try:
            out_path = Path(output_dir) / file_name
            out_bytes = data.encode('utf-8') if isinstance(data, str) else data
            out_path.write_bytes(out_bytes)
            st.caption(f"✅ Also saved to `{out_path}`")
        except Exception as e:
            logger.warning(f"Could not save '{file_name}' to output folder '{output_dir}': {e}")
            st.caption(f"⚠️ Could not save to output folder: {e}")
