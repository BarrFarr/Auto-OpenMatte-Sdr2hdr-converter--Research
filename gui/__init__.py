"""
Auto OpenMatte SDR-to-HDR Converter - GUI Control Surface

This package provides a PySide6-based GUI that orchestrates the existing
auto_openmatte backend pipeline. It is a control surface only - it does NOT
implement new processing, algorithms, decoders, or renderers.

Architecture:
    models/     - Application state, project persistence, sync state
    widgets/    - Qt widget panels (Source, Sync, Preview, Shot Lock, etc.)
    controllers/ - Business logic adapters connecting GUI to backend
    resources/  - Stylesheets, icons, and other assets
"""

__version__ = "0.1.0"
__app_name__ = "Auto OpenMatte HDR"
