# Auto OpenMatte HDR - GUI Control Surface

PySide6-based GUI for the Auto OpenMatte SDR-to-HDR video processing pipeline.

This is a **control surface only** - it orchestrates the existing backend pipeline
without implementing new processing, algorithms, decoders, or renderers.

## Requirements

- Python 3.10+
- PySide6 >= 6.5
- Windows with NVIDIA GPU (for full pipeline functionality)
- `auto-openmatte` backend package installed

## Installation

```bash
# From the gui/ directory:
pip install -e .

# Or install PySide6 separately if not bundled:
pip install PySide6>=6.5
```

## Usage

```bash
# Run as module:
python -m gui

# Or use the entry point (after pip install):
openmatte-gui
```

## Architecture

The GUI follows MVC separation:

```
gui/
├── models/          - Application state, project persistence, sync state
├── widgets/         - Qt widget panels (views)
├── controllers/     - Business logic adapters connecting GUI to backend
└── resources/       - Stylesheets, icons, and assets
```

### Panels (Workflow Tabs)

1. **Source** - HDR and OpenMatte file selection with drag-drop, ffprobe
   inspection, and compatibility validation.

2. **Sync** - Auto-sync runs as a *proposal* only. Shows offset N-1/N/N+1
   simultaneously. User adjusts with -1/+1, -10/+10, or manual entry, then
   confirms via LOCK. Offset is shot-level.

3. **Preview** - Dual-source display (HDR | OM) with 4 modes: side-by-side,
   overlay, wiper, checkerboard. Frame-accurate navigation and zoom controls.

4. **Shot Lock** - Displays all shots with lock status. Fitting gate opens
   only after all shots are locked.

5. **Output** - Current / Before-After / A-B toggle views of processed output.

6. **Quality** - Diagnostic metrics: seam chroma, hue shift, delta-E ITP,
   nonfinite pixels, out-of-range. No automatic global PASS/FAIL.

7. **Render** - Output path, codec, CRF, pixel format, resolution. Progress
   monitoring: FPS, ETA, elapsed time, VRAM usage.

### Design Principles

- **Control surface only**: The GUI calls backend functions, never implements
  new processing.
- **Auto-sync is a proposal**: Never automatically trusted. User must confirm.
- **Shot-level offset**: No per-frame independent sync.
- **Fitting gate**: Fitting cannot start until sync is locked.
- **No experimental models**: Only the verified production model is exposed.
- **No global PASS/FAIL**: Quality metrics are informational for the operator.

### Project File Format

Projects are saved as `.omhdr` JSON files containing:
- Source file paths (HDR and OpenMatte)
- Shot offsets and lock status
- Render parameters
- Project version

## Development

```bash
# Run syntax validation (no PySide6 required):
find gui/ -name '*.py' -exec python3 -c \
    "import ast, sys; ast.parse(open(sys.argv[1]).read()); print('OK:', sys.argv[1])" {} \;

# Run tests (requires PySide6):
pytest gui/tests/ -v
```
