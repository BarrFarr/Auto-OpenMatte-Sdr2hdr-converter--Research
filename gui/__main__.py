"""
Entry point for running the GUI as a module: python -m gui
"""
import sys
from gui.app import create_application, run


def main():
    """Launch the Auto OpenMatte GUI application."""
    app = create_application(sys.argv)
    sys.exit(run(app))


if __name__ == "__main__":
    main()
