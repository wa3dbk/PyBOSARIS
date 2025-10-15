"""
PyBOSARIS Command-Line Interface

Main entry point for the pybosaris command-line tool.
"""

import sys


def main():
    """Main entry point for the CLI."""
    print("PyBOSARIS v0.1.0")
    print("Python implementation of the BOSARIS Toolkit")
    print()
    print("Usage: pybosaris <command> [options]")
    print()
    print("Commands:")
    print("  calibrate    Calibrate scores to log-likelihood-ratios")
    print("  fuse         Fuse multiple systems")
    print("  evaluate     Compute performance metrics")
    print("  plot         Generate visualizations (DET, NBER)")
    print("  convert      Convert between file formats")
    print()
    print("Run 'pybosaris <command> --help' for more information")
    print()
    print("Note: Full CLI implementation coming soon!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
