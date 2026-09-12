#!/usr/bin/env python
"""
⚠️  DEPRECATED - DO NOT USE  ⚠️

This file has been moved to: dev_tools/synthetic_data/SYNTHETIC_seaice_generator.py

The new version requires explicit safety flags to prevent accidental use of fake data:
  --use-synthetic-data --i-understand-this-is-fake

Example:
  python dev_tools/synthetic_data/SYNTHETIC_seaice_generator.py \\
    --use-synthetic-data \\
    --i-understand-this-is-fake

This file will be removed in a future cleanup.
"""

import sys

print("\n" + "="*70)
print("⚠️  ERROR: This script has been deprecated")
print("="*70)
print("Synthetic data generation has been moved to prevent accidental use.")
print("\nNew location:")
print("  dev_tools/synthetic_data/SYNTHETIC_seaice_generator.py")
print("\nRequired flags:")
print("  --use-synthetic-data --i-understand-this-is-fake")
print("\nExample:")
print("  python dev_tools/synthetic_data/SYNTHETIC_seaice_generator.py \\")
print("    --use-synthetic-data \\")
print("    --i-understand-this-is-fake")
print("="*70)
sys.exit(1)
