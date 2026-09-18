"""Make the iceberg_drift package importable when running the tests.

The package lives under src/ and is not installed into the environment, so
`import iceberg_drift` fails and every test module errors during collection -
the whole suite reports 5 collection errors rather than a single result.

Putting src/ on sys.path here fixes that without requiring an editable install,
which matters because this repository is run straight from a clone.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
