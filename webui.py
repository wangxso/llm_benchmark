#!/usr/bin/env python3
"""Launch the Streamlit WebUI"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import streamlit.web.cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).parent / "src" / "webui" / "app.py"),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
    ]

    sys.exit(stcli.main())
