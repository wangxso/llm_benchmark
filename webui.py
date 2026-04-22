#!/usr/bin/env python3
"""Launch the Streamlit WebUI"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set working directory to project root
os.chdir(project_root)

if __name__ == "__main__":
    import streamlit.web.cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(project_root / "src" / "webui" / "app.py"),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
    ]

    sys.exit(stcli.main())
