#!/usr/bin/env python3
"""
One-click launcher for the Customer Support Ticket Router.
Double-click this file (or run `python run_server.py`) to start the server.
The browser will open automatically.
"""

import os
import sys
import time
import webbrowser
from pathlib import Path


def check_python():
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or higher is required.")
        print(f"  You have Python {sys.version.split()[0]}")
        print("  Download the latest version: https://www.python.org/downloads/")
        input("\nPress Enter to exit...")
        sys.exit(1)


def check_dependencies():
    try:
        import fastapi
        import openenv
        import uvicorn
    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else "a dependency"
        print(f"Missing dependency: {missing}")
        print("\nInstalling dependencies...")
        os.system(f'"{sys.executable}" -m pip install -r server/requirements.txt')
        print()
        try:
            import fastapi
            import openenv
            import uvicorn
        except ImportError:
            print("ERROR: Failed to install dependencies.")
            print(f"Try running manually: pip install -r server/requirements.txt")
            input("\nPress Enter to exit...")
            sys.exit(1)


def find_available_port(start_port=8000, max_attempts=10):
    import socket
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


def main():
    print("=" * 60)
    print("  Customer Support Ticket Router")
    print("=" * 60)
    print()

    check_python()

    os.chdir(Path(__file__).parent)

    check_dependencies()

    port = find_available_port()
    url = f"http://localhost:{port}"

    print(f"Starting server on {url}")
    print("Opening browser...")
    print()
    print("Press Ctrl+C to stop the server.")
    print("=" * 60)
    print()

    webbrowser.open(url)

    from server.app import main as server_main
    server_main(host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
