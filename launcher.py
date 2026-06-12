import threading
import subprocess
import sys
import os
import time
import urllib.request
import webview

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8501


def start_streamlit():
    os.chdir(BASE_DIR)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
        "--server.headless", "true",
        "--server.port", str(PORT),
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ])


def wait_for_server(timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}", timeout=1)
            return True
        except Exception:
            time.sleep(0.3)
    return False


if __name__ == "__main__":
    # Kill any process already occupying the port (including other Streamlit apps)
    result = subprocess.run(
        ["lsof", "-ti", f":{PORT}"],
        capture_output=True, text=True
    )
    pids = result.stdout.strip().splitlines()
    for pid in pids:
        subprocess.run(["kill", "-9", pid.strip()], capture_output=True)

    # Wait for the port to be fully released
    for _ in range(20):
        check = subprocess.run(["lsof", "-ti", f":{PORT}"], capture_output=True)
        if not check.stdout.strip():
            break
        time.sleep(0.3)

    # Start Streamlit in a background daemon thread
    t = threading.Thread(target=start_streamlit, daemon=True)
    t.start()

    # Wait for the server to be ready
    if not wait_for_server():
        print("Streamlit failed to start.", file=sys.stderr)
        sys.exit(1)

    # Open the native window
    window = webview.create_window(
        title="PneumoScan AI",
        url=f"http://localhost:{PORT}",
        width=1280,
        height=820,
        min_size=(900, 650),
        text_select=True,
    )
    webview.start(gui="cocoa")
