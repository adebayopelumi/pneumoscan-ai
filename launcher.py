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
    # Kill any leftover Streamlit on this port
    subprocess.run(
        ["pkill", "-f", "streamlit run app/streamlit_app.py"],
        capture_output=True
    )
    time.sleep(0.5)

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
