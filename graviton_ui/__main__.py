"""Entry-point: ``python -m graviton_ui`` or ``graviton-ui`` CLI."""

from __future__ import annotations

# MPS bellek limitini kaldır (Apple Silicon'da büyük modeller için)
# UYARI: Sistem donması riski var; diğer ağır uygulamaları kapatın
import os
if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import signal
import sys
import threading
import webbrowser


def main():
    """Launch Graviton UI — opens a browser with the chat interface."""
    parser = argparse.ArgumentParser(
        prog="graviton-ui",
        description="Launch the Graviton chat interface in your browser.",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to serve on (default: 7860)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't auto-open the browser",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=[url]).start()

    if sys.platform != "win32":
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    print(f"\n  Graviton UI running at \033[1;35m{url}\033[0m\n")

    import uvicorn
    uvicorn.run(
        "graviton_ui.server:app",
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
