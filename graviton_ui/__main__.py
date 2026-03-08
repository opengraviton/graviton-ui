"""Entry-point: ``python -m graviton_ui`` or ``graviton-ui`` CLI."""

from __future__ import annotations

import argparse
import threading
import webbrowser


def main():
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
