#!/usr/bin/env python3
"""Build, validate, and serve a fresh static documentation preview."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from tempfile import TemporaryDirectory

from build import add_translation_arguments, build_site, selected_languages


class PreviewHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # Every refresh reads the current generated file, without a cached 304.
        for header in ("If-Modified-Since", "If-None-Match"):
            if header in self.headers:
                del self.headers[header]
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=1313)
    add_translation_arguments(parser)
    args = parser.parse_args()
    url = f"http://localhost:{args.port}/"
    # A new Hugo process and output directory avoid incremental render state.
    with TemporaryDirectory(prefix="comlrl-docs-static-") as directory:
        count = build_site(
            directory,
            base_url=url,
            languages=selected_languages(args),
            cache_only=args.cache_only,
            endpoint=args.ollama_url,
            model=args.model,
        )
        handler = partial(PreviewHandler, directory=directory)
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            print(f"Validated {count} HTML files. Static preview: {url}", flush=True)
            print(
                "After editing, restart this command and refresh the browser.",
                flush=True,
            )
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
