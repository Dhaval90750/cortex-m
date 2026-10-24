from __future__ import annotations

"""Serve the interactive local UI for observing real Cortex-M execution."""

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from cortex_m.assistant import RuleBasedCortex
from cortex_m.store import MemoryStore
from cortex_m.integrated import CortexMIMOSystem

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "visualizer" / "index.html"
LEXICON = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
DB_PATH = ROOT / "data" / "visualizer_memory.sqlite3"
MIMO_CHECKPOINT: Path | None = None
MIMO_GIST_CHECKPOINT: Path | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "CortexMVisualizer/1.0"

    def _host_allowed(self) -> bool:
        """Restrict Host routing to loopback names to reduce DNS-rebinding risk."""
        host_header = self.headers.get("Host", "")
        hostname = urlparse(f"//{host_header}").hostname
        return hostname in {"localhost", "127.0.0.1"}

    def _origin_allowed(self) -> bool:
        """Allow same-origin browser writes; permit Origin-less local CLI clients."""
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        parsed = urlparse(origin)
        return (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}
                and parsed.netloc == self.headers.get("Host"))

    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, obj: object) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(status, data, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if not self._host_allowed():
            self._json(403, {"error": "Only loopback Host values are accepted."})
            return
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            cortex = self._assistant()
            try:
                self._json(200, {"available": cortex.store.available,
                                 "facts": [self._record(f) for f in
                                           cortex.store.facts(include_superseded=True)]})
            except Exception as exc:
                self._json(200, {"available": False, "facts": [], "error": str(exc)})
            finally:
                cortex.close()
        elif path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 20_000:
                # Drain modest rejected bodies so clients receive the response
                # instead of a TCP reset caused by unread request bytes.
                if 20_000 < length <= 1_048_576:
                    self.rfile.read(length)
                self._json(413, {"error": "Request is too large"})
                return
            body = self.rfile.read(length)
            if not self._host_allowed() or not self._origin_allowed():
                self._json(403, {"error": "Cross-origin or non-loopback request rejected."})
                return
            content_type = self.headers.get_content_type()
            if content_type != "application/json":
                self._json(415, {"error": "Content-Type must be application/json."})
                return
            payload = json.loads(body or b"{}")
            cortex = self._assistant()
            try:
                if path == "/api/task":
                    mode, text = payload.get("mode"), str(payload.get("text", "")).strip()
                    if not text:
                        self._json(400, {"error": "Enter a statement or question first."})
                        return
                    if mode == "auto":
                        if isinstance(cortex, CortexMIMOSystem):
                            trace = cortex.run(text)
                        elif self._looks_like_question(text):
                            trace = cortex.trace_query(text)
                        else:
                            trace = cortex.trace_observe(text)
                            if not trace["accepted_count"]:
                                trace["kind"] = "unsupported"
                                trace["action"] = "not-supported"
                                trace["response"] = (
                                    "This prototype does not support that request yet. "
                                    "It can store supported explicit facts and answer from "
                                    "saved evidence; it does not provide general chat or "
                                    "generate or execute code."
                                )
                    elif mode == "statement":
                        trace = cortex.trace_observe(text)
                    elif mode == "question":
                        trace = cortex.trace_query(text)
                    else:
                        self._json(400, {"error": "Choose automatic, statement, or question mode."})
                        return
                    self._json(200, trace)
                elif path == "/api/clear":
                    count = cortex.store.clear_all()
                    self._json(200, {"deleted": count})
                else:
                    self._json(404, {"error": "Not found"})
            finally:
                cortex.close()
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": f"Invalid request: {exc}"})
        except Exception as exc:
            self._json(500, {"error": f"Execution error: {type(exc).__name__}: {exc}"})

    @staticmethod
    def _assistant() -> RuleBasedCortex | CortexMIMOSystem:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(DB_PATH)
        if MIMO_CHECKPOINT is not None:
            return CortexMIMOSystem(keys=LEXICON["keys"], values=LEXICON["values"],
                                    store=store, checkpoint_path=MIMO_CHECKPOINT,
                                    gist_checkpoint_path=MIMO_GIST_CHECKPOINT)
        if MIMO_GIST_CHECKPOINT is not None:
            return CortexMIMOSystem(keys=LEXICON["keys"], values=LEXICON["values"],
                                    store=store, gist_checkpoint_path=MIMO_GIST_CHECKPOINT)
        return RuleBasedCortex(LEXICON["keys"], LEXICON["values"], store)

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        return bool(re.search(
            r"^\s*(?:what|who|when|where|why|how|which|whose|whom|can|could|do|does|did|is|are|am|was|were|will|would|should|has|have|had)\b|\?\s*$",
            text, re.IGNORECASE,
        ))

    @staticmethod
    def _record(fact: object) -> dict:
        return {"key": fact.key, "value": fact.value, "status": fact.status,
                "source_turn": fact.source_turn, "evidence_text": fact.evidence_text,
                "correction": fact.correction}

    def log_message(self, format: str, *args: object) -> None:
        print(f"[visualizer] {self.address_string()} {format % args}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: local machine only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mimo-checkpoint", type=Path, default=None,
                        help="Enable the optional trained M2M N=8 specialist for structured MIMO requests")
    parser.add_argument("--mimo-gist-checkpoint", type=Path, default=None,
                        help="Enable MIMO inference conditioned on a dynamically retrieved Cortex-M mode gist")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("the visualizer is local-only; --host must be 127.0.0.1 or localhost")
    global MIMO_CHECKPOINT
    global MIMO_GIST_CHECKPOINT
    if args.mimo_checkpoint is not None:
        MIMO_CHECKPOINT = args.mimo_checkpoint.resolve()
        if not MIMO_CHECKPOINT.is_file():
            parser.error(f"MIMO checkpoint not found: {MIMO_CHECKPOINT}")
    if args.mimo_gist_checkpoint is not None:
        MIMO_GIST_CHECKPOINT = args.mimo_gist_checkpoint.resolve()
        if not MIMO_GIST_CHECKPOINT.is_file():
            parser.error(f"Gist-conditioned MIMO checkpoint not found: {MIMO_GIST_CHECKPOINT}")
    if not HTML.exists():
        raise FileNotFoundError(f"Visualizer page missing: {HTML}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Cortex-M Visualizer: http://{args.host}:{args.port}")
    print(f"Persistent demo memory: {DB_PATH}")
    if MIMO_CHECKPOINT is not None:
        print(f"M2M-SparseRoute specialist enabled: {MIMO_CHECKPOINT}")
    if MIMO_GIST_CHECKPOINT is not None:
        print(f"Gist-conditioned MIMO specialist enabled: {MIMO_GIST_CHECKPOINT}")
    print("Press Ctrl+C to stop. This server binds to localhost by default.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping visualizer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
