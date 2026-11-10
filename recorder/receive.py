#!/usr/bin/env python3
"""Receiver + static server for the clip recorder. stdlib only.

Run: python3 receive.py   (serves 127.0.0.1:8765, exposes it over tailnet HTTPS via `tailscale serve`)
"""
import argparse, json, re, subprocess, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
CLIPS = DATA / "clips"
MANIFEST = DATA / "manifest.jsonl"
LOCK = threading.Lock()
MAX_BYTES = 200 * 1024 * 1024
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
EXT = {"video/webm": "webm", "video/mp4": "mp4"}
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/index.html": ("index.html", "text/html; charset=utf-8"),
          "/sentences.txt": ("sentences.txt", "text/plain; charset=utf-8")}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body=b"", ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in STATIC:
            name, ctype = STATIC[path]
            return self._send(200, (HERE / name).read_bytes(), ctype)
        if path == "/count":
            n = sum(1 for _ in MANIFEST.open()) if MANIFEST.exists() else 0
            return self._send(200, json.dumps({"clips": n}))
        self._send(404, '{"error":"not found"}')

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/upload":
            return self._send(404, '{"error":"not found"}')
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        cid, text, mode = q.get("id", ""), q.get("text", "").strip(), q.get("mode", "")
        res = q.get("res", "")
        res = res if re.fullmatch(r"\d{2,5}x\d{2,5}", res) else ""
        ctype = self.headers.get("Content-Type", "").split(";")[0].strip()
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._send(411, '{"error":"length required"}')
        if not ID_RE.match(cid) or mode not in ("mouthed", "spoken") or not text or ctype not in EXT or not 0 < n <= MAX_BYTES:
            return self._send(400, '{"error":"bad id, mode, text, type or size"}')
        body = self.rfile.read(n)
        if len(body) != n:
            return self._send(400, '{"error":"short body"}')
        CLIPS.mkdir(parents=True, exist_ok=True)
        fname = f"{cid}.{EXT[ctype]}"
        with LOCK:
            tmp = CLIPS / (fname + ".part")
            tmp.write_bytes(body)
            tmp.replace(CLIPS / fname)
            with MANIFEST.open("a") as f:
                f.write(json.dumps({"id": cid, "text": text, "mode": mode, **({"res": res} if res else {})}, ensure_ascii=False) + "\n")
        print(f"saved {fname} ({n // 1024} KB) [{mode}] {text}", flush=True)
        self._send(200, json.dumps({"ok": True, "file": fname}))

    def log_message(self, *a):
        pass


def tailnet_host():
    try:
        d = json.loads(subprocess.check_output(["tailscale", "status", "--json"], text=True, timeout=10))
        return d["Self"]["DNSName"].rstrip("."), d["Self"].get("TailscaleIPs", [])
    except Exception:
        return None, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-serve", action="store_true", help="skip `tailscale serve` (use http://localhost only)")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    host, ips = tailnet_host()
    if a.no_serve or not host:
        print(f"Local only: http://localhost:{a.port}/  (camera works on localhost; for the Mac use: ssh -L {a.port}:localhost:{a.port} <spark>)")
    else:
        r = subprocess.run(["tailscale", "serve", "--bg", "--https=443", f"http://127.0.0.1:{a.port}"], capture_output=True, text=True)
        if r.returncode:
            print("tailscale serve failed:", (r.stderr or r.stdout).strip(), file=sys.stderr)
            print(f"Run manually: tailscale serve --bg --https=443 http://127.0.0.1:{a.port}", file=sys.stderr)
        print(f"\nOpen on the Mac:  https://{host}/\n")
    print(f"Saving to {DATA}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
