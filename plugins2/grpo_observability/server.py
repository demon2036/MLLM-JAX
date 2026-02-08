from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from plugins2.grpo_observability.types import RunRequest


DEFAULT_HTML = """<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>plugins2 GRPO Observability</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      textarea, input { width: 100%; margin-top: 6px; margin-bottom: 10px; }
      button { padding: 8px 16px; }
      table { border-collapse: collapse; width: 100%; margin-top: 10px; }
      th, td { border: 1px solid #ddd; padding: 6px; font-size: 12px; }
      th { background: #f5f5f5; }
      .card { border: 1px solid #ddd; border-radius: 6px; padding: 10px; margin-bottom: 12px; }
      .mono { font-family: ui-monospace, Menlo, Consolas, monospace; }
      .error { color: #b00020; }
    </style>
  </head>
  <body>
    <h2>plugins2 · GRPO Token Observability</h2>
    <label>System Prompt</label>
    <textarea id=\"system_prompt\" rows=\"5\"></textarea>
    <label>User Prompt</label>
    <textarea id=\"user_prompt\" rows=\"4\"></textarea>
    <label>Label (GSM8K answer)</label>
    <input id=\"label\" type=\"text\" />
    <label>K (num generations)</label>
    <input id=\"k\" type=\"number\" value=\"8\" min=\"1\" />
    <button onclick=\"runOnce()\">Run GRPO + Backprop</button>
    <pre id=\"status\" class=\"mono\"></pre>
    <div id=\"results\"></div>

    <script>
      async function runOnce() {
        const status = document.getElementById('status');
        const results = document.getElementById('results');
        status.textContent = 'Running...';
        results.innerHTML = '';
        try {
          const payload = {
            system_prompt: document.getElementById('system_prompt').value,
            user_prompt: document.getElementById('user_prompt').value,
            label: document.getElementById('label').value,
            k: Number(document.getElementById('k').value || 8),
          };
          const response = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.error || ('HTTP ' + response.status));
          }
          status.textContent = JSON.stringify(data.summary, null, 2);
          renderSamples(data.samples || []);
        } catch (error) {
          status.innerHTML = '<span class="error">' + String(error) + '</span>';
        }
      }

      function renderSamples(samples) {
        const results = document.getElementById('results');
        samples.forEach((sample) => {
          const card = document.createElement('div');
          card.className = 'card';
          const reward = sample.reward || {};
          card.innerHTML = `
            <div><b>Sample #${sample.sample_index}</b></div>
            <div><b>Reward total:</b> ${Number(reward.total || 0).toFixed(4)}</div>
            <div><b>Answer:</b> <span class="mono">${escapeHtml(sample.answer || '')}</span></div>
          `;

          const table = document.createElement('table');
          table.innerHTML = `
            <thead>
              <tr>
                <th>pos</th><th>token_id</th><th>token_text</th><th>prob</th><th>logprob</th><th>grad_logprob</th><th>loss_contrib</th>
              </tr>
            </thead>
          `;
          const tbody = document.createElement('tbody');
          (sample.tokens || []).forEach((token) => {
            const row = document.createElement('tr');
            row.innerHTML = `
              <td>${token.position}</td>
              <td>${token.token_id}</td>
              <td class="mono">${escapeHtml(String(token.token_text || ''))}</td>
              <td>${Number(token.prob || 0).toExponential(4)}</td>
              <td>${Number(token.logprob || 0).toFixed(6)}</td>
              <td>${Number(token.grad_logprob || 0).toExponential(4)}</td>
              <td>${Number(token.loss_contrib || 0).toExponential(4)}</td>
            `;
            tbody.appendChild(row);
          });
          table.appendChild(tbody);
          card.appendChild(table);
          results.appendChild(card);
        });
      }

      function escapeHtml(str) {
        return str
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')
          .replaceAll('"', '&quot;')
          .replaceAll("'", '&#039;');
      }

      fetch('/api/engine')
        .then((resp) => resp.json())
        .then((data) => {
          const status = document.getElementById('status');
          status.textContent = JSON.stringify(data, null, 2);
        });
    </script>
  </body>
</html>
"""



def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")



def _read_html_template(path: str | None) -> str:
    if path is None:
        return DEFAULT_HTML
    file_path = Path(path)
    if not file_path.exists():
        return DEFAULT_HTML
    return file_path.read_text(encoding="utf-8")



def serve_observability_http(
    *,
    engine: Any,
    host: str,
    port: int,
    html_template_path: str | None = None,
) -> None:
    html = _read_html_template(html_template_path)

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Any, status_code: int = 200) -> None:
            body = _json_bytes(payload)
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body: str, status_code: int = 200) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(html)
                return
            if parsed.path == "/api/healthz":
                self._send_json({"ok": True})
                return
            if parsed.path == "/api/engine":
                self._send_json(engine.describe())
                return
            self._send_json({"error": f"not found: {parsed.path}"}, status_code=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/run":
                self._send_json({"error": f"not found: {parsed.path}"}, status_code=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8") if body else "{}")

                request = RunRequest(
                    system_prompt=str(payload.get("system_prompt", "")),
                    user_prompt=str(payload.get("user_prompt", "")),
                    label=str(payload.get("label", "")),
                    k=None if payload.get("k") in (None, "") else int(payload.get("k")),
                )
                result = engine.run_request(request)
                self._send_json(result)
            except Exception as e:
                self._send_json(
                    {
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                    status_code=500,
                )

    server = ThreadingHTTPServer((host, int(port)), Handler)
    print(f"plugins2 observability server listening on http://{host}:{int(port)}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = ["serve_observability_http"]
