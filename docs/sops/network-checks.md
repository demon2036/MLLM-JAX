# Network Checks SOPs

- **Title**: SOP: Verify network access and do quick web lookup
  **Prereqs**: Ubuntu Linux; `curl` available; Python 3.12.2; JAX not installed; GPU unknown (`nvidia-smi` not found)
  **Steps**:
  - `curl -I -s https://github.com | head`
  - `curl -s 'https://lite.duckduckgo.com/lite/?q=<query>' | head`
  **Expected Result**: Commands return HTTP headers / HTML content (network reachable)
  **Troubleshooting**: If blocked, check proxy env vars and retry with a different host
  **References**: N/A
