# Electerm SOPs

- **Title**: SOP: Reinstall Electerm (user-local AppImage, Ubuntu 23.10+ workaround)
  **Prereqs**: Ubuntu 25.04 (x86_64, Wayland); `curl` 8.16.0; Python 3.13.9; network access to GitHub
  **Steps**:
  - Uninstall existing user-local install (AppImage + desktop entry):
    - `rm -f ~/.local/bin/electerm ~/.local/bin/electerm.AppImage ~/.local/share/applications/electerm.desktop`
  - Find the latest Linux x86_64 AppImage URL from GitHub:
    - `tmp=$(mktemp); curl -sSL -H 'Accept: application/vnd.github+json' -H 'User-Agent: mllm-jax-sop' https://api.github.com/repos/electerm/electerm/releases/latest -o "$tmp"; python3 -c "import json, re; j=json.load(open('$tmp','r',encoding='utf-8')); print('tag_name=', j.get('tag_name')); print('appimages='); [print(u) for u in [a.get('browser_download_url','') for a in j.get('assets',[])] if re.search(r'(?i)\\.AppImage$',u) and re.search(r'(?i)linux',u) and re.search(r'(?i)(x86_64|x64|amd64)',u)]"; rm -f "$tmp"`
  - Install the AppImage (this run used v2.3.198):
    - `ELECTERM_URL="https://github.com/electerm/electerm/releases/download/v2.3.198/electerm-2.3.198-linux-x86_64.AppImage"`
    - `mkdir -p ~/.local/bin ~/.local/share/applications`
    - `curl -fL "$ELECTERM_URL" -o ~/.local/bin/electerm.AppImage`
    - `chmod +x ~/.local/bin/electerm.AppImage`
  - Fix launch on Ubuntu 23.10+ (AppArmor userns restriction): wrap with `ELECTRON_DISABLE_SANDBOX=1`
    - `cat > ~/.local/bin/electerm <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export ELECTRON_DISABLE_SANDBOX=1
exec "$HOME/.local/bin/electerm.AppImage" "$@"
EOF`
    - `chmod +x ~/.local/bin/electerm`
  - Desktop launcher entry:
    - `printf '%s\n' '[Desktop Entry]' 'Type=Application' 'Name=Electerm' 'Comment=Terminal/SSH/SFTP client' 'Exec=/home/john/.local/bin/electerm' 'Icon=electerm' 'Terminal=false' 'Categories=Network;TerminalEmulator;' 'StartupWMClass=electerm' > ~/.local/share/applications/electerm.desktop`
  - Minimal verification:
    - `electerm --version`
    - `timeout 10 electerm || true`
  **Expected Result**: Running `electerm` (or launching “Electerm” from the app menu) starts without the Chromium sandbox fatal error
  **Troubleshooting**:
  - If you see `The SUID sandbox helper binary was found, but is not configured correctly ... chrome-sandbox ... mode 4755`, you are not using the wrapper; re-check `head -n 5 ~/.local/bin/electerm`.
  - Disabling the sandbox is less secure; if you need a sandboxed install, use a system package (requires `sudo`) instead of AppImage.
  **References**:
  - https://github.com/electerm/electerm/releases
  - https://chromium.googlesource.com/chromium/src/+/main/docs/security/apparmor-userns-restrictions.md

