# SOP: Install `gcloud` on Linux via tarball (no apt)

- **Title**: SOP: Install Google Cloud CLI (`gcloud`) on a Linux workstation from the official tarball (and enable `alpha` TPU VM commands)
- **Prereqs**: `curl`, `tar`, sudo/root write access to `/usr/local`; outbound network access to `dl.google.com`
- **Environment (verified)**: Ubuntu `25.10` (x86_64)

## Steps (commands actually run)

- Install Google Cloud CLI under `/usr/local/google-cloud-sdk` and put `gcloud` on `PATH`:
  - ```bash
    set -euo pipefail
    GCLOUD_DIR="/usr/local/google-cloud-sdk"
    TARBALL_URL="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-470.0.0-linux-x86_64.tar.gz"
    TMPDIR="$(mktemp -d -p /var/tmp gcloudsdk.XXXXXX)"
    trap 'rm -rf "$TMPDIR"' EXIT
    cd "$TMPDIR"
    curl -fsSLo gcloud.tgz "$TARBALL_URL"
    tar -xzf gcloud.tgz -C /usr/local
    ln -sf "$GCLOUD_DIR/bin/gcloud" /usr/local/bin/gcloud
    ```

- Disable prompts/usage reporting (good default for non-interactive scripts):
  - ```bash
    gcloud config set core/disable_usage_reporting true
    gcloud config set core/disable_prompts true
    ```

- Install `alpha` component (needed for `gcloud alpha compute tpus tpu-vm ...`):
  - ```bash
    gcloud components install alpha --quiet
    ```

- Verify:
  - ```bash
    gcloud --version
    gcloud alpha compute tpus tpu-vm --help >/dev/null
    ```

## Expected Result

- `gcloud --version` prints `Google Cloud SDK 470.0.0` (or similar) and `alpha ...`.
- `gcloud alpha compute tpus tpu-vm --help` exits `0`.

## Troubleshooting

- `tar: ... No space left on device`:
  - Don’t extract in `/tmp` if it’s a small `tmpfs`. Use `/var/tmp` (on-disk) or extract directly to `/usr/local` as above.

## References

- Google Cloud CLI install docs: https://cloud.google.com/sdk/docs/install

