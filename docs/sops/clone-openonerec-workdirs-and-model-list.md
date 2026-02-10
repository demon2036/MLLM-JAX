# SOP: Clone OpenOneRec into repo-local `workdirs/` and verify public models

- **Title**: SOP: Clone `Kuaishou-OneRec/OpenOneRec` into this repo's `workdirs/` and validate published model IDs
- **Prereqs**: Ubuntu Linux; `git`; Python 3; outbound network access to GitHub and Hugging Face
- **Environment (verified)**:
  - Repo: `/home/john/workdir/openonerec`
  - Date: 2026-02-10
  - Shell: `bash`

## Steps (commands actually used)

1. Create local clone directory and clone OpenOneRec:

```bash
mkdir -p workdirs
GIT_TERMINAL_PROMPT=0 git clone --depth 1 https://github.com/Kuaishou-OneRec/OpenOneRec.git workdirs/OpenOneRec
```

2. Verify clone revision and remote:

```bash
git -C workdirs/OpenOneRec rev-parse --short HEAD
git -C workdirs/OpenOneRec status -sb
git -C workdirs/OpenOneRec remote -v
```

3. Inspect model declarations in docs:

```bash
sed -n '1,260p' workdirs/OpenOneRec/README.md
sed -n '1,220p' workdirs/OpenOneRec/tokenizer/README.md
rg -n "huggingface\.co/OpenOneRec|OpenOneRec/OneRec-|OpenOneRec/" workdirs/OpenOneRec --glob '*.md'
```

4. Cross-check against Hugging Face API (author = `OpenOneRec`):

```bash
python - <<'PY'
import json, urllib.request
url='https://huggingface.co/api/models?author=OpenOneRec&limit=200'
with urllib.request.urlopen(url, timeout=20) as r:
    data=json.load(r)
print('count', len(data))
for item in data:
    print(item.get('id'))
PY
```

5. Validate doc IDs vs API IDs:

```bash
python - <<'PY'
import re, json, urllib.request, pathlib
readme = pathlib.Path('workdirs/OpenOneRec/README.md').read_text(encoding='utf-8')
tokenizer_readme = pathlib.Path('workdirs/OpenOneRec/tokenizer/README.md').read_text(encoding='utf-8')
ids = set(re.findall(r'OpenOneRec/[A-Za-z0-9_.\-]+', readme + '\n' + tokenizer_readme))
ids = {x for x in ids if x.startswith('OpenOneRec/OneRec-')}
with urllib.request.urlopen('https://huggingface.co/api/models?author=OpenOneRec&limit=200', timeout=20) as r:
    api_ids = {item['id'] for item in json.load(r)}
print('missing_in_docs:', sorted(api_ids - ids))
print('extra_in_docs:', sorted(ids - api_ids))
PY
```

## Expected Result

- `workdirs/OpenOneRec` exists with a valid git clone.
- Model IDs found in docs and API are consistent.
- As of 2026-02-10, verified public models are:
  - `OpenOneRec/OneRec-1.7B`
  - `OpenOneRec/OneRec-8B`
  - `OpenOneRec/OneRec-1.7B-pro`
  - `OpenOneRec/OneRec-8B-pro`
  - `OpenOneRec/OneRec-tokenizer`

## Troubleshooting

- If clone fails due to auth prompt/hang, keep `GIT_TERMINAL_PROMPT=0`.
- If `workdirs/OpenOneRec` already exists, either delete it or run `git -C workdirs/OpenOneRec fetch origin --depth 1` then checkout desired ref.
- If Hugging Face API is unavailable, retry later and treat README list as temporary source-of-truth.

## References

- `workdirs/OpenOneRec/README.md`
- `workdirs/OpenOneRec/tokenizer/README.md`
- `memory/20260210_clone-openonerec-models/README.md`
