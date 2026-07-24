# Publishing checklist

Everything is automated by `.github/workflows/release.yml` — pushing a tag
`vX.Y.Z` runs tests, creates a GitHub Release with `bihgrid-dataset.zip`
attached, and publishes the package to PyPI. Three one-time setups (steps 2–4)
need your accounts; after those, releasing is step 5 only.

## 1. Push the code (every release)

```bash
git push origin main
```

## 2. One-time: PyPI trusted publisher

No API tokens needed — PyPI trusts the GitHub workflow directly.

1. Log in at https://pypi.org (create an account if needed).
2. Go to **Your projects → Publishing** (https://pypi.org/manage/account/publishing/).
3. Under "Add a new pending publisher", fill in:
   - PyPI project name: `bihgrid`
   - Owner: `FarukDziho`
   - Repository name: `bih-power-data`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
4. On GitHub: repo **Settings → Environments → New environment** → name it
   `pypi` (no other configuration needed).

## 3. One-time: Zenodo ↔ GitHub link (DOI)

1. Log in at https://zenodo.org with your GitHub account.
2. Go to https://zenodo.org/account/settings/github/ and flip
   **FarukDziho/bih-power-data** to ON.
3. That's it — every GitHub release from now on is archived automatically
   with a versioned DOI (plus one concept DOI covering all versions).
   Metadata comes from `.zenodo.json` (already in the repo).
4. After the first release, copy the DOI badge from the Zenodo record into
   `README.md` and `DATASET_CARD.md` (there's a placeholder in the citation
   section).

## 4. One-time: sanity check

- GitHub repo **Settings → Actions → General** → allow GitHub Actions to
  create releases (default "Read and write permissions" is fine).

## 5. Release (every version)

```bash
git tag v0.1.0
git push origin v0.1.0
```

Then watch the **Actions** tab: `release` should go green, the release
appears under **Releases** with `bihgrid-dataset.zip`, `pip install bihgrid`
works a few minutes later, and Zenodo mints the DOI.

## Version bumping

Bump `version` in `pyproject.toml` AND `__version__` in
`bihgrid/__init__.py`, commit, then tag. PyPI rejects re-uploads of an
existing version — always bump before tagging.

## What's deliberately NOT automated

- The data paper (Scientific Data / NeurIPS D&B) — next roadmap item.
- PyG upstream PR — per the plan, only after the dataset has citations.
