# Host FREE with auto-deploy like Vercel 🚀

**Yes — Hugging Face Spaces works exactly like Vercel:**
every Space is a git repo, and **every push / file edit auto-rebuilds and redeploys**
(~3–6 min build, logs visible on the Space page, rollback = revert).

Your project is already a clean git repo (committed ✅). Pick a path:

## Path 1 — I push it for you (fastest, phone-only) ⚡

1. Go to **https://huggingface.co** → Sign up / log in
2. Create a token: **Settings → Access Tokens → Create** (fine-grained, ✅ write access to one Space — revoke it after, safe)
3. Create the Space: **https://huggingface.co/new-space**
   - Name: `diabetes-xai`, SDK: **Streamlit**, Public
4. **Paste here in chat:** your HF username + Space name + the token

I'll push the whole repo from here → your Space builds → you get a permanent link.
Delete the token after. Done, you never touch a file. 📱

## Path 2 — You upload via phone browser (5 min, no git)

1. Log in at **https://huggingface.co** → **New Space** → name `diabetes-xai`, SDK **Streamlit**
2. **Files → Add file → Upload** these 4 small files from this workspace:
   - `app.py`, `train.py`, `bootstrap.py`, `requirements.txt`
3. Wait ~5 min → live forever at `huggingface.co/spaces/<you>/diabetes-xai`

> Skip `data/` + `models/` (13 MB) — the app downloads data + trains itself on first boot. Verified ✅
> Every future upload/edit = auto-rebuild, Vercel-style.

## Path 3 — Git push from laptop later 💻

```bash
git clone https://huggingface.co/spaces/<you>/diabetes-xai
# copy all project files in, then:
git add -A && git commit -m "deploy" && git push
# → Space auto-rebuilds. Every future push = auto-redeploy, like Vercel.
```

## Optional: GitHub as source of truth → auto-sync to HF

Keep code on GitHub, add this Action so **every GitHub push auto-updates the Space**:

`.github/workflows/sync-to-hf.yml`:
```yaml
name: Sync to HF Space
on:
  push:
    branches: [main]
jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - run: git push https://<you>:${{ secrets.HF_TOKEN }}@huggingface.co/spaces/<you>/diabetes-xai main:main
```
(Add `HF_TOKEN` under GitHub repo → Settings → Secrets.)

## Troubleshooting

| Problem | Fix |
|---|---|
| Build fails on a package | Space ⋮ menu → "Factory reboot" |
| First load shows error | Wait 2 min + refresh (first boot trains models) |
| Picked wrong SDK | Space → Settings → change SDK to Streamlit |
