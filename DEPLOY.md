# Host FREE with auto-deploy like Vercel 🚀 (Streamlit Cloud)

**Why not Hugging Face?** HF now requires a paid PRO plan for any Space that runs
Python compute (Gradio/Docker/Streamlit) — only Static stays free. So we use:

**Streamlit Community Cloud** — free forever for public apps, ~1 GB RAM,
sleeps after 12 quiet hours (wakes when visited), and **auto-redeploys on every
GitHub push — exactly like Vercel.** You get a link like
`https://diabetes-xai-abc123.streamlit.app`.

Your repo here is committed and push-ready ✅ (already optimized with lazy SHAP/LIME loading for 1 GB RAM).

## Path 1 — I push to your GitHub (fastest, phone-only) ⚡

1. Log in at **https://github.com** (sign up if needed)
2. Create repo: **https://github.com/new** → name `diabetes-xai` → **Public** → Create
   (don't tick "Add a README")
3. Create token: profile pic → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**
   - Name: `arena-deploy`, select repo `diabetes-xai`, permission **Contents: Read and Write** → Generate → **Copy** (starts with `github_pat_...`)
4. **Paste here in chat:** your GitHub username + the token

I'll push the whole repo **including trained models** (instant boot, no training wait).
Then you do the fun 3-tap part below. Revoke the token after. 📱

## Path 2 — You upload via phone browser (no token)

1. Create the repo as in Path 1, steps 1–2
2. In the repo → **Add file → Upload files** — upload these 5 from this workspace:
   - `app.py`, `train.py`, `bootstrap.py`, `requirements.txt`, `data/diabetes.csv`
3. Teams will auto-train models on first boot (~2 min) — verified working ✅

## The 3-tap deploy (both paths) 🎯

1. Go to **https://share.streamlit.io** → **Sign in with GitHub** → authorize
2. **New app** → Repository: `you/diabetes-xai` → Branch: main → Main file: `app.py`
3. Tap **Deploy** → wait ~5 min → permanent link! 🎉

## Vercel-style auto-updates 🔄

- Every push / web-upload to the GitHub repo = **automatic rebuild + redeploy**
- Logs + reboot + delete: open your app → **Manage app** (bottom-right menu)
- Rollback: revert the commit on GitHub → auto-redeploys the old version

## Troubleshooting

| Problem | Fix |
|---|---|
| First visit slow / "Please wait" | Cold start + install; wait 1–2 min, refresh |
| App 'sleeping' after 12 h idle | Normal on free tier — visit wakes it in ~1 min |
| Build error on a package | Manage app → Reboot; versions in requirements.txt are verified |
| "Add a README" ticked at repo creation | Fine for Path 2; for Path 1 leave it unticked (avoids push conflict) |
