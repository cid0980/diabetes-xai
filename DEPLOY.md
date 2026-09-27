# Deploy FREE on your phone (no laptop) 📱 — ~5 minutes

**Use Hugging Face Spaces** — free forever, no credit card, gives you a permanent link like
`https://huggingface.co/spaces/YOURNAME/diabetes-xai`

The project is now **self-bootstrapping**: you upload only 4 small files, and the
app downloads the dataset + trains itself on first launch. Verified working ✅

## Step 1 — Get the 4 files onto your phone (1 min)

Open this workspace on your phone's browser and download these 4 files
(each is small —KBs, no copy-paste needed):

1. `diabetes-xai/app.py`
2. `diabetes-xai/train.py`
3. `diabetes-xai/bootstrap.py`
4. `diabetes-xai/requirements.txt`

> You do NOT need `data/` or `models/` (13 MB) — the cloud rebuilds them automatically.

## Step 2 — Create a free Hugging Face account (1 min)

1. Go to **https://huggingface.co** → Sign Up (Google one-tap works)
2. Verify email if asked

## Step 3 — Create the Space (1 min)

1. Go to **https://huggingface.co/new-space**
2. Fill in:
   - **Space name:** `diabetes-xai`
   - **SDK:** select **Streamlit** ⚠️ (important!)
   - Visibility: Public (free) — or Private, both free
3. Tap **Create Space**

## Step 4 — Upload the 4 files (1 min)

1. In your new Space → **Files** tab → **Add file → Upload files**
2. Select the 4 downloaded files → **Commit**
3. Wait ~4–6 min while it builds (you'll see "Building..." then your app 🎉)

First launch trains the models (~1–2 min) — just wait, then it works forever.

## Step 5 — Share it

Your permanent link: `https://huggingface.co/spaces/<you>/diabetes-xai`
Put this link in your project report under "Deployed Demo". It never sleeps.

## Troubleshooting

| Problem | Fix |
|---|---|
| Build error mentioning a package | Tap ⋮ → "Factory reboot", it retries clean |
| App shows error on first load | Wait 2 min and refresh — first boot trains models |
| Wrong SDK picked | Space Settings → change SDK to Streamlit |

## Alternative: Streamlit Cloud (also free)

Needs a GitHub account + repo upload — slightly more steps on mobile, so
Hugging Face above is recommended. If you prefer it: share.streamlit.io → deploy
from the same 4 files pushed to GitHub.
