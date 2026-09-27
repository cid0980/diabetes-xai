"""Live-site QA bot v2: drives the deployed app INSIDE its hosting iframe.
Run: python3 qa_live.py
"""
import json
import re
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "https://diabetes-xai.streamlit.app/"
OUT = Path("/home/user/diabetes-xai/qa")
OUT.mkdir(exist_ok=True)
results = {"url": URL, "checks": [], "console_errors": [], "failed": []}


def check(name, ok, detail=""):
    results["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:300]})
    print(("PASS " if ok else "FAIL ") + name + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        results["failed"].append(name)


def app_frame(page, timeout=300000):
    """Return a locator for the inner Streamlit app iframe (outer shell has empty body)."""
    page.locator('iframe[src*="/~/+"]').wait_for(timeout=timeout)
    fl = page.frame_locator('iframe[src*="/~/+"]')
    fl.get_by_text("Explainable Diabetes Risk Predictor").wait_for(timeout=timeout)
    return fl


def inner_html(page):
    fr = page.frame(url=re.compile(r"/~/\+"))
    return fr.content() if fr else ""


def errors_visible(frame):
    return frame.get_by_text("This app has encountered an error").count()


with sync_playwright() as pw:
    browser = pw.chromium.launch(args=["--no-sandbox"])
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    page = ctx.new_page()
    page.on("console", lambda m: results["console_errors"].append(m.text[:200]) if m.type == "error" else None)
    page.on("pageerror", lambda e: results["console_errors"].append(str(e)[:200]))

    # 1. LOAD
    t0 = time.time()
    page.goto(URL, wait_until="domcontentloaded", timeout=120000)
    try:
        frame = app_frame(page)
        check("app loads inside iframe", True, f"{time.time()-t0:.0f}s")
    except Exception as e:
        check("app loads inside iframe", False, str(e)[:150])
        page.screenshot(path=str(OUT / "0-load-fail.png"), full_page=True)
        raise SystemExit(1)
    check("no error box on load", errors_visible(frame) == 0)
    page.wait_for_timeout(3000)
    page.screenshot(path=str(OUT / "1-predict-initial.png"), full_page=True)

    # 2. PRESET
    frame.get_by_text("High-risk example").click()
    page.wait_for_timeout(4000)
    try:
        gval = frame.locator('input[aria-label="Glucose"]').input_value(timeout=15000)
    except Exception:
        gval = frame.locator('input[type="number"]').nth(1).input_value(timeout=15000)
    check("preset fills Glucose=185", gval == "185", f"Glucose={gval}")

    # 3. PREDICT
    t1 = time.time()
    frame.locator('[data-testid="stButton"] button', has_text="Predict Risk").click()
    try:
        frame.get_by_text("Diabetes probability").wait_for(timeout=240000)
        check("prediction verdict renders", True, f"{time.time()-t1:.0f}s")
    except Exception as e:
        check("prediction verdict renders", False, str(e)[:150])
    try:
        frame.get_by_text("per-patient impact").wait_for(timeout=240000)
        check("SHAP chart renders", True)
    except Exception as e:
        check("SHAP chart renders", False, str(e)[:150])
    try:
        frame.get_by_text("local rules for this patient").wait_for(timeout=240000)
        check("LIME chart renders", True)
    except Exception as e:
        check("LIME chart renders", False, str(e)[:150])
    try:
        frame.get_by_text("Download report").wait_for(timeout=60000)
        check("report download button renders", True)
    except Exception as e:
        check("report download button renders", False, str(e)[:150])
    check("no error box after predict", errors_visible(frame) == 0)
    page.wait_for_timeout(2000)
    page.screenshot(path=str(OUT / "2-predict-result.png"), full_page=True)

    # 4. COMPARISON
    frame.get_by_text("Model Comparison").first.click()
    try:
        frame.get_by_text("Paper vs Our System").wait_for(timeout=120000)
        check("Comparison page loads", True)
    except Exception as e:
        check("Comparison page loads", False, str(e)[:150])
    page.wait_for_timeout(5000)
    check("no error on Comparison (KeyError gone)", errors_visible(frame) == 0)
    page.screenshot(path=str(OUT / "3-comparison.png"), full_page=True)

    # 5. GLOBAL
    frame.get_by_text("Global Explanations").first.click()
    try:
        frame.get_by_text("What drives diabetes risk overall?").wait_for(timeout=120000)
        check("Global page loads", True)
    except Exception as e:
        check("Global page loads", False, str(e)[:150])
    page.wait_for_timeout(4000)
    page.screenshot(path=str(OUT / "4-global.png"), full_page=True)
    imgs_before = frame.locator('[data-testid="stImage"] img').count()
    frame.get_by_text("Compute SHAP summary").click()
    grew = False
    for _ in range(48):
        page.wait_for_timeout(5000)
        if frame.locator('[data-testid="stImage"] img').count() > imgs_before:
            grew = True
            break
        if errors_visible(frame) > 0:
            break
    check("SHAP summary plot renders", grew)
    check("no error on Global", errors_visible(frame) == 0)
    page.screenshot(path=str(OUT / "5-global-shap.png"), full_page=True)

    # 6. ABOUT
    frame.get_by_text("About Paper").first.click()
    try:
        frame.get_by_text("About this project").wait_for(timeout=120000)
        check("About page loads", True)
    except Exception as e:
        check("About page loads", False, str(e)[:150])
    page.wait_for_timeout(3000)
    check("no error on About", errors_visible(frame) == 0)
    page.screenshot(path=str(OUT / "6-about.png"), full_page=True)

    # 7. TOOLBAR state (inner app toolbar should be minimal now)
    inner = inner_html(page)
    check("inner toolbar has no GitHub link", "github.com/cid0980" not in inner.lower())

    # 8. EMBED MODE (cleanest link for report/viva?)
    ep = ctx.new_page()
    ep.goto(URL + "?embed=true", wait_until="domcontentloaded", timeout=120000)
    try:
        eframe = app_frame(ep)
        check("embed-mode loads", True)
        ep.wait_for_timeout(4000)
        outer = ep.content()
        check("embed-mode hides outer GitHub/Fork",
              "github.com/cid0980" not in outer.lower() and "fork" not in outer.lower())
        ep.screenshot(path=str(OUT / "7-embed-mode.png"), full_page=True)
    except Exception as e:
        check("embed-mode loads", False, str(e)[:150])

    # 9. MOBILE viewport
    mctx = browser.new_context(viewport={"width": 390, "height": 844},
                               is_mobile=True, has_touch=True, device_scale_factor=2)
    mp = mctx.new_page()
    mp.goto(URL, wait_until="domcontentloaded", timeout=120000)
    try:
        app_frame(mp)
        check("mobile viewport loads", True)
    except Exception as e:
        check("mobile viewport loads", False, str(e)[:150])
    mp.wait_for_timeout(3000)
    mp.screenshot(path=str(OUT / "8-mobile-predict.png"), full_page=True)

    browser.close()

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n==== SUMMARY ====")
print(f"passed: {sum(1 for c in results['checks'] if c['ok'])}/{len(results['checks'])}")
print("failed:", results["failed"] or "NONE")
print("console errors:", len(results["console_errors"]))
for e in results["console_errors"][:5]:
    print("  -", e)
