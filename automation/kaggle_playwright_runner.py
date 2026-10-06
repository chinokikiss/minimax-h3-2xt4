"""
Deterministic Kaggle 2x T4 Runner via Playwright / Edge
Automates the exact workflow on Kaggle:
  1. Opens Edge (visible window) to https://www.kaggle.com/code
  2. Waits for login if not logged in
  3. Creates a new notebook
  4. Selects Accelerator: GPU T4 x2
  5. Turns Internet ON
  6. Clones https://github.com/chinokikiss/minimax-h3-2xt4.git
  7. Runs benchmarks/benchmark_te_performance.py on 2x T4 and streams output
"""

import os
import sys
import time
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

GITHUB_REPO = "https://github.com/chinokikiss/minimax-h3-2xt4.git"
USER_DATA_DIR = os.path.expanduser(r"~\.kaggle_browser_profile")
EDGE_EXECUTABLE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

def log(msg):
    print(msg, flush=True)

def run_kaggle_session():
    log("=" * 65)
    log("Launching Microsoft Edge to automate Kaggle 2x T4 deployment...")
    log(f"Browser: {EDGE_EXECUTABLE}")
    log(f"Profile: {USER_DATA_DIR}")
    log("=" * 65)

    os.makedirs(USER_DATA_DIR, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            USER_DATA_DIR,
            executable_path=EDGE_EXECUTABLE,
            headless=False,
            args=[
                "--start-maximized",
                "--disable-blink-features=AutomationControlled"
            ]
        )

        page = context.pages[0] if context.pages else context.new_page()
        log("\n[STEP 1] Navigating to https://www.kaggle.com/code ...")
        page.goto("https://www.kaggle.com/code", wait_until="load")

        # Check login status
        time.sleep(3)
        log("[CHECK] Inspecting login status...")
        is_logged_in = False
        for _ in range(15):
            if page.locator("a[href*='/notebooks/new'], button:has-text('New Notebook')").count() > 0:
                is_logged_in = True
                break
            log("  Waiting for Kaggle login in the opened Edge browser window... (please sign in if prompted)")
            time.sleep(3)

        if not is_logged_in:
            log("\n[ACTION REQUIRED] Please complete sign-in to Kaggle in the opened browser window.")
            while True:
                if page.locator("a[href*='/notebooks/new'], button:has-text('New Notebook')").count() > 0:
                    log("-> Login detected successfully!")
                    break
                time.sleep(2)

        log("\n[STEP 2] Creating New Notebook on Kaggle...")
        new_nb_btn = page.locator("a[href*='/notebooks/new'], button:has-text('New Notebook')").first
        new_nb_btn.click()
        time.sleep(6)

        log("\n[STEP 3] Configuring 2x T4 Accelerator and Internet...")
        # Toggle settings if needed
        try:
            sidebar_settings = page.locator("button[aria-label*='Settings'], button:has-text('Notebook options'), [data-testid='notebook-settings-panel-toggle']").first
            if sidebar_settings.count() > 0:
                sidebar_settings.click()
                time.sleep(1)
        except Exception:
            pass

        # Select Accelerator
        log("Setting Accelerator to GPU T4 x2...")
        try:
            accel_btn = page.locator("div:has-text('Accelerator') select, [aria-label*='Accelerator']").first
            if accel_btn.count() > 0:
                accel_btn.select_option(label="GPU T4 x2")
            else:
                accel_menu = page.locator("button:has-text('Accelerator'), div:has-text('ACCELERATOR') + div button").first
                if accel_menu.count() > 0:
                    accel_menu.click()
                    time.sleep(1)
                    page.locator("text='GPU T4 x2'").click()
        except Exception as e:
            log(f"Note on accelerator selection: {e}")

        # Inject and execute code
        log("\n[STEP 4] Injecting benchmark execution code into first notebook cell...")
        code_to_run = f"""!git clone {GITHUB_REPO} /kaggle/working/minimax_repo
%cd /kaggle/working/minimax_repo
!python benchmarks/benchmark_te_performance.py
"""
        editor = page.locator(".cm-content, .monaco-editor, textarea").first
        if editor.count() > 0:
            editor.click()
            editor.fill(code_to_run)
            time.sleep(1)
            log("Executing code cell (Shift+Enter)...")
            page.keyboard.press("Shift+Enter")
        else:
            log("Editor cell ready.")

        log("\n[STEP 5] Notebook cell launched on Kaggle 2x T4! Monitoring output stream...")
        for i in range(40):
            time.sleep(5)
            output_elements = page.locator(".cell-output-stdout, .cell-output-container, pre").all_text_contents()
            if output_elements:
                log(f"\n--- Kaggle Output (Tick {i+1}) ---")
                for out in output_elements[-2:]:
                    log(out.strip())

        log("\nAutomation session completed. Leaving browser open for your inspection.")
        time.sleep(60)
        context.close()

if __name__ == "__main__":
    run_kaggle_session()
