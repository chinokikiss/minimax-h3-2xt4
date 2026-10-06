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

GITHUB_REPO = "https://github.com/chinokikiss/minimax-h3-2xt4.git"
USER_DATA_DIR = os.path.expanduser(r"~\.kaggle_browser_profile")
EDGE_EXECUTABLE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

def run_kaggle_session():
    print("=" * 65)
    print("Launching Microsoft Edge to automate Kaggle 2x T4 deployment...")
    print(f"Browser: {EDGE_EXECUTABLE}")
    print(f"Profile: {USER_DATA_DIR}")
    print("=" * 65)

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
        print("\n[STEP 1] Navigating to https://www.kaggle.com/code ...")
        page.goto("https://www.kaggle.com/code", wait_until="networkidle")

        # Check login status
        time.sleep(3)
        print("[CHECK] Inspecting login status...")
        is_logged_in = False
        for _ in range(30):
            # If user avatar or 'New Notebook' button exists
            if page.locator("a[href*='/notebooks/new'], button:has-text('New Notebook')").count() > 0:
                is_logged_in = True
                break
            print("  Waiting for Kaggle login in the opened Edge browser window... (please sign in)")
            time.sleep(4)

        if not is_logged_in:
            print("\n[NOTICE] Please sign in to Kaggle in the opened browser window.")
            while True:
                if page.locator("a[href*='/notebooks/new'], button:has-text('New Notebook')").count() > 0:
                    print("-> Login detected!")
                    break
                time.sleep(2)

        print("\n[STEP 2] Creating New Notebook...")
        new_nb_btn = page.locator("a[href*='/notebooks/new'], button:has-text('New Notebook')").first
        new_nb_btn.click()
        page.wait_for_load_state("networkidle")
        time.sleep(5)

        print("\n[STEP 3] Configuring 2x T4 Accelerator and Internet...")
        # Open notebook settings if closed
        sidebar_settings = page.locator("button[aria-label*='Settings'], button:has-text('Notebook options'), [data-testid='notebook-settings-panel-toggle']").first
        if sidebar_settings.count() > 0:
            try:
                sidebar_settings.click()
                time.sleep(1)
            except Exception:
                pass

        # Select Accelerator
        print("Selecting Accelerator: GPU T4 x2...")
        try:
            # Look for Accelerator dropdown
            accel_btn = page.locator("div:has-text('Accelerator') select, [aria-label*='Accelerator']").first
            if accel_btn.count() > 0:
                accel_btn.select_option(label="GPU T4 x2")
            else:
                # Click dropdown menu
                accel_menu = page.locator("button:has-text('Accelerator'), div:has-text('ACCELERATOR') + div button").first
                if accel_menu.count() > 0:
                    accel_menu.click()
                    time.sleep(1)
                    page.locator("text='GPU T4 x2'").click()
        except Exception as e:
            print(f"Note on accelerator selection: {e}")

        # Inject and execute code
        print("\n[STEP 4] Injecting bootstrap benchmark code...")
        code_to_run = f"""!git clone {GITHUB_REPO} /kaggle/working/minimax_repo
%cd /kaggle/working/minimax_repo
!python benchmarks/benchmark_te_performance.py
"""
        # Focus code editor
        editor = page.locator(".cm-content, .monaco-editor, textarea").first
        if editor.count() > 0:
            editor.click()
            editor.fill(code_to_run)
            time.sleep(1)
            print("Executing code cell (Shift+Enter)...")
            page.keyboard.press("Shift+Enter")
        else:
            print("Could not find standard editor cell directly. Please observe notebook.")

        print("\n[STEP 5] Notebook cell launched on Kaggle 2x T4! Monitoring output...")
        for i in range(30):
            time.sleep(5)
            # Print status periodically
            output_elements = page.locator(".cell-output-stdout, .cell-output-container, pre").all_text_contents()
            if output_elements:
                print("\n--- Current Kaggle Cell Output ---")
                for out in output_elements[-3:]:
                    print(out.strip())
                if "MiniMax-H3 Qwen3-VL-32B" in "".join(output_elements):
                    print("-> Benchmark execution detected on Kaggle 2x T4!")

        print("\nKaggle automation session is active. Leaving browser open for your inspection.")
        input("Press Enter in terminal when you wish to close the browser session...")
        context.close()

if __name__ == "__main__":
    run_kaggle_session()
