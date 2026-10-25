"""
Kaggle CLI Workflow Runner: Push & Output
Executes the standard, headless Kaggle CLI workflow:
  1. Verifies Kaggle credentials (~/.kaggle/kaggle.json or access_token)
  2. Pushes kernel with 2x T4 accelerator:
     kaggle kernels push -p kaggle_kernel --accelerator NvidiaTeslaT4
  3. Monitors execution status until complete:
     kaggle kernels status <kernel_id>
  4. Downloads outputs and benchmark charts:
     kaggle kernels output <kernel_id> -p kaggle_output
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

KAGGLE_EXE = r"C:\ProgramData\miniconda3\envs\comfyui\Scripts\kaggle.exe"
KERNEL_DIR = Path(__file__).parent / "kaggle_kernel"
OUTPUT_DIR = Path(__file__).parent / "kaggle_output"
METADATA_FILE = KERNEL_DIR / "kernel-metadata.json"

# Ensure proxy is set for Kaggle API access
if "HTTP_PROXY" not in os.environ:
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
if "HTTPS_PROXY" not in os.environ:
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

def log(msg):
    print(msg, flush=True)

def check_credentials():
    kaggle_dir = Path.home() / ".kaggle"
    token_file = kaggle_dir / "access_token"
    json_file = kaggle_dir / "kaggle.json"
    
    has_token = token_file.exists() or json_file.exists()
    has_env = "KAGGLE_API_TOKEN" in os.environ or ("KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ)
    
    if not (has_token or has_env):
        log("\n[AUTHENTICATION REQUIRED]")
        log("Kaggle credentials not detected in ~/.kaggle/kaggle.json or ~/.kaggle/access_token.")
        log("To authenticate, you can:")
        log("  1. Run in terminal: kaggle auth login")
        log("  2. Or download kaggle.json from https://www.kaggle.com/settings/api and place it at:")
        log(f"     {json_file}")
        return False
    return True

def get_kernel_id():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("id")

def set_kernel_username(username):
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    slug = meta.get("id", "").split("/")[-1] or "minimax-h3-2xt4-benchmark"
    meta["id"] = f"{username}/{slug}"
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"Updated kernel ID to: {meta['id']}")

def push_kernel():
    log("\n" + "=" * 65)
    log("STEP 1: Pushing kernel to Kaggle with 2x T4 Accelerator...")
    log(f"Command: kaggle kernels push -p {KERNEL_DIR} --accelerator NvidiaTeslaT4")
    log("=" * 65)
    
    cmd = [KAGGLE_EXE, "kernels", "push", "-p", str(KERNEL_DIR), "--accelerator", "NvidiaTeslaT4"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    log(result.stdout.strip())
    if result.stderr:
        log("STDERR: " + result.stderr.strip())
    
    if result.returncode != 0:
        log("Kernel push failed. Check output above.")
        return False
    return True

def monitor_kernel(kernel_id, poll_interval=15):
    log("\n" + "=" * 65)
    log(f"STEP 2: Monitoring Kernel Execution on Kaggle: {kernel_id}")
    log("=" * 65)

    cmd = [KAGGLE_EXE, "kernels", "status", kernel_id]
    
    while True:
        res = subprocess.run(cmd, capture_output=True, text=True)
        status_line = res.stdout.strip()
        log(f"[{time.strftime('%X')}] {status_line}")
        
        lower_status = status_line.lower()
        if "complete" in lower_status:
            log("-> Kernel run completed successfully!")
            break
        elif "error" in lower_status:
            log("-> Kernel finished with an ERROR.")
            break
        elif "cancel" in lower_status:
            log("-> Kernel was cancelled.")
            break
            
        time.sleep(poll_interval)

def fetch_outputs(kernel_id):
    log("\n" + "=" * 65)
    log(f"STEP 3: Downloading Kernel Outputs to: {OUTPUT_DIR}")
    log("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [KAGGLE_EXE, "kernels", "output", kernel_id, "-p", str(OUTPUT_DIR)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    log(res.stdout.strip())
    if res.stderr:
        log("STDERR: " + res.stderr.strip())

    # List output files
    output_files = list(OUTPUT_DIR.glob("*"))
    log(f"\nDownloaded {len(output_files)} files:")
    for f in output_files:
        log(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

def main():
    if not check_credentials():
        sys.exit(1)

    kernel_id = get_kernel_id()
    if push_kernel():
        monitor_kernel(kernel_id)
        fetch_outputs(kernel_id)

if __name__ == "__main__":
    main()
