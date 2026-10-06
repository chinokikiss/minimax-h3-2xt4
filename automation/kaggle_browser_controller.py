"""
Kaggle Automation Controller using browser-use
Automates:
  1. Opening Kaggle (https://www.kaggle.com/code) in Edge browser (visible window)
  2. Logging in or using persistent session from ~/.kaggle_browser_profile
  3. Creating a new notebook with Accelerator: 2x NVIDIA T4 GPUs
  4. Injecting setup commands (cloning chinokikiss/minimax-h3-2xt4)
  5. Running Text Encoder TP=2 vs PP=2 benchmarks on Kaggle 2x T4
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows console to prevent emoji logging crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from browser_use import Agent
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession
from browser_use.llm import ChatOpenRouter

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
EDGE_EXECUTABLE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
USER_DATA_DIR = os.path.expanduser(r"~\.kaggle_browser_profile")
GITHUB_REPO = "https://github.com/chinokikiss/minimax-h3-2xt4.git"

async def run_kaggle_automation():
    print("=" * 60)
    print("Starting Kaggle 2x T4 Automation with browser-use")
    print(f"Browser: Edge ({EDGE_EXECUTABLE})")
    print(f"Profile Directory: {USER_DATA_DIR}")
    print(f"Target GitHub Repo: {GITHUB_REPO}")
    print("=" * 60)

    # 1. Initialize LLM via OpenRouter (using gpt-4o-mini to avoid credit exhaustion)
    llm = ChatOpenRouter(
        api_key=OPENROUTER_API_KEY,
        model="openai/gpt-4o-mini"
    )

    # 2. Configure Browser Profile (Visible window so user can interact / observe)
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    profile = BrowserProfile(
        executable_path=EDGE_EXECUTABLE,
        user_data_dir=USER_DATA_DIR,
        headless=False,
        enable_default_extensions=False,
    )
    browser_session = BrowserSession(browser_profile=profile)

    task_prompt = f"""
1. Navigate to https://www.kaggle.com/code.
2. Check if the user is already logged in to Kaggle.
   - If not logged in, stop and ask the user to log in to Kaggle in the opened browser window.
   - If logged in, click "New Notebook" to create a new Python notebook.
3. In the notebook settings panel on the right:
   - Find 'Accelerator' and change it from 'None' to 'GPU T4 x2' (2x T4 GPUs).
   - Ensure 'Internet' is toggled ON.
4. In the first code cell of the notebook, insert and execute the following code to clone our repo and start the 2x T4 Text Encoder benchmark:

```python
!git clone {GITHUB_REPO} /kaggle/working/minimax_repo
%cd /kaggle/working/minimax_repo
!python benchmarks/benchmark_te_performance.py
```

5. Confirm that the command is running and report the output.
"""

    agent = Agent(
        task=task_prompt,
        llm=llm,
        browser_session=browser_session,
        use_vision=True
    )

    print("\n[AGENT LAUNCH] Executing browser-use agent with GPT-4o...")
    history = await agent.run(max_steps=25)
    print("\n[AGENT COMPLETE] Finished steps:", len(history.history))
    return history

if __name__ == "__main__":
    asyncio.run(run_kaggle_automation())
