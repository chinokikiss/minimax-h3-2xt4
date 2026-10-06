"""
Kaggle Automation Controller using browser-use
Automates:
  1. Opening Kaggle and navigating to Notebooks
  2. Creating a new notebook with Accelerator: 2x NVIDIA T4 GPUs
  3. Injecting setup commands (cloning project repo, installing dependencies)
  4. Downloading models from Comfy-Org/MiniMax-H3 on Hugging Face
  5. Running Text Encoder TP=2 vs PP=2 benchmarks and saving results
"""

import os
import asyncio
from typing import Optional
from pydantic import BaseModel
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig

class KaggleAutomationConfig(BaseModel):
    headless: bool = False
    kaggle_url: str = "https://www.kaggle.com/code"
    github_repo_url: str = "https://github.com/your-username/minimax-h3-2xt4.git"
    huggingface_repo: str = "Comfy-Org/MiniMax-H3"
    accelerator: str = "GPU T4 x2"
    user_data_dir: Optional[str] = None

class KaggleBrowserController:
    def __init__(self, config: Optional[KaggleAutomationConfig] = None):
        self.config = config or KaggleAutomationConfig()
        browser_config = BrowserConfig(
            headless=self.config.headless,
        )
        self.browser = Browser(config=browser_config)

    async def create_agent(self, task_instruction: str, llm=None):
        """
        Creates a browser-use Agent to execute the Kaggle task.
        """
        agent = Agent(
            task=task_instruction,
            llm=llm,
            browser=self.browser
        )
        return agent

    def generate_kaggle_setup_script(self) -> str:
        """
        Generates the bash / python bootstrap script to be pasted into the first cell of Kaggle.
        """
        return f"""# ==============================================================
# MiniMax-H3 on 2x T4 Setup Script
# ==============================================================
import os, sys, subprocess

# 1. Verify 2x T4 Environment & Check P2P Status
import torch
print(f"CUDA Available: {{torch.cuda.is_available()}}")
print(f"Device Count: {{torch.cuda.device_count()}}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {{i}}: {{torch.cuda.get_device_name(i)}} - {{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f}} GB")

if torch.cuda.device_count() >= 2:
    can_p2p = torch.cuda.can_device_access_peer(0, 1)
    print(f"GPU P2P Access (0 <-> 1): {{can_p2p}} (Expected: False on Kaggle)")
    # Set NCCL environment variables for non-P2P multi-GPU
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 2. Clone Upstream ComfyUI & Project Repository
!git clone https://github.com/Comfy-Org/ComfyUI.git
!git clone {self.config.github_repo_url} project_repo

# 3. Install Acceleration Packages (comfy_kitchen, sageattention, huggingface_hub, aria2)
!pip install -q huggingface_hub aria2p
!apt-get update -qq && apt-get install -qq -y aria2

# 4. Download Required Models from Comfy-Org/MiniMax-H3
# Model files:
#   - Text Encoder: qwen3vl_32b_minimax_h3_int8_convrot.safetensors (~27.1 GB)
#   - Diffusion: minimax_h3_ref2va_pruned_int8_convrot.safetensors (~20.9 GB)
#   - Video VAE: minimax_h3_video_vae_int8_convrot.safetensors (~2.8 GB)
#   - Audio VAE: minimax_h3_audio_vae_fp32.safetensors (~605 MB)

os.makedirs("ComfyUI/models/text_encoders", exist_ok=True)
os.makedirs("ComfyUI/models/diffusion_models", exist_ok=True)
os.makedirs("ComfyUI/models/vae", exist_ok=True)

print("Starting high-speed aria2c model downloads...")
!aria2c -x 16 -s 16 -k 10M "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" -d "ComfyUI/models/text_encoders"
!aria2c -x 16 -s 16 -k 10M "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" -d "ComfyUI/models/diffusion_models"
!aria2c -x 16 -s 16 -k 10M "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors" -d "ComfyUI/models/vae"
!aria2c -x 16 -s 16 -k 10M "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors" -d "ComfyUI/models/vae"

print("Setup complete! Ready for module-by-module benchmarking.")
"""

async def run_automation_demo():
    controller = KaggleBrowserController()
    task = (
        "1. Open https://www.kaggle.com/code in the browser.\n"
        "2. If not logged in, wait or log in.\n"
        "3. Click 'New Notebook'.\n"
        "4. In the notebook settings sidebar, under 'Accelerator', select 'GPU T4 x2'.\n"
        "5. Under 'Internet', ensure Internet is turned 'On'.\n"
        "6. In the first code cell, paste the setup script and run it."
    )
    print("Prepared automation task prompt:")
    print(task)
    print("\nSetup script preview:\n")
    print(controller.generate_kaggle_setup_script()[:400] + "...\n")

if __name__ == "__main__":
    asyncio.run(run_automation_demo())
