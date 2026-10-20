"""CLI compatibility entry point for whole-layer offline LoRA baking."""
import retest  # Establish the selected COMFY_ROOT before importing Native ops.
from patches.offline_lora import merged_weight,convert,main


if __name__=='__main__':main()
