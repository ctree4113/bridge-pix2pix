from datasets import load_dataset
import os

os.makedirs("magicbrush", exist_ok=True)

dataset = load_dataset("osunlp/MagicBrush")

dataset.save_to_disk("magicbrush")