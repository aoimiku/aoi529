import sys
import os
import io

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.getcwd())

from src.bot_brain import BotBrain

print("Initialize BotBrain for indexing...", flush=True)

# Initialize loads model, but index might be empty
brain = BotBrain()

print("Starting index generation (10,000 items)...", flush=True)
try:
    # Explicitly call build with sample size
    brain._build_index_sample(sample_size=30000)
    print("Done! Index generated successfully.", flush=True)
except Exception as e:
    print(f"Error during indexing: {e}", flush=True)
    import traceback
    traceback.print_exc()
