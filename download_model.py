from sentence_transformers import SentenceTransformer
import os

# Set cache folder (optional, but good for clarity)
# os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/app/model_cache'

print("Pre-downloading model: all-MiniLM-L6-v2")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model downloaded successfully.")
