import kagglehub

# Download latest version
path = kagglehub.model_download("google/gemma/flax/1.1-2b-it",)

print("Path to model files:", path)