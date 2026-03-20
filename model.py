from pathlib import Path

model_dir = Path("/home/albasha/mistral_models/7B-Instruct-v0.3")
print(model_dir.exists())
print([p.name for p in model_dir.iterdir()])
