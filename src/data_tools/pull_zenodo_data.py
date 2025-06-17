import requests
from pathlib import Path

# Data accessed from https://zenodo.org/records/15292499
url = "https://zenodo.org/records/15292499/files/DENGSP.csv?download=1" 

project_dir = Path.cwd()
output_path = project_dir / "data" / "raw" / "DENGSP.csv"

response = requests.get(url, stream=True)
with open(output_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print("Download complete.")
