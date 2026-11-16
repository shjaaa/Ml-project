import numpy as np
from pathlib import Path

for file in Path("dataset").glob("*.npz"):
    print(f"\nFile: {file.name}")

    data = np.load(file)

    for key in data.files:
        print(f"  {key}: {data[key].shape}")