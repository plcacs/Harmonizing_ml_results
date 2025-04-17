import os
import re

for filename in os.listdir("."):
    if filename.endswith(".py"):
        # Match: base.py_<suffix>_<hash>.py → base.py
        match = re.match(r"(.+?\.py)_(deepseek|gpt4|o1_mini)_[a-f0-9]+\.py$", filename)
        if match:
            new_name = match.group(1)
            if not os.path.exists(new_name):  # Avoid overwriting
                os.rename(filename, new_name)
                print(f"Renamed: {filename} → {new_name}")
