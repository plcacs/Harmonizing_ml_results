import os

for root, dirs, files in os.walk("."):
    for filename in files:
        if filename.endswith(".py.py"):
            old_path = os.path.join(root, filename)
            new_filename = filename.replace(".py.py", ".py")
            new_path = os.path.join(root, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} → {new_path}")