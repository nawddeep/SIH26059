import os
import shutil
import re
from pathlib import Path

base_dir = Path("/Users/pratiksmac/Downloads/data")
raw_dir = base_dir / "raw"

# Move everything from raw_dir/YYYY to base_dir/YYYY
for item in raw_dir.iterdir():
    if item.is_dir() and re.match(r'^\d{4}$', item.name):
        dest = base_dir / item.name
        dest.mkdir(exist_ok=True)
        # Move all files
        for f in item.iterdir():
            shutil.move(str(f), str(dest / f.name))
        item.rmdir()

# Now look at files directly in base_dir
for item in base_dir.iterdir():
    if item.is_file():
        # Match year in format YYYY-MM-DD
        match = re.search(r'(\d{4})-\d{2}-\d{2}', item.name)
        if match:
            year = match.group(1)
            # If the file spans multiple years, don't move it?
            # Let's check if it spans multiple years by looking for a second year
            matches = re.findall(r'(\d{4})-\d{2}-\d{2}', item.name)
            if len(matches) >= 2 and matches[0] != matches[1]:
                print(f"Skipping {item.name} because it spans multiple years: {matches[0]} to {matches[1]}")
                continue
            
            year_dir = base_dir / year
            year_dir.mkdir(exist_ok=True)
            print(f"Moving {item.name} to {year_dir}")
            shutil.move(str(item), str(year_dir / item.name))

print("Organization complete.")
