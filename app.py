import os
import sys

pipeline = ["data.py", "preprocessing.py", "model.py"]

print("started pipeline")

for script in pipeline:
    print(f"> now onto {script}...")
    
    exit_code = os.system(f"{sys.executable} {script}")
    
    if exit_code != 0:
        print(f"error in {script}. stopped.")
        break
else:
    print("finished pipeline successfully!")