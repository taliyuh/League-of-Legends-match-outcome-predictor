# the same as multi_models.py but for preprocessing.py, so it's completly different

import subprocess
import sys
import os

def run_preprocessing(rank=None, queue=None):

    # generating args for configs
    cmd = [sys.executable, "preprocessing.py"]
    
    config = ""
    if rank:
        cmd.append(f"rank:={rank}")
        config += f"{rank} "
    if queue:
        cmd.append(f"queue:={queue}")
        config += f"{'SoloQ' if queue == 420 else 'Flex'}"
    
    if not config:
        config = "ALL data"
    
    print(f"\npreprocessing: {config.strip()}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"success! {config.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"unsuccess! failed: {config.strip()}")
        return False

if __name__ == "__main__":
    ranks = ['BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND', 'MASTER', 'GRANDMASTER', 'CHALLENGER']
    
    print("="*60)
    print("MULTI-PREPROCESSING")
    print("="*60)
    print("generating datasets ")
    
    # 1. all data
    run_preprocessing(rank=None, queue=None)
    
    # 2. all ranks, soloq only
    run_preprocessing(rank=None, queue=420)
    
    # 3. each rank, soloq only
    for rank in ranks:
        run_preprocessing(rank=rank, queue=420)
    
    # 4. each rank, both queues
    for rank in ranks:
        run_preprocessing(rank=rank, queue=None)
    
    print("\n" + "="*60)
    print("all datasets generated and saved!")

    print("="*60)
