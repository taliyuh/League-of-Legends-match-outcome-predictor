# the same as multi_models.py but for preprocessing.py, so it's completly different

import subprocess
import sys
import os

# rank tier groupings
RANK_TIERS = {
    'low': ['BRONZE', 'SILVER'],
    'mid': ['GOLD', 'PLATINUM'],
    'high': ['EMERALD', 'DIAMOND'],
    'elite': ['MASTER', 'GRANDMASTER', 'CHALLENGER']
}

def run_preprocessing(tier=None, queue=None):

    # generating args for configs
    cmd = [sys.executable, "preprocessing.py"]
    
    config = ""
    if tier:
        cmd.append(f"tier:={tier}")
        config += f"{tier.upper()} "
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
    tiers = list(RANK_TIERS.keys())
    
    print("="*60)
    print("MULTI-PREPROCESSING (TIERED)")
    print("="*60)
    print("generating datasets for 4 skill tiers")
    print(f"tiers: {', '.join([t.upper() for t in tiers])}")
    
    # 1. all data
    run_preprocessing(tier=None, queue=None)
    
    # 2. all ranks, soloq only
    run_preprocessing(tier=None, queue=420)
    
    # 3. each tier, soloq only
    for tier in tiers:
        run_preprocessing(tier=tier, queue=420)
    
    # 4. each tier, both queues
    for tier in tiers:
        run_preprocessing(tier=tier, queue=None)
    
    print("\n" + "="*60)
    print("all datasets generated and saved!")
    print("="*60)
