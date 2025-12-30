# runs model.py for multiple rank and queue configurations

import subprocess
import sys
from datetime import datetime

def run_model_training(rank=None, queue=None):

    cmd = [sys.executable, "model.py"]
    
    # generating configuration string
    config = ""
    if rank:
        cmd.append(f"rank:={rank}")
        config += f"{rank} "
    if queue:
        cmd.append(f"queue:={queue}")
        config += f"{'SoloQ' if queue == 420 else 'Flex'}"
    
    if not config:
        config = "all data"
    
    print("\n" + "="*70)
    print(f"training model: {config.strip()}")
    print("="*70)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"succesfully trained: {config.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"unsuccesfully untrained: {config.strip()}")
        print(f"ay carambua. your error for today is: {e}")
        return False
    except Exception as e:
        print(f"error training {config.strip()}: {e}")
        return False

if __name__ == "__main__":
    start_time = datetime.now()
    
    # ranks, ignoring iron because players there are too unskilled to matter
    ranks = ['BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND', 'MASTER', 'GRANDMASTER', 'CHALLENGER']
    
    print("="*70)
    print("MULTI-MODEL TRAINING")
    print("="*70)
    print(f"time.is: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nour program for today is:")
    print("  1. all ranks, both queues")
    print("  2. all ranks, soloq only")
    print("  3. each rank individually, soloq only (9 models)")
    print("  4. each rank individually, both queues (9 models)")
    print(f"\nyour total is: 20 models")
    print("="*70)
    
    results = []
    
    # 1. all data
    print("\n### BASELINE MODELS ###")
    success = run_model_training(rank=None, queue=None)
    results.append(("ALL_ALL", success))
    
    # 2. all ranks, soloq only
    success = run_model_training(rank=None, queue=420)
    results.append(("ALL_SoloQ", success))
    
    # 3. specific rank, soloq only
    print("\n### RANK-SPECIFIC MODELS (SoloQ) ###")
    for rank in ranks:
        success = run_model_training(rank=rank, queue=420)
        results.append((f"{rank}_SoloQ", success))
    
    # 4. specific rank, both queues
    print("\n### RANK-SPECIFIC MODELS (Both Queues) ###")
    for rank in ranks:
        success = run_model_training(rank=rank, queue=None)
        results.append((f"{rank}_ALL", success))
    
    # yeah that's all
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"wasted energy for: {duration}")
    print(f"\nresults:")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for config, success in results:
        status = "bravo!" if success else "not bravo!"
        print(f"  {status} {config}")
    
    print(f"\nsuccess rate: {successful}/{total} ({successful/total*100:.1f}%)")
    print("="*70)
    
    if successful == total:
        print("\nsuprisingly, all models trained successfully!")
        print("please proceed to visualize_models.py now")
    else:
        print(f"\n {total - successful} model(s) flopped")
