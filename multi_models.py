# runs model.py for multiple rank and queue configurations

import subprocess
import sys
import re
from datetime import datetime

# rank tier groupings
RANK_TIERS = {
    'low': ['BRONZE', 'SILVER'],
    'mid': ['GOLD', 'PLATINUM'],
    'high': ['EMERALD', 'DIAMOND'],
    'elite': ['MASTER', 'GRANDMASTER', 'CHALLENGER']
}

def run_model_training(tier=None, queue=None):

    cmd = [sys.executable, "model.py"]
    
    # generating configuration string
    config = ""
    if tier:
        cmd.append(f"tier:={tier}")
        config += f"{tier.upper()} "
    if queue:
        cmd.append(f"queue:={queue}")
        config += f"{'SoloQ' if queue == 420 else 'Flex'}"
    
    if not config:
        config = "all data"
    
    print("\n" + "="*70)
    print(f"training model: {config.strip()}")
    print("="*70)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = result.stdout
        print(output)  # Print the output to console
        
        # parse accuracy and auc from output
        accuracy_match = re.search(r'accuracy: ([\d.]+)%', output)
        auc_match = re.search(r'auc score: ([\d.]+)', output)
        
        accuracy = float(accuracy_match.group(1)) if accuracy_match else None
        auc = float(auc_match.group(1)) if auc_match else None
        
        print(f"succesfully trained: {config.strip()}")
        return True, accuracy, auc
    except subprocess.CalledProcessError as e:
        print(f"unsuccesfully untrained: {config.strip()}")
        print(f"ay carambua. your error for today is: {e}")
        return False, None, None
    except Exception as e:
        print(f"error training {config.strip()}: {e}")
        return False, None, None

if __name__ == "__main__":
    start_time = datetime.now()
    
    # skill tiers instead of individual ranks
    tiers = list(RANK_TIERS.keys())
    
    print("="*70)
    print("MULTI-MODEL TRAINING (TIERED)")
    print("="*70)
    print(f"time.is: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nour program for today is:")
    print("  1. all ranks, both queues")
    print("  2. all ranks, soloq only")
    print(f"  3. each tier individually, soloq only (4 models)")
    print(f"  4. each tier individually, both queues (4 models)")
    print(f"\nyour total is: 10 models (reduced from 20!)")
    print("="*70)
    
    results = []
    
    # 1. all data
    print("\n### BASELINE MODELS ###")
    success, acc, auc = run_model_training(tier=None, queue=None)
    results.append(("ALL_ALL", success, acc, auc))
    
    # 2. all ranks, soloq only
    success, acc, auc = run_model_training(tier=None, queue=420)
    results.append(("ALL_SoloQ", success, acc, auc))
    
    # 3. specific tier, soloq only
    print("\n### TIER-SPECIFIC MODELS (SoloQ) ###")
    for tier in tiers:
        success, acc, auc = run_model_training(tier=tier, queue=420)
        results.append((f"{tier.upper()}_SoloQ", success, acc, auc))
    
    # 4. specific tier, both queues
    print("\n### TIER-SPECIFIC MODELS (Both Queues) ###")
    for tier in tiers:
        success, acc, auc = run_model_training(tier=tier, queue=None)
        results.append((f"{tier.upper()}_ALL", success, acc, auc))
    
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
    
    successful = sum(1 for _, success, _, _ in results if success)
    total = len(results)
    
    for config, success, _, _ in results:
        status = "bravo!" if success else "not bravo!"
        print(f"  {status} {config}")
    
    print(f"\nsuccess rate: {successful}/{total} ({successful/total*100:.1f}%)")
    print("="*70)
    
    if successful == total:
        print("\nsuprisingly, all models trained successfully!")
        
        # print detailed performance table
        print("\n" + "="*70)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*70)
        print(f"{'Model':<20} {'Accuracy':>10} {'AUC':>10}")
        print("-"*70)
        
        for config, success, acc, auc in results:
            if success and acc is not None and auc is not None:
                print(f"{config:<20} {acc:>9.2f}% {auc:>10.3f}")
            elif success:
                print(f"{config:<20} {'N/A':>10} {'N/A':>10}")
        
        print("="*70)
        print("\nplease proceed to visualize_models.py now")
    else:
        print(f"\n {total - successful} model(s) flopped")
