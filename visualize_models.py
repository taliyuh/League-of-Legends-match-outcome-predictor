# last file, hopefully short

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

sns.set_style("darkgrid")

def extract_model_config(filename):
    # for 100th time, get rank and queue
    # format: lol_predictor[_rank][_queue].json
    name = filename.replace('lol_predictor', '').replace('.json', '')
    
    rank = None
    queue = None
    
    parts = name.strip('_').split('_')
    
    # probably could done some macro because why is it in every file
    # but we are already like 2 weeks after deadline
    for part in parts:
        if part.upper() in ['BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND', 'MASTER', 'GRANDMASTER', 'CHALLENGER']:
            rank = part.upper()
        elif part in ['soloq', 'flex']:
            queue = 420 if part == 'soloq' else 440
    
    return rank if rank else 'ALL', queue if queue else 'ALL'

def load_model_performance(model_path):
    # load model and evaluate on corresponding test set
    try:
        # --||--
        model = xgb.Booster()
        model.load_model(model_path)
        
        # choose the right model
        filename = os.path.basename(model_path)
        rank, queue = extract_model_config(filename)
        
        # find the dir
        config_name = ""
        if rank != 'ALL':
            config_name = rank.lower()
        if queue != 'ALL':
            if config_name:
                config_name += "_"
            config_name += "soloq" if queue == 420 else "flex"
        
        if not config_name:
            config_name = "all"
        
        data_dir = f"data/{config_name}"
        
        # find the config
        X_test = pd.read_csv(f"{data_dir}/X_test.csv")
        y_test = pd.read_csv(f"{data_dir}/y_test.csv")['blue_team_win']
        
        # prediction phase (i predict that we get 5 for this assignment :))
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'samples': len(y_test)
        }
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

def visualize_rank_comparison():
    # compare across the ranks and queues
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        print("[ERROR] models/ directory not found!")
        return
    
    # collect the victoria secret models
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.json') and f.startswith('lol_predictor')]
    
    print(f"found {len(model_files)} models")
    
    # load their performance
    results = []
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        rank, queue = extract_model_config(model_file)
        
        print(f"Loading {model_file}...")
        perf = load_model_performance(model_path)
        
        if perf:
            results.append({
                'model': model_file,
                'rank': rank,
                'queue': 'SoloQ' if queue == 420 else 'Both' if queue == 'ALL' else 'Flex',
                'accuracy': perf['accuracy'] * 100,
                'auc': perf['auc'],
                'samples': perf['samples']
            })
    
    if not results:
        print("no valid results to visualize!")
        return
    
    df = pd.DataFrame(results)
    
    # dir for storing the charts
    os.makedirs('charts', exist_ok=True)
    
    # 1. accuracy by rank (soloq only)
    plt.figure(figsize=(14, 7))
    soloq_df = df[df['queue'] == 'SoloQ'].copy()
    
    if len(soloq_df) > 0:
        rank_order = ['ALL', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND', 'MASTER', 'GRANDMASTER', 'CHALLENGER']
        soloq_df['rank'] = pd.Categorical(soloq_df['rank'], categories=rank_order, ordered=True)
        soloq_df = soloq_df.sort_values('rank')
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(soloq_df)))
        bars = plt.bar(soloq_df['rank'], soloq_df['accuracy'], color=colors)
        
        plt.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
        plt.xlabel('rank', fontsize=12)
        plt.ylabel('accuracy (%)', fontsize=12)
        plt.title('model accuracy by rank (SoloQ)', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # labeling
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('charts/model_accuracy_by_rank_soloq.png', dpi=300, bbox_inches='tight')
        print("saved: charts/model_accuracy_by_rank_soloq.png")
        plt.close()
    
    # 2. accuracy by rank (both queues))
    plt.figure(figsize=(14, 7))
    both_df = df[df['queue'] == 'Both'].copy()
    
    if len(both_df) > 0:
        both_df['rank'] = pd.Categorical(both_df['rank'], categories=rank_order, ordered=True)
        both_df = both_df.sort_values('rank')
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(both_df)))
        bars = plt.bar(both_df['rank'], both_df['accuracy'], color=colors)
        
        plt.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
        plt.xlabel('rank', fontsize=12)
        plt.ylabel('accuracy (%)', fontsize=12)
        plt.title('model accuracy by rank (both queues)', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('charts/model_accuracy_by_rank_both.png', dpi=300, bbox_inches='tight')
        print("saved: charts/model_accuracy_by_rank_both.png")
        plt.close()
    
    # 3. soloq vs both queues comparison
    plt.figure(figsize=(14, 7))

    comparison_data = []
    for rank in rank_order:
        soloq_acc = soloq_df[soloq_df['rank'] == rank]['accuracy'].values
        both_acc = both_df[both_df['rank'] == rank]['accuracy'].values
        
        if len(soloq_acc) > 0 and len(both_acc) > 0:
            comparison_data.append({
                'rank': rank,
                'soloq': soloq_acc[0],
                'both': both_acc[0]
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        x = np.arange(len(comp_df))
        width = 0.35
        
        plt.bar(x - width/2, comp_df['soloq'], width, label='SoloQ Only', color='#3498db')
        plt.bar(x + width/2, comp_df['both'], width, label='Both Queues', color='#e74c3c')
        
        plt.axhline(y=50, color='black', linestyle='--', linewidth=2, alpha=0.5)
        plt.xlabel('rank', fontsize=12)
        plt.ylabel('accuracy (%)', fontsize=12)
        plt.title('soloq vs both queues: model accuracy comparison', fontsize=16, fontweight='bold')
        plt.xticks(x, comp_df['rank'], rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('charts/model_soloq_vs_both.png', dpi=300, bbox_inches='tight')
        print("saved: charts/model_soloq_vs_both.png")
        plt.close()
    
    # 4. auc comparisment
    plt.figure(figsize=(14, 7))
    
    all_models = df.sort_values(['queue', 'rank'])
    x_labels = [f"{row['rank']}\n({row['queue']})" for _, row in all_models.iterrows()]
    
    colors = ['#3498db' if q == 'SoloQ' else '#e74c3c' if q == 'Both' else '#95a5a6' 
              for q in all_models['queue']]
    
    bars = plt.bar(range(len(all_models)), all_models['auc'], color=colors)
    
    plt.xlabel('model configuration', fontsize=12)
    plt.ylabel('auc Score', fontsize=12)
    plt.title('model auc scores across all configurations', fontsize=16, fontweight='bold')
    plt.xticks(range(len(all_models)), x_labels, rotation=45, ha='right', fontsize=8)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # labelling
    for i, (bar, auc) in enumerate(zip(bars, all_models['auc'])):
        plt.text(i, auc, f'{auc:.3f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('charts/model_auc_comparison.png', dpi=300, bbox_inches='tight')
    print("saved: charts/model_auc_comparison.png")
    plt.close()
    
    # 5. summarization table
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'model':<40} {'queue':<10} {'accuracy':<12} {'auc':<8} {'samples':<10}")
    print("-"*80)
    
    for _, row in df.sort_values(['queue', 'rank']).iterrows():
        print(f"{row['rank']:<40} {row['queue']:<10} {row['accuracy']:>6.2f}%     {row['auc']:>6.3f}   {row['samples']:<10}")
    
    print("="*80)
    
    # awards best models
    print("\n BEST MODELS:")
    best_acc = df.loc[df['accuracy'].idxmax()]
    best_auc = df.loc[df['auc'].idxmax()]
    
    print(f"  highest accuracy: {best_acc['rank']} ({best_acc['queue']}) - {best_acc['accuracy']:.2f}%")
    print(f"  highest auc:      {best_auc['rank']} ({best_auc['queue']}) - {best_auc['auc']:.3f}")
    
    print("\n all visualizations saved to charts/")

if __name__ == "__main__":
    visualize_rank_comparison()

# ok was so not short