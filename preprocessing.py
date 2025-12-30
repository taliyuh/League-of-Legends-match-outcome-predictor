# preprocessing the data, engineering features, saving train/test splits

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

class MatchPreprocessor:
    def __init__(self):
        self.feature_names = None

    def load_data(self, filepath="match_data.csv"):
        # load raw match data from CSV
        df = pd.read_csv(filepath)
        print(f"loaded {len(df)} matches")
        
        # filter out matches with more than 3 void grubs total (old matches, currently the limit is 3 grubs)
        df['total_grubs'] = df['blue_grubs_15'] + df['red_grubs_15']
        initial_count = len(df)
        df = df[df['total_grubs'] <= 3].copy()
        df = df.drop(columns=['total_grubs'])
        
        return df

    def engineer_features(self, df, include_champion_features=False, include_rank=False):
        # all features are differentials at 15 minutes
        df['gold_diff_15'] = df['blue_gold_15'] - df['red_gold_15']
        df['kill_diff_15'] = df['blue_kills_15'] - df['red_kills_15']
        df['cs_diff_15'] = df['blue_cs_15'] - df['red_cs_15']
        df['tower_diff_15'] = df['blue_towers_15'] - df['red_towers_15']
        df['dragon_diff_15'] = df['blue_dragons_15'] - df['red_dragons_15']
        df['grub_diff_15'] = df['blue_grubs_15'] - df['red_grubs_15']
        df['experience_diff_15'] = df['blue_experience_15'] - df['red_experience_15']

        # which queue
        df['is_soloq'] = (df['queue_id'] == 420).astype(int)

        # rank encoding 
        if include_rank and 'player_rank_tier' in df.columns:
            rank_order = {
                'IRON': 1, 'BRONZE': 2, 'SILVER': 3, 'GOLD': 4, 
                'PLATINUM': 5, 'EMERALD': 6, 'DIAMOND': 7, 
                'MASTER': 8, 'GRANDMASTER': 9, 'CHALLENGER': 10, 'UNRANKED': 0
            }
            df['rank_numeric'] = df['player_rank_tier'].map(rank_order) * 4 + (4 - df['player_rank_division'])
        
        # champion classes
        if include_champion_features and 'blue_class_1' in df.columns:
            classes = ['Tank', 'Fighter', 'Assassin', 'Mage', 'Marksman', 'Support', 'Enchanter']
            for team in ['blue', 'red']:
                for cls in classes:
                    df[f'{team}_{cls.lower()}_count'] = sum(
                        df[f'{team}_class_{i}'] == cls for i in range(1, 6)
                    )

        # early game impact score (weighted) based on my personal experience
        df['early_impact_score'] = (
            df['gold_diff_15'] * 0.7 + 
            df['tower_diff_15'] * 1.0 + 
            df['kill_diff_15'] * 0.5 +
            df['cs_diff_15'] * 0.2 +
            df['grub_diff_15'] * 0.15 +
            df['experience_diff_15'] * 0.4
        )
        
        return df

    def create_feature_matrix(self, df, include_champion_features=False, include_rank=False):
        # final features for model
        feature_cols = [
            'gold_diff_15', 'kill_diff_15', 'cs_diff_15',
            'tower_diff_15', 'dragon_diff_15', 'grub_diff_15',
            'experience_diff_15', 'early_impact_score', 'is_soloq'
        ]
        
        # include rank
        if include_rank and 'rank_numeric' in df.columns:
            feature_cols.append('rank_numeric')
        
        # include also champion class counts
        if include_champion_features:
            classes = ['tank', 'fighter', 'assassin', 'mage', 'marksman', 'support', 'Enchanter']
            for team in ['blue', 'red']:
                for cls in classes:
                    col_name = f'{team}_{cls}_count'
                    if col_name in df.columns:
                        feature_cols.append(col_name)
        
        X = df[feature_cols]
        y = df['blue_team_win']
        
        self.feature_names = feature_cols
        return X, y

    def split_and_save(self, X, y):
        # split into train and test sets
        # good model because we have separate train and test sets and not everyone has that
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    import sys
    
    target_rank = None
    target_queue = None
    
    for arg in sys.argv[1:]:
        if arg.startswith('rank:='):
            target_rank = arg.split(':=')[1].upper()
        elif arg.startswith('queue:='):
            target_queue = int(arg.split(':=')[1])
    
    processor = MatchPreprocessor()
    
    # loading our glorious data (it took ages to collect)
    df = processor.load_data("match_data.csv")
    
    # filtering based on command line args
    if target_rank:
        initial_count = len(df)
        df = df[df['player_rank_tier'] == target_rank]
        print(f"only matches from {target_rank}: {len(df)} from {initial_count}")
    
    if target_queue:
        initial_count = len(df)
        df = df[df['queue_id'] == target_queue]
        queue_name = "SoloQ" if target_queue == 420 else "Flex"
        print(f"only matches from {queue_name}: {len(df)} from {initial_count}")
    
    if len(df) < 50:
        print(f"\n[ERROR] not enough data after filtering ({len(df)} matches)")
        exit()
    
    # engineering the features
    df_features = processor.engineer_features(
        df, 
        include_champion_features=False,
        include_rank=False
    )
    X, y = processor.create_feature_matrix(
        df_features,
        include_champion_features=False,
        include_rank=False
    )
    
    X_train, X_test, y_train, y_test = processor.split_and_save(X, y)
    
    # determine target folder based on rank + queue type
    config_name = ""
    if target_rank:
        config_name += target_rank.lower()
    if target_queue:
        if config_name:
            config_name += "_"
        config_name += "soloq" if target_queue == 420 else "flex"
    
    if not config_name:
        config_name = "all"
    
    # ssave data there
    config_dir = f"data/{config_name}"
    os.makedirs(config_dir, exist_ok=True)
    
    X_train.to_csv(f"{config_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{config_dir}/X_test.csv", index=False)
    y_train.to_frame('blue_team_win').to_csv(f"{config_dir}/y_train.csv", index=False)
    y_test.to_frame('blue_team_win').to_csv(f"{config_dir}/y_test.csv", index=False)
 
    correlations = df_features[processor.feature_names + ['blue_team_win']].corr()['blue_team_win'].sort_values(ascending=False)
    print("\ncorrelation with winning for nerds:")
    print(correlations)