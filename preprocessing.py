import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

class MatchPreprocessor:
    def __init__(self):
        self.feature_names = None

    def load_data(self, filepath="match_data.csv"):
        df = pd.read_csv(filepath)
        print(f"loaded {len(df)} matches")
        return df

    def engineer_features(self, df):
        # all features are differentials at 15 minutes
        df['gold_diff_15'] = df['blue_gold_15'] - df['red_gold_15']
        df['kill_diff_15'] = df['blue_kills_15'] - df['red_kills_15']
        df['cs_diff_15'] = df['blue_cs_15'] - df['red_cs_15']
        df['tower_diff_15'] = df['blue_towers_15'] - df['red_towers_15']
        df['dragon_diff_15'] = df['blue_dragons_15'] - df['red_dragons_15']

        # early game impact score (weighted) based on my experience
        df['early_impact_score'] = (
            df['gold_diff_15'] * 0.7 + 
            df['tower_diff_15'] * 1.0 + 
            df['kill_diff_15'] * 0.5 +
            df['cs_diff_15'] * 0.2
        )
        
        return df

    def create_feature_matrix(self, df):
        feature_cols = [
            'gold_diff_15', 'kill_diff_15', 'cs_diff_15',
            'tower_diff_15', 'dragon_diff_15', 'early_impact_score'
        ]
        
        X = df[feature_cols]
        y = df['blue_team_win']
        self.feature_names = feature_cols
        return X, y

    def split_and_save(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        if not os.path.exists("data"): os.makedirs("data")
        X_train.to_csv("data/X_train.csv", index=False)
        X_test.to_csv("data/X_test.csv", index=False)
        y_train.to_csv("data/y_train.csv", index=False)
        y_test.to_csv("data/y_test.csv", index=False)
        print("processed data saved to data/")
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    processor = MatchPreprocessor()
    df = processor.load_data("match_data.csv")
    df_features = processor.engineer_features(df)
    X, y = processor.create_feature_matrix(df_features)
    processor.split_and_save(X, y)
    
    # check correlation
    corrs = df_features[processor.feature_names + ['blue_team_win']].corr()['blue_team_win']
    print("\ncorrelation with winning (at 15 mins):")
    print(corrs.sort_values(ascending=False))