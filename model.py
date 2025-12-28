import xgboost as xgb
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class LoLPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def train(self, X_train, y_train, X_test, y_test, feature_names):
        # store feature names for later use
        self.feature_names = feature_names
        
        # convert to xgboost's efficient matrix format
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # hyperparameters tuned for 15-minute lol predictions
        params = {
            'objective': 'binary:logistic',  # predicts win probability
            'max_depth': 5,  # prevents overfitting on small dataset
            'learning_rate': 0.1,  # faster convergence
            'subsample': 0.8,  # random sampling of rows
            'colsample_bytree': 0.8,  # random sampling of features
            'eval_metric': 'logloss',  # measures probability accuracy
            'tree_method': 'hist',  # faster training
            'device': 'cpu'
        }
        
        # training with early stopping to prevent overfitting
        evals = [(dtrain, 'train'), (dtest, 'eval')]
        self.model = xgb.train(
            params, dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=30,
            verbose_eval=50
        )
        
        # evaluate on test set
        test_probs = self.model.predict(dtest)
        test_preds = (test_probs > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, test_preds)
        auc = roc_auc_score(y_test, test_probs)
        
        print(f"\n{'='*50}")
        print(f"model performance on test set:")
        print(f"accuracy: {accuracy:.2%}")
        print(f"auc score: {auc:.3f}")
        print(f"{'='*50}")
        print("\ndetailed classification report:")
        print(classification_report(y_test, test_preds, target_names=['red win', 'blue win']))
        
        return self.model
    
    def predict_single(self, match_features):
        # predict win probability for a single match
        if self.model is None:
            raise ValueError("model not trained or loaded")
        
        # convert dict to dataframe if needed
        if isinstance(match_features, dict):
            match_features = pd.DataFrame([match_features])
        
        dmatrix = xgb.DMatrix(match_features)
        probability = self.model.predict(dmatrix)[0]
        return probability
    
    def get_feature_importance(self, X_sample):
        # get feature importance using built-in method
        if self.model is None:
            raise ValueError("model not trained or loaded")
        
        importance = self.model.get_score(importance_type='gain')
        # sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_importance
    
    def save_model(self, filepath="models/lol_predictor.json"):
        # create models directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.model:
            self.model.save_model(filepath)
            print(f"\nmodel saved to {filepath}")
            # also save feature names for the app
            pd.DataFrame({'features': self.feature_names}).to_csv("models/feature_names.csv", index=False)
    
    def load_model(self, filepath="models/lol_predictor.json"):
        # load pre-trained model
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        
        # load feature names
        if os.path.exists("models/feature_names.csv"):
            self.feature_names = pd.read_csv("models/feature_names.csv")['features'].tolist()
        
        print(f"model loaded from {filepath}")

if __name__ == "__main__":
    # check if processed data exists
    if not os.path.exists("data/X_train.csv"):
        print("error: processed data not found. run preprocessing.py first.")
        exit()
    
    # load processed data
    print("loading processed data...")
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv")['blue_team_win']
    y_test = pd.read_csv("data/y_test.csv")['blue_team_win']
    
    print(f"training set: {X_train.shape[0]} matches")
    print(f"test set: {X_test.shape[0]} matches")
    print(f"features: {X_train.shape[1]}")
    
    # train model
    predictor = LoLPredictor()
    predictor.train(X_train, y_train, X_test, y_test, X_train.columns.tolist())
    
    # feature importance
    print("\ntop features by importance:")
    importance = predictor.get_feature_importance(X_test)
    for feature, score in importance[:5]:
        print(f"  {feature}: {score:.2f}")
    
    # save model
    predictor.save_model()
    
    # quick test prediction
    print(f"\nquick test on first test sample:")
    sample = X_test.iloc[0:1]
    prob = predictor.predict_single(sample)
    actual = y_test.iloc[0]
    print(f"predicted blue win probability: {prob:.2%}")
    print(f"actual result: {'blue win' if actual == 1 else 'red win'}")