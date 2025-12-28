import time
import pandas as pd
import requests
import urllib3
import os
from datetime import datetime
from data import DataCollector
from processing import MatchPreprocessor
from model import LoLPredictor

# ignore local client ssl warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class LoLAutomatedPredictor:
    def __init__(self):
        self.collector = DataCollector()
        self.preprocessor = MatchPreprocessor()
        self.predictor = LoLPredictor()
        
        self.live_api_url = "https://127.0.0.1:2999/liveclientdata/allgamedata"
        self.trained_for_lobby = False
        self.prediction_made = False
        self.current_match_id = None
        self.last_prediction = None # stores {prob, winner}

    def get_live_data(self):
        try:
            response = requests.get(self.live_api_url, verify=False, timeout=1)
            return response.json() if response.status_code == 200 else None
        except:
            return None

    def run_lobby_training(self, player_list):
        print(f"\n[system] lobby detected. training model on players' history...")
        # collect 15 matches for each of the 10 players
        raw_data = self.collector.collect_data(player_list, match_count=15)
        
        if raw_data.empty:
            print("[error] could not gather enough data for training.")
            return False

        df_features = self.preprocessor.engineer_features(raw_data)
        X, y = self.preprocessor.create_feature_matrix(df_features)
        X_train, X_test, y_train, y_test = self.preprocessor.split_and_save(X, y)
        self.predictor.train(X_train, y_train, X_test, y_test, X.columns.tolist())
        
        print("[success] model specialized for this match is ready.")
        return True

    def estimate_gold(self, kills, cs, game_time_sec):
        # estimate gold lead since enemy gold is hidden in live api
        passive_gold = (game_time_sec - 110) * 2.04 if game_time_sec > 110 else 0
        return 500 + passive_gold + (kills * 300) + (cs * 19)

    def make_live_prediction(self, data):
        game_time = data['gameData']['gameTime']
        print(f"\n[system] 15:00 reached. analyzing live match state...")
        
        blue_kills, red_kills = 0, 0
        blue_cs_list, red_cs_list = [], []
        
        for p in data['allPlayers']:
            kills = p['scores']['kills']
            cs = p['scores']['creepScore']
            if p['team'] == 'ORDER':
                blue_kills += kills; blue_cs_list.append(cs)
            else:
                red_kills += kills; red_cs_list.append(cs)

        # support filter
        blue_cs_list.remove(min(blue_cs_list))
        red_cs_list.remove(min(red_cs_list))
        
        # estimate gold
        b_gold_est = self.estimate_gold(blue_kills, sum(blue_cs_list), game_time)
        r_gold_est = self.estimate_gold(red_kills, sum(red_cs_list), game_time)

        # extract objectives
        b_towers, r_towers, b_dragons, r_dragons, b_grubs, r_grubs = 0, 0, 0, 0, 0, 0
        for event in data['events']['Events']:
            name = event['EventName']
            if name == 'TurretKilled':
                if "Order" in event['KillerName']: b_towers += 1
                else: r_towers += 1
            elif name == 'DragonKill':
                if "Order" in event['KillerName']: b_dragons += 1
                else: r_dragons += 1
            elif name == 'HordeKill':
                if "Order" in event['KillerName']: b_grubs += 1
                else: r_grubs += 1

        live_stats = {
            'gold_diff_15': b_gold_est - r_gold_est,
            'kill_diff_15': blue_kills - red_kills,
            'cs_diff_15': sum(blue_cs_list) - sum(red_cs_list),
            'tower_diff_15': b_towers - r_towers,
            'dragon_diff_15': b_dragons - r_dragons,
            'grub_diff_15': b_grubs - r_grubs,
        }
        
        live_stats['early_impact_score'] = (
            live_stats['gold_diff_15'] * 0.7 + live_stats['tower_diff_15'] * 1.0 + 
            live_stats['kill_diff_15'] * 0.4 + live_stats['cs_diff_15'] * 0.2 + 
            live_stats['grub_diff_15'] * 0.3
        )

        prob = self.predictor.predict_single(live_stats)
        winner = "BLUE" if prob > 0.5 else "RED"
        confidence = prob if prob > 0.5 else 1 - prob
        
        print("\n" + "="*40 + f"\n MATCH PREDICTION: {winner}\n Confidence: {confidence:.2%}\n" + "="*40)
        
        # store for logging later
        self.last_prediction = {'winner': winner, 'probability': confidence}

    def log_match_result(self, player_name):
        """waits for game to end, then logs the accuracy of the prediction"""
        print("[system] match ended. waiting for riot servers to update...")
        time.sleep(120) # wait 2 mins for match to appear in history
        
        try:
            puuid = self.collector.resolve_riot_id(player_name.split("#")[0], player_name.split("#")[1])
            last_match_id = self.collector.get_match_history(puuid, count=1)[0]
            
            # fetch final result
            url = f"https://{self.collector.region}.api.riotgames.com/lol/match/v5/matches/{last_match_id}"
            res = requests.get(url, headers=self.collector.headers).json()
            
            actual_winner = "BLUE" if res['info']['participants'][0]['win'] else "RED"
            prediction_correct = (self.last_prediction['winner'] == actual_winner)
            
            # save to csv
            log_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'match_id': last_match_id,
                'predicted_winner': self.last_prediction['winner'],
                'confidence': self.last_prediction['probability'],
                'actual_winner': actual_winner,
                'correct': int(prediction_correct)
            }
            
            os.makedirs("logs", exist_ok=True)
            log_df = pd.DataFrame([log_data])
            log_df.to_csv("logs/prediction_history.csv", mode='a', header=not os.path.exists("logs/prediction_history.csv"), index=False)
            
            print(f"[log] prediction was {'CORRECT' if prediction_correct else 'WRONG'}. saved to logs/.")
            
        except Exception as e:
            print(f"[log error] could not verify match result: {e}")

    def start_loop(self, main_player_riot_id):
        print(f"[!] automated system running for: {main_player_riot_id}")
        while True:
            data = self.get_live_data()
            if data:
                game_time = data['gameData']['gameTime']
                if game_time < 180 and not self.trained_for_lobby:
                    players = [p['summonerName'] for p in data['allPlayers']]
                    # note: if live names don't have tags, you'll need a helper to add them
                    self.trained_for_lobby = self.run_lobby_training(players)
                
                if 900 <= game_time <= 915 and not self.prediction_made:
                    if self.trained_for_lobby:
                        self.make_live_prediction(data)
                        self.prediction_made = True
            else:
                if self.prediction_made and self.last_prediction:
                    self.log_match_result(main_player_riot_id)
                    self.trained_for_lobby = False
                    self.prediction_made = False
                    self.last_prediction = None
            time.sleep(5)

if __name__ == "__main__":
    # replace with your own riot id
    app = LoLAutomatedPredictor()
    app.start_loop("charli xxl#chop")