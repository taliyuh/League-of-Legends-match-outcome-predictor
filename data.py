import os
import requests
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class DataCollector:
    def __init__(self):
        self.api_key = os.getenv("RIOT_API_KEY")
        self.platform = 'eun1' #name of server
        self.region = 'europe'
        self.periodt = 1.2 
        self.headers = {"X-Riot-Token": self.api_key}

    # converting riot id (name#tag) to PUUID
    def resolve_riot_id(self, game_name, tag_line):
        url = f"https://{self.region}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        response = requests.get(url, headers=self.headers)
        return response.json()['puuid'] if response.status_code == 200 else None

    # getting match history for a PUUID
    def get_match_history(self, puuid, count=100):
        url = f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"queue": 440, "count": count, "start": 0} # 440 is ranked flex queue, 420 is soloq
        response = requests.get(url, headers=self.headers, params=params)
        return response.json() if response.status_code == 200 else []

    # getting detailed match data
    def get_match_details(self, match_id):
        time.sleep(self.periodt)
        try:
            url = f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
            res = requests.get(url, headers=self.headers)
            if res.status_code != 200: return None
            details = res.json()
            
            t_url = f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
            t_res = requests.get(t_url, headers=self.headers)
            if t_res.status_code != 200: return None
            timeline = t_res.json()

            frames = timeline['info']['frames']
            if len(frames) < 16: return None 
            
            target_frame = frames[15] # we asses win probability at 15 minutes
            p_frames = target_frame['participantFrames']
            
            blue_cs, red_cs = [], []
            blue_gold, red_gold = 0, 0
            
            for pid in range(1, 11):
                p_data = p_frames[str(pid)]
                cs = p_data['minionsKilled'] + p_data['jungleMinionsKilled']
                gold = p_data['totalGold']
                
                if pid <= 5:
                    blue_cs.append(cs); blue_gold += gold
                else:
                    red_cs.append(cs); red_gold += gold

            # remove supports cs score
            blue_cs.remove(min(blue_cs)) 
            red_cs.remove(min(red_cs))

            b_kills, r_kills = 0, 0
            b_towers, r_towers = 0, 0
            b_drag, r_drag = 0, 0

            for frame in frames[:16]:
                for e in frame.get('events', []):
                    if e.get('type') == 'CHAMPION_KILL':
                        if 1 <= e.get('killerId', 0) <= 5: b_kills += 1
                        else: r_kills += 1
                    elif e.get('type') == 'BUILDING_KILL' and e.get('buildingType') == 'TOWER_BUILDING':
                        if e.get('teamId') == 200: b_towers += 1 
                        else: r_towers += 1
                    elif e.get('type') == 'ELITE_MONSTER_KILL':
                        m_type = e.get('monsterType')
                        team_id = e.get('killerTeamId')
                        
                        if m_type == 'DRAGON':
                            if team_id == 100: b_drag += 1
                            else: r_drag += 1

            return {
                'match_id': match_id,
                'blue_team_win': int(details['info']['participants'][0]['win']),
                'blue_gold_15': blue_gold, 'red_gold_15': red_gold,
                'blue_kills_15': b_kills, 'red_kills_15': r_kills,
                'blue_cs_15': sum(blue_cs), 'red_cs_15': sum(red_cs),
                'blue_towers_15': b_towers, 'red_towers_15': r_towers,
                'blue_dragons_15': b_drag, 'red_dragons_15': r_drag
            }
        except Exception as e:
            print(f"error processing {match_id}: {e}")
            return None

    def collect_data(self, summoner_identifiers, match_count=50):
        all_match_data = []
        for identifier in summoner_identifiers:
            print(f"collecting data for: {identifier}")
            if "#" in identifier:
                game_name, tag_line = identifier.split("#", 1)
                puuid = self.resolve_riot_id(game_name, tag_line)
            else:
                puuid = identifier
            
            if not puuid:
                print(f"failed to resolve {identifier}")
                continue
            
            match_ids = self.get_match_history(puuid, count=match_count)
            print(f"found {len(match_ids)} matches")
            
            for match_id in match_ids:
                match_data = self.get_match_details(match_id)
                if match_data:
                    all_match_data.append(match_data)
            
            time.sleep(2) # be nice to riot's servers
        
        return pd.DataFrame(all_match_data)

if __name__ == "__main__":
    collector = DataCollector()
    summoners = ["charli xxl#chop", "orejnik#joric", "CaitThatLean#EUNE", "ice spice latte#6389"]
    data = collector.collect_data(summoners, match_count=50)
    
    if not data.empty:
        data.to_csv("match_data.csv", index=False)
        print(f"saved {len(data)} matches to match_data.csv")
    else:
        print("error: no data collected")