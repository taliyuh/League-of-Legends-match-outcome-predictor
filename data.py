# purpose of this file is to collect match data from the riot api
# for a list of players defined in players.py
# it's honestly very long but most of it is error handling and rate limiting
# and the actual data collection logic is quite straightforward

# currently my favourite movie is norbit so all the error messages are rasputia quotes

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from threading import Lock
from collections import deque
from datetime import datetime
from champion_data import get_champion_class

load_dotenv()

# timestamps for console logs, to determine how long the process has been running

start_time = None

def timestamp():
    global start_time
    if start_time is None:
        start_time = time.time()
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"

# rate limiter to handle riot api limits, so i don't get banned

class RateLimiter:
    def __init__(self, short_limit=20, short_window=1.0, long_limit=100, long_window=120.0):
        self.short_limit = short_limit
        self.short_window = short_window
        self.long_limit = long_limit
        self.long_window = long_window
        
        self.short_requests = deque()
        self.long_requests = deque()
        self.lock = Lock()
    
    def acquire(self):
        with self.lock:
            now = time.time()
            
            while self.short_requests and self.short_requests[0] < now - self.short_window:
                self.short_requests.popleft()
            
            while self.long_requests and self.long_requests[0] < now - self.long_window:
                self.long_requests.popleft()
            
            if len(self.short_requests) >= self.short_limit:
                wait_time = self.short_window - (now - self.short_requests[0])
                if wait_time > 0:
                    self.lock.release()
                    time.sleep(wait_time)
                    self.lock.acquire()
                    now = time.time()
            
            if len(self.long_requests) >= self.long_limit:
                wait_time = self.long_window - (now - self.long_requests[0])
                if wait_time > 0:
                    self.lock.release()
                    time.sleep(wait_time)
                    self.lock.acquire()
                    now = time.time()
            
            self.short_requests.append(now)
            self.long_requests.append(now)

class DataCollector:
    def __init__(self):
        self.api_key = os.getenv("RIOT_API_KEY")
        self.region = 'europe'  # the region i'm playing on
        self.platform_region = 'eun1'  # server in the region
        self.csv_path = "match_data.csv"
        self.rejections = Counter()
        self.session = requests.Session()
        self.session.headers.update({"X-Riot-Token": self.api_key})
        self.rate_limiter = RateLimiter(short_limit=20, short_window=1.0, long_limit=100, long_window=120.0)

    # summoner name and tag to puuid

    def resolve_riot_id(self, game_name, tag_line):
        url = f"https://{self.region}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        try:
            self.rate_limiter.acquire()
            res = self.session.get(url, timeout=5)
            if res.status_code == 200:
                return res.json().get('puuid')
            else:
                if res.status_code != 429:
                    print(f"{timestamp()}   HELL TO THE NO! puuid lookup failed with status {res.status_code}")
                return None
        except Exception as e:
            print(f"{timestamp()}   HELL TO THE NO! puuid lookup exception: {str(e)}")
            return None

    # self-explanatory

    def get_match_history(self, puuid, count=100):
        url = f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        count = min(count, 100)
        
        all_ids = []
        for q in [420, 440]: # 420 is soloq, 440 is flex
            params = {"queue": q, "count": count, "start": 0}
            try:
                self.rate_limiter.acquire()
                res = self.session.get(url, params=params, timeout=5)
                if res.status_code == 200:
                    matches = res.json()
                    all_ids.extend(matches)
                elif res.status_code != 429: 
                    print(f"{timestamp()}   queue {q}: api status {res.status_code}")
            except Exception as e:
                print(f"{timestamp()}   queue {q}: {str(e)}")
        return list(set(all_ids))
    
    # get player rank based on puuid

    def get_summoner_rank(self, puuid):
        try:
            url_rank = f"https://{self.platform_region}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
            self.rate_limiter.acquire()
            res = self.session.get(url_rank, timeout=5)
            if res.status_code != 200:
                return "UNRANKED", 0
            
            for entry in res.json():
                if entry['queueType'] == 'RANKED_SOLO_5x5':
                    tier = entry['tier']
                    rank = entry['rank']

                    # convert rank to number
                    rank_num = {'IV': 4, 'III': 3, 'II': 2, 'I': 1}.get(rank, 0)
                    return tier, rank_num
            return "UNRANKED", 0
        except Exception as e:
            print(f"{timestamp()}   rank lookup exception: {str(e)}")
            return "UNRANKED", 0

    # bread and butter: get match details

    def get_match_details(self, match_id, player_rank_tier="UNRANKED", player_rank_division=0):
        try:
            # get match info + error handling
            self.rate_limiter.acquire()
            res = self.session.get(f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/{match_id}", timeout=10)
            
            if res.status_code == 429:
                wait = int(res.headers.get("Retry-After", 5))
                print(f"{timestamp()} [WARNING] I AM A VERY LARGE WOMAN! rate limit exceeded - waiting {wait}s...")
                time.sleep(wait)
                return None
            
            if res.status_code != 200:
                self.rejections[f'API_Match_{res.status_code}'] += 1
                return None
            
            details = res.json()

            # get match timeline
            self.rate_limiter.acquire()
            t_res = self.session.get(f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline", timeout=10)
            
            if t_res.status_code == 429:
                wait = int(t_res.headers.get("Retry-After", 5))
                print(f"{timestamp()} [WARNING] I AM A VERY LARGE WOMAN! rate limit exceeded - waiting {wait}s...")
                time.sleep(wait)
                return None
            
            if t_res.status_code != 200:
                self.rejections[f'API_Timeline_{t_res.status_code}'] += 1
                return None
            
            timeline = t_res.json()
            frames = timeline['info']['frames']
            
            if len(frames) < 16:
                self.rejections['Game_Too_Short'] += 1
                return None

            # extract data at 15 minutes
            f15 = frames[15]['participantFrames']
            
            # gold
            b_gold = sum(f15[str(i)]['totalGold'] for i in range(1, 6))
            r_gold = sum(f15[str(i)]['totalGold'] for i in range(6, 11))
            
            # cs score (excluding lowest [support])
            b_cs_list = sorted([f15[str(i)]['minionsKilled'] + f15[str(i)]['jungleMinionsKilled'] for i in range(1, 6)])
            r_cs_list = sorted([f15[str(i)]['minionsKilled'] + f15[str(i)]['jungleMinionsKilled'] for i in range(6, 11)])

            # experience
            b_exp = sum(f15[str(i)]['xp'] for i in range(1, 6))
            r_exp = sum(f15[str(i)]['xp'] for i in range(6, 11))

            # collecting stats (kills, towers, dragons, grubs)
            b_k, r_k, b_t, r_t, b_d, r_d, b_g, r_g = 0, 0, 0, 0, 0, 0, 0, 0
            for f in frames[:16]:
                for e in f.get('events', []):
                    etype = e.get('type')
                    if etype == 'CHAMPION_KILL':
                        if 1 <= e.get('killerId', 0) <= 5: b_k += 1
                        else: r_k += 1
                    elif etype == 'BUILDING_KILL' and e.get('buildingType') == 'TOWER_BUILDING':
                        if e.get('teamId') == 200: b_t += 1
                        else: r_t += 1
                    elif etype == 'ELITE_MONSTER_KILL':
                        m, team = e.get('monsterType'), e.get('killerTeamId')
                        if m == 'DRAGON':
                            if team == 100: b_d += 1
                            else: r_d += 1
                        elif m == 'HORDE': # Void Grubs
                            if team == 100: b_g += 1
                            else: r_g += 1

            # get queue type, champions, match champions to their classes, get roles
            queue_id = details['info']['queueId']
            participants = details['info']['participants']
            
            blue_champs = [participants[i]['championId'] for i in range(5)]
            red_champs = [participants[i]['championId'] for i in range(5, 10)]
            
            blue_classes = [get_champion_class(cid) for cid in blue_champs]
            red_classes = [get_champion_class(cid) for cid in red_champs]
            
            blue_roles = [participants[i]['teamPosition'] for i in range(5)]
            red_roles = [participants[i]['teamPosition'] for i in range(5, 10)]

            return {
                'match_id': match_id, 
                'blue_team_win': int(details['info']['participants'][0]['win']),
                'queue_id': queue_id,
                'player_rank_tier': player_rank_tier,
                'player_rank_division': player_rank_division,
                'blue_gold_15': b_gold, 'red_gold_15': r_gold,
                'blue_kills_15': b_k, 'red_kills_15': r_k,
                'blue_cs_15': sum(b_cs_list[1:]), 'red_cs_15': sum(r_cs_list[1:]),
                'blue_towers_15': b_t, 'red_towers_15': r_t,
                'blue_dragons_15': b_d, 'red_dragons_15': r_d,
                'blue_grubs_15': b_g, 'red_grubs_15': r_g,
                'blue_experience_15': b_exp, 'red_experience_15': r_exp,
                # champions for each player
                'blue_champ_1': blue_champs[0], 'blue_champ_2': blue_champs[1], 'blue_champ_3': blue_champs[2],
                'blue_champ_4': blue_champs[3], 'blue_champ_5': blue_champs[4],
                'red_champ_1': red_champs[0], 'red_champ_2': red_champs[1], 'red_champ_3': red_champs[2],
                'red_champ_4': red_champs[3], 'red_champ_5': red_champs[4],
                # classes for each player
                'blue_class_1': blue_classes[0], 'blue_class_2': blue_classes[1], 'blue_class_3': blue_classes[2],
                'blue_class_4': blue_classes[3], 'blue_class_5': blue_classes[4],
                'red_class_1': red_classes[0], 'red_class_2': red_classes[1], 'red_class_3': red_classes[2],
                'red_class_4': red_classes[3], 'red_class_5': red_classes[4],
                # roles for each player
                'blue_role_1': blue_roles[0], 'blue_role_2': blue_roles[1], 'blue_role_3': blue_roles[2],
                'blue_role_4': blue_roles[3], 'blue_role_5': blue_roles[4],
                'red_role_1': red_roles[0], 'red_role_2': red_roles[1], 'red_role_3': red_roles[2],
                'red_role_4': red_roles[3], 'red_role_5': red_roles[4],
            }
        except Exception as e:
            self.rejections[f'Exception_{type(e).__name__}'] += 1
            return None

if __name__ == "__main__":
    collector = DataCollector()
    
    # load players from players.py
    try:
        from players import all_players
        
        if not all_players:
            print(f"{timestamp()} NORBIT! add some players to players.py first")
            exit()
        
        print(f"{timestamp()} loaded {len(all_players)} players")
        
        # filter out players that have already been processed
        processed_players_file = "processed_players.txt"
        processed_players = set()
        
        if os.path.exists(processed_players_file):
            with open(processed_players_file, 'r') as f:
                processed_players = set(line.strip() for line in f)
            print(f"{timestamp()}  {len(processed_players)} players already processed")
        
        new_players = [p for p in all_players if p not in processed_players]
        
        if new_players == 0:
            print(f"{timestamp()} already done all players! nothing to do")
            exit()
        
        print(f"{timestamp()} processing {len(new_players)} new players")
        print(f"{timestamp()} collecting 80 matches per player\n")
        
        # collect data for each new player
        for i, player_tag in enumerate(new_players):
            print(f"{timestamp()} [{i+1}/{len(new_players)}] now handling {player_tag}")
            
            try:
                name, tag = player_tag.split("#")
                puuid = collector.resolve_riot_id(name, tag)
                
                if not puuid:
                    print(f"{timestamp()}   i'm Rasputia Latimore-Rice and i could not resolve that puuid")
                    continue
                
                # get rank
                rank_tier, rank_div = collector.get_summoner_rank(puuid)
                print(f"{timestamp()}   rank: {rank_tier} {rank_div}")
                
                # get matches
                match_ids = collector.get_match_history(puuid, count=40)
                print(f"{timestamp()}   found {len(match_ids)} matches")
                
                # get match details
                matches_collected = 0
                for match_id in match_ids:
                    result = collector.get_match_details(match_id, rank_tier, rank_div)
                    if result:
                        # append to csv
                        df_new = pd.DataFrame([result])
                        
                        if os.path.exists("match_data.csv"):
                            df_existing = pd.read_csv("match_data.csv")
                            # check if match already exists in csv
                            if match_id not in df_existing['match_id'].values:
                                df_combined = pd.concat([df_existing, df_new])
                                df_combined.to_csv("match_data.csv", index=False)
                                matches_collected += 1
                        else:
                            df_new.to_csv("match_data.csv", index=False)
                            matches_collected += 1
                
                print(f"{timestamp()}   appended {matches_collected} new matches")
                
                # add player to processed
                processed_players.add(player_tag)
                with open(processed_players_file, 'a') as f:
                    f.write(f"{player_tag}\n")
                
            except Exception as e:
                print(f"{timestamp()}   HELL TO THE NO! error: {str(e)}")
                continue
        
        # print something in the terminal
        if os.path.exists("match_data.csv"):
            df = pd.read_csv("match_data.csv")
            print(f"\n{timestamp()} === Collection Complete ===")
            print(f"{timestamp()} Total unique matches: {len(df)}")
            print(f"{timestamp()} Players processed: {len(processed_players)}")
        
    except ImportError:
        print(f"{timestamp()} HOW YOU DOIN? bad because could not import from players.py")
        exit()