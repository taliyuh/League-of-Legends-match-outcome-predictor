# visualisation for nerds (sorry for inconsistency in writing visualiSe)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import requests

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 10)

def fetch_champion_names():
    # champion id -> champion name
    try:
        # get latest version
        version_url = "https://ddragon.leagueoflegends.com/api/versions.json"
        versions = requests.get(version_url, timeout=5).json()
        latest_version = versions[0]
        
        # get champion data
        champ_url = f"https://ddragon.leagueoflegends.com/cdn/{latest_version}/data/en_US/champion.json"
        champ_data = requests.get(champ_url, timeout=10).json()
        
        # actual mapping (bad input bad output)
        id_to_name = {}
        for champ_name, champ_info in champ_data['data'].items():
            champ_id = int(champ_info['key'])
            id_to_name[champ_id] = champ_info['name']
        
        print(f"loaded {len(id_to_name)} champion names from DarkWeb")
        return id_to_name
    except Exception as e:
        print(f"failed to load champion names: {e}")
        return {}

# global champion names cache
CHAMPION_NAMES = fetch_champion_names()

def load_data():
    # loading the data from .csv
    try:
        df = pd.read_csv('match_data.csv')
        print(f"Loaded {len(df)} matches")
        return df
    except FileNotFoundError:
        print("Error: match_data.csv not found!")
        exit()

def visualize_rank_distribution(df):
    # distribution of matches by rank
    plt.figure(figsize=(12, 6))
    
    rank_order = ['BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND', 'MASTER', 'GRANDMASTER', 'CHALLENGER', 'UNRANKED']
    rank_counts = df['player_rank_tier'].value_counts()
    
    # sort by ranks
    sorted_ranks = [r for r in rank_order if r in rank_counts.index]
    sorted_counts = [rank_counts[r] for r in sorted_ranks]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_ranks)))
    bars = plt.bar(sorted_ranks, sorted_counts, color=colors)
    
    plt.title('matcha distribution by player rank', fontsize=16, fontweight='bold') # labubu matcha
    plt.xlabel('rank', fontsize=12)
    plt.ylabel('number of matches', fontsize=12)
    plt.xticks(rotation=45)
    
    # some chart generation
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('charts/rank_distribution.png', dpi=300, bbox_inches='tight')
    print("saved: charts/rank_distribution.png")
    plt.close()

def visualize_win_rate_by_rank(df):
    # win rate by rank, boring, wouldnt do it again
    plt.figure(figsize=(12, 6))
    
    rank_order = ['BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND', 'MASTER', 'GRANDMASTER', 'CHALLENGER']
    
    win_rates = []
    ranks = []
    
    for rank in rank_order:
        rank_df = df[df['player_rank_tier'] == rank]
        if len(rank_df) > 0:
            win_rate = rank_df['blue_team_win'].mean() * 100
            win_rates.append(win_rate)
            ranks.append(rank)
    
    colors = ['green' if wr > 50 else 'red' for wr in win_rates]
    bars = plt.bar(ranks, win_rates, color=colors, alpha=0.7)
    
    plt.axhline(y=50, color='black', linestyle='--', linewidth=2, label='50% (balanced)')
    plt.title('blue team win rate by player rank', fontsize=16, fontweight='bold')
    plt.xlabel('rank', fontsize=12)
    plt.ylabel('win rate (%)', fontsize=12) # of course its %
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.legend()
    
    # percentages
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('charts/winrate_by_rank.png', dpi=300, bbox_inches='tight')
    print("saved: charts/winrate_by_rank.png")
    plt.close()

def visualize_champion_popularity(df):
    # show most popular champions, idk why if i cannot show this during presentation, since nobody cares about it
    # collect all champions from all matches (200k+ :)) )
    all_champs = []
    for i in range(1, 6):
        all_champs.extend(df[f'blue_champ_{i}'].tolist())
        all_champs.extend(df[f'red_champ_{i}'].tolist())
    
    champ_counts = Counter(all_champs)
    top_20 = champ_counts.most_common(20)
    
    champ_ids, counts = zip(*top_20)
    champ_labels = [CHAMPION_NAMES.get(int(cid), f'Champion {int(cid)}') for cid in champ_ids]
    
    plt.figure(figsize=(14, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, 20))
    bars = plt.barh(range(20), counts, color=colors)
    plt.yticks(range(20), champ_labels)
    plt.xlabel('pick count', fontsize=12)
    plt.title('top 20 most popular champions', fontsize=16, fontweight='bold')
    # where is my main taliyah???????????
    plt.gca().invert_yaxis()
    
    # graphing
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(count, i, f' {count}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('charts/champion_popularity.png', dpi=300, bbox_inches='tight')
    print("saved: charts/champion_popularity.png")
    plt.close()

def visualize_class_distribution(df):
    # champion class pie chart (i get hungry when i see word pie)
    all_classes = []
    for i in range(1, 6):
        all_classes.extend(df[f'blue_class_{i}'].dropna().tolist())
        all_classes.extend(df[f'red_class_{i}'].dropna().tolist())
    
    class_counts = Counter(all_classes)
    
    # sort by cunt for better visualization
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_classes)
    
    plt.figure(figsize=(12, 8))
    
    # colour coding the pie chart
    colour_map = {
        'Fighter': '#E40303',
        'Marksman': '#FF8C00',
        'Mage': '#FFED00',
        'Tank': '#008026',
        'Assassin': '#24408E',
        'Support': '#732982',
        'Enchanter': '#F5A9B8',
        'Unknown': '#5BCEFA'
    }
    colors = [colour_map.get(cls, '#95A5A6') for cls in classes]
    
    # create pie chart with better formatting
    wedges, texts, autotexts = plt.pie(counts, labels=classes, autopct='%1.1f%%', 
                                        startangle=90, colors=colors,
                                        textprops={'fontsize': 11})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.title('champion class distribution', fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('charts/class_distribution.png', dpi=300, bbox_inches='tight')
    print("saved: charts/class_distribution.png")
    plt.close()

def visualize_champion_winrates(df):
    # displaying best winrate champions
    # calculate win rates for all champions
    champ_stats = {}
    
    for i in range(1, 6):
        for idx, row in df.iterrows():
            blue_champ = int(row[f'blue_champ_{i}'])
            red_champ = int(row[f'red_champ_{i}'])
            blue_win = row['blue_team_win']
            
            # blue team champions
            if blue_champ not in champ_stats:
                champ_stats[blue_champ] = {'wins': 0, 'games': 0}
            champ_stats[blue_champ]['games'] += 1
            champ_stats[blue_champ]['wins'] += blue_win
            
            # red team champions 
            if red_champ not in champ_stats:
                champ_stats[red_champ] = {'wins': 0, 'games': 0}
            champ_stats[red_champ]['games'] += 1
            champ_stats[red_champ]['wins'] += (1 - blue_win)
    
    # filter champions with at least 10 games (all should have 10+, since our dataset is splendid)
    champ_winrates = []
    for champ_id, stats in champ_stats.items():
        if stats['games'] >= 10:
            winrate = (stats['wins'] / stats['games']) * 100
            champ_winrates.append((champ_id, winrate, stats['games']))
    
    # sorting
    champ_winrates.sort(key=lambda x: x[1], reverse=True)
    
    # top 15 
    top_15 = champ_winrates[:15] if len(champ_winrates) >= 15 else champ_winrates

    plt.figure(figsize=(14, 8))
    champ_ids, winrates, games = zip(*top_15)
    labels = [f"{CHAMPION_NAMES.get(cid, f'champion {cid}')} ({g} games)" 
              for cid, g in zip(champ_ids, games)]
    
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_15)))
    bars = plt.barh(range(len(top_15)), winrates, color=colors)
    plt.yticks(range(len(top_15)), labels)
    plt.xlabel('Win Rate (%)', fontsize=12)
    plt.title(f'Top {len(top_15)} champions by win rate (min 10 games)', 
              fontsize=16, fontweight='bold')
    plt.xlim(0, 100)
    plt.gca().invert_yaxis()
    
    for i, (bar, wr) in enumerate(zip(bars, winrates)):
        plt.text(wr, i, f' {wr:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('charts/champion_winrates_top.png', dpi=300, bbox_inches='tight')
    print("saved: charts/champion_winrates_top.png")
    plt.close()

def visualize_game_stats_distribution(df):
    # some other statistics, only for the nerdiest nerds
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    stats = [
        ('gold_diff_15', 'gold difference @ 15min', 'gold'),
        ('kill_diff_15', 'kill difference @ 15min', 'red'),
        ('cs_diff_15', 'cd difference @ 15min', 'blue'),
        ('tower_diff_15', 'tower difference @ 15min', 'green'),
        ('dragon_diff_15', 'dragon difference @ 15min', 'purple'),
        ('grub_diff_15', 'grub difference @ 15min', 'orange')
    ]
    
    # differentials, copy paste from preprocessing.py
    df['gold_diff_15'] = df['blue_gold_15'] - df['red_gold_15']
    df['kill_diff_15'] = df['blue_kills_15'] - df['red_kills_15']
    df['cs_diff_15'] = df['blue_cs_15'] - df['red_cs_15']
    df['tower_diff_15'] = df['blue_towers_15'] - df['red_towers_15']
    df['dragon_diff_15'] = df['blue_dragons_15'] - df['red_dragons_15']
    df['grub_diff_15'] = df['blue_grubs_15'] - df['red_grubs_15']
    
    for idx, (stat, title, color) in enumerate(stats):
        ax = axes[idx // 3, idx % 3]
        
        # histogram
        ax.hist(df[stat], bins=30, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Even')
        ax.set_xlabel('Difference (Blue - Red)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('charts/game_stats_distribution.png', dpi=300, bbox_inches='tight')
    # https://open.spotify.com/track/5N3hjp1WNayUPZrA8kJmJP, give us 5 for the assigment
    print("saved: charts/game_stats_distribution.png")
    plt.close()

def visualize_queue_distribution(df):
    # soloq vs flexq p-word chart
    queue_counts = df['queue_id'].value_counts()
    
    plt.figure(figsize=(8, 8))
    labels = ['soloq (420)' if q == 420 else 'flex (440)' for q in queue_counts.index]
    colors = ['#3498db', '#e74c3c']
    
    plt.pie(queue_counts.values, labels=labels, autopct='%1.1f%%', 
            startangle=90, colors=colors, textprops={'fontsize': 12})
    plt.title('queue type distribution', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/queue_distribution.png', dpi=300, bbox_inches='tight')
    print("saved: charts/queue_distribution.png")
    plt.close()

def visualize_early_lead_vs_outcome(df):
    # how gold affect win rate (proves that money indeed brings happiness)
    df['gold_diff_15'] = df['blue_gold_15'] - df['red_gold_15']
    
    plt.figure(figsize=(12, 7))
    
    # separate by outcome
    wins = df[df['blue_team_win'] == 1]['gold_diff_15']
    losses = df[df['blue_team_win'] == 0]['gold_diff_15']
    
    plt.hist([wins, losses], bins=40, label=['blue win', 'blue loss'], 
             color=['green', 'red'], alpha=0.6, edgecolor='black')

    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='even gold')
    plt.xlabel('gold difference @ 15min (blue - red)', fontsize=12)
    plt.ylabel('frequency', fontsize=12)
    plt.title('gold lead @ 15min vs match outcome', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('charts/gold_lead_vs_outcome.png', dpi=300, bbox_inches='tight')
    print("saved: charts/gold_lead_vs_outcome.png")
    plt.close()

def generate_summary_stats(df):
    # give some stats and go home
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"total matches: {len(df)}")
    print(f"\nqueue distribution:")
    print(f"  soloq (420): {len(df[df['queue_id'] == 420])} ({len(df[df['queue_id'] == 420])/len(df)*100:.1f}%)")
    print(f"  flex (440): {len(df[df['queue_id'] == 440])} ({len(df[df['queue_id'] == 440])/len(df)*100:.1f}%)")

    print(f"\nrank distribution:")
    for rank in df['player_rank_tier'].value_counts().index:
        count = len(df[df['player_rank_tier'] == rank])
        print(f"  {rank}: {count} matches ({count/len(df)*100:.1f}%)")

    print(f"\noverall win rate (blue team): {df['blue_team_win'].mean()*100:.2f}%")
    print(f"\naverage stats @ 15min:")
    print(f"  gold (blue): {df['blue_gold_15'].mean():.0f}")
    print(f"  kills (blue): {df['blue_kills_15'].mean():.1f}")
    print(f"  cs (blue): {df['blue_cs_15'].mean():.0f}")
    print(f"  towers (blue): {df['blue_towers_15'].mean():.2f}")
    print(f"  dragons (blue): {df['blue_dragons_15'].mean():.2f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    print("loading data...")
    df = load_data()
    
    print("\ngenerating visualizations...")
    generate_summary_stats(df)
    
    visualize_rank_distribution(df)
    visualize_win_rate_by_rank(df)
    visualize_queue_distribution(df)
    visualize_champion_popularity(df)
    visualize_class_distribution(df)
    visualize_champion_winrates(df)
    visualize_game_stats_distribution(df)
    visualize_early_lead_vs_outcome(df)
    
    print("\nall visualizations generated successfully!")
    print("generated files:")
    print("  - rank_distribution.png")
    print("  - winrate_by_rank.png")
    print("  - queue_distribution.png")
    print("  - champion_popularity.png")
    print("  - class_distribution.png")
    print("  - champion_winrates_top.png")
    print("  - game_stats_distribution.png")
    print("  - gold_lead_vs_outcome.png")
