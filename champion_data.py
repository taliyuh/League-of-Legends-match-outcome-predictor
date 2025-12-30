"""
Champion class and metadata mappings
Data fetched from Riot Data Dragon API
Last updated: 2025-12-29
"""

# Champion ID to primary class mapping
CHAMPION_CLASSES = {
    1: 'Mage',            # Annie
    2: 'Fighter',         # Olaf
    3: 'Tank',            # Galio
    4: 'Mage',            # Twisted Fate
    5: 'Fighter',         # Xin Zhao
    6: 'Fighter',         # Urgot
    7: 'Assassin',        # LeBlanc
    8: 'Mage',            # Vladimir
    9: 'Mage',            # Fiddlesticks
    10: 'Mage',           # Kayle
    11: 'Fighter',        # Master Yi
    12: 'Tank',           # Alistar
    13: 'Mage',           # Ryze
    14: 'Tank',           # Sion
    15: 'Marksman',       # Sivir
    16: 'Enchanter',        # Soraka
    17: 'Marksman',       # Teemo
    18: 'Marksman',       # Tristana
    19: 'Fighter',        # Warwick
    20: 'Tank',           # Nunu & Willump
    21: 'Marksman',       # Miss Fortune
    22: 'Marksman',       # Ashe
    23: 'Fighter',        # Tryndamere
    24: 'Fighter',        # Jax
    25: 'Mage',           # Morgana
    26: 'Mage',          # Zilean
    27: 'Tank',           # Singed
    28: 'Assassin',       # Evelynn
    29: 'Marksman',       # Twitch
    30: 'Mage',           # Karthus
    31: 'Tank',           # Cho'Gath
    32: 'Tank',           # Amumu
    33: 'Tank',           # Rammus
    34: 'Mage',           # Anivia
    35: 'Assassin',       # Shaco
    36: 'Tank',           # Dr. Mundo
    37: 'Enchanter',         # Sona
    38: 'Assassin',       # Kassadin
    39: 'Fighter',        # Irelia
    40: 'Enchanter',         # Janna
    41: 'Fighter',        # Gangplank
    42: 'Marksman',       # Corki
    43: 'Mage',           # Karma
    44: 'Tank',           # Taric
    45: 'Mage',           # Veigar
    48: 'Fighter',        # Trundle
    50: 'Mage',           # Swain
    51: 'Marksman',       # Caitlyn
    53: 'Tank',           # Blitzcrank
    54: 'Tank',           # Malphite
    55: 'Assassin',       # Katarina
    56: 'Fighter',        # Nocturne
    57: 'Tank',           # Maokai
    58: 'Fighter',        # Renekton
    59: 'Fighter',        # Jarvan IV
    60: 'Assassin',       # Elise
    61: 'Mage',           # Orianna
    62: 'Fighter',        # Wukong
    63: 'Mage',           # Brand
    64: 'Fighter',        # Lee Sin
    67: 'Marksman',       # Vayne
    68: 'Fighter',        # Rumble
    69: 'Mage',           # Cassiopeia
    72: 'Tank',           # Skarner
    74: 'Mage',           # Heimerdinger
    75: 'Fighter',        # Nasus
    76: 'Assassin',       # Nidalee
    77: 'Fighter',        # Udyr
    78: 'Tank',           # Poppy
    79: 'Fighter',        # Gragas
    80: 'Fighter',        # Pantheon
    81: 'Marksman',       # Ezreal
    82: 'Fighter',        # Mordekaiser
    83: 'Fighter',        # Yorick
    84: 'Assassin',       # Akali
    85: 'Mage',           # Kennen
    86: 'Fighter',        # Garen
    89: 'Tank',           # Leona
    90: 'Mage',           # Malzahar
    91: 'Assassin',       # Talon
    92: 'Fighter',        # Riven
    96: 'Marksman',       # Kog'Maw
    98: 'Tank',           # Shen
    99: 'Mage',           # Lux
    101: 'Mage',          # Xerath
    102: 'Fighter',       # Shyvana
    103: 'Mage',          # Ahri
    104: 'Marksman',      # Graves
    105: 'Assassin',      # Fizz
    106: 'Fighter',       # Volibear
    107: 'Assassin',      # Rengar
    110: 'Marksman',      # Varus
    111: 'Tank',          # Nautilus
    112: 'Mage',          # Viktor
    113: 'Tank',          # Sejuani
    114: 'Fighter',       # Fiora
    115: 'Mage',          # Ziggs
    117: 'Enchanter',        # Lulu
    119: 'Marksman',      # Draven
    120: 'Fighter',       # Hecarim
    121: 'Assassin',      # Kha'Zix
    122: 'Fighter',       # Darius
    126: 'Fighter',       # Jayce
    127: 'Mage',          # Lissandra
    131: 'Fighter',       # Diana
    133: 'Marksman',      # Quinn
    134: 'Mage',          # Syndra
    136: 'Mage',          # Aurelion Sol
    141: 'Fighter',       # Kayn
    142: 'Mage',          # Zoe
    143: 'Mage',          # Zyra
    145: 'Marksman',      # Kai'Sa
    147: 'Enchanter',       # Seraphine
    150: 'Fighter',       # Gnar
    154: 'Tank',          # Zac
    157: 'Fighter',       # Yasuo
    161: 'Mage',          # Vel'Koz
    163: 'Mage',          # Taliyah
    164: 'Fighter',       # Camille
    166: 'Marksman',      # Akshan
    200: 'Fighter',       # Bel'Veth
    201: 'Tank',          # Braum
    202: 'Marksman',      # Jhin
    203: 'Marksman',      # Kindred
    221: 'Marksman',      # Zeri
    222: 'Marksman',      # Jinx
    223: 'Tank',          # Tahm Kench
    233: 'Fighter',       # Briar
    234: 'Fighter',       # Viego
    235: 'Marksman',      # Senna
    236: 'Marksman',      # Lucian
    238: 'Assassin',      # Zed
    240: 'Fighter',       # Kled
    245: 'Assassin',      # Ekko
    246: 'Assassin',      # Qiyana
    254: 'Fighter',       # Vi
    266: 'Fighter',       # Aatrox
    267: 'Enchanter',        # Nami
    268: 'Mage',          # Azir
    350: 'Enchanter',        # Yuumi
    360: 'Marksman',      # Samira
    412: 'Tank',          # Thresh
    420: 'Fighter',       # Illaoi
    421: 'Fighter',       # Rek'Sai
    427: 'Enchanter',        # Ivern
    429: 'Marksman',      # Kalista
    432: 'Tank',          # Bard
    497: 'Enchanter',        # Rakan
    498: 'Marksman',      # Xayah
    516: 'Tank',          # Ornn
    517: 'Mage',          # Sylas
    518: 'Mage',          # Neeko
    523: 'Marksman',      # Aphelios
    526: 'Tank',          # Rell
    555: 'Assassin',       # Pyke
    711: 'Mage',          # Vex
    777: 'Fighter',       # Yone
    799: 'Fighter',       # Ambessa
    800: 'Mage',          # Mel
    804: 'Marksman',      # Yunara
    875: 'Fighter',       # Sett
    876: 'Fighter',       # Lillia
    887: 'Fighter',       # Gwen
    888: 'Enchanter',        # Renata Glasc
    893: 'Mage',          # Aurora
    895: 'Fighter',       # Nilah
    897: 'Tank',          # K'Sante
    901: 'Marksman',      # Smolder
    902: 'Enchanter',        # Milio
    904: 'Fighter',       # Zaahen
    910: 'Mage',          # Hwei
    950: 'Assassin',      # Naafiri
}

def get_champion_class(champion_id):
    """Get primary class for a champion ID"""
    return CHAMPION_CLASSES.get(champion_id, 'Unknown')

def get_class_counts(champ_ids):
    """Count occurrences of each class in a list of champion IDs"""
    from collections import Counter
    classes = [get_champion_class(cid) for cid in champ_ids]
    return Counter(classes)
