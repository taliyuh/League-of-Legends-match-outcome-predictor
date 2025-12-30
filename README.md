# League of Legends Match Prediction

## Aim
The aim of this project is to predict the outcome of League of Legends matches at the 15-minute mark, based on statistics like player kills, gold, experience, collected objectives or destroyed turrets, using gradient boosting algorithms.

## Setup
It is highly advised to run the scripts inside a virtual environment. Remember to install all the dependencies mentioned in `requirements.txt`.

## File Breakdown

- **`.env`** - Store RIOT API key in format `RIOT_API_KEY=&lt;key&gt;`. Key should be obtained individually at https://developer.riotgames.com/.
- **`app.py`** - Complete pipeline of collecting the data, preprocessing it, and training the model.
- **`champion_data.py`** - Maps champion IDs to their class.
- **`data.py`** - Appends data to the dataset, based on players listed in `players.py`.
- **`match_data.csv`** - Dataset of matches. Due to Riot API rules and regulations, it is not permitted to share the data online. Example data is available there.
- **`model.py`** - Training and evaluating the model.
- **`multi_models.py`** - Training and evaluating models for all rank/queue combinations.
- **`multi_preprocessing.py`** - Preprocessing models for all rank/queue combinations.
- **`players.py`** - Contains list of players for `data.py` to refine match data for. The list is not included, as it would violate Riot rules and regulations.
- **`preprocessing.py`** - Preprocesses the data from `match_data.csv`.
- **`requirements.txt`** - Project dependencies.
- **`visualise_data.py`** - Visualises data from `match_data.csv` in form of graphs and charts.
- **`visualize_models.py`** - Visualises performance of obtained models.

## Recommended Pipeline

1. Obtain dataset.
2. Analyse it via `visualise_data.py`.
3. Run `multi_preprocessing.py`.
4. Run `multi_models.py`.
5. Analyse obtained models via `visualize_models.py`.

## Results
Our database contains over 22000 unique matches from approximetely 400 players across different ranks. Visualisation of database and performances of the models is available in the `/charts` folder. Obtained models are available in the `/models` folder.
