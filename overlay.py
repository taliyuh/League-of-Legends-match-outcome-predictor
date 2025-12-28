import tkinter as tk
from automated_system import LoLAutomatedPredictor
import threading
import time

class LoLOverlay:
    def __init__(self):
        self.app = LoLAutomatedPredictor()
        
        # 1. Setup Tkinter Window
        self.root = tk.Tk()
        self.root.title("LoL Predictor Overlay")
        
        # Positioning: Top Right corner (change as needed)
        self.root.geometry("250x100+1600+50") 
        
        # Remove borders and make it stay on top
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.8) # 80% transparency
        self.root.configure(bg='black')

        # 2. UI Elements
        self.label_status = tk.Label(self.root, text="Waiting for game...", fg="white", bg="black", font=("Arial", 10))
        self.label_status.pack(pady=5)

        self.label_prediction = tk.Label(self.root, text="", fg="#00FF00", bg="black", font=("Arial", 14, "bold"))
        self.label_prediction.pack(pady=5)

        # 3. Start the background monitoring thread
        self.thread = threading.Thread(target=self.monitor_logic, daemon=True)
        self.thread.start()

        self.root.mainloop()

    def monitor_logic(self):
        """ This runs in the background so the GUI doesn't freeze """
        while True:
            data = self.app.get_live_data()
            if data:
                game_time = data['gameData']['gameTime']
                minutes = int(game_time // 60)
                seconds = int(game_time % 60)
                
                self.label_status.config(text=f"Game Time: {minutes:02d}:{seconds:02d}")

                # Phase 1: Training in lobby
                if game_time < 180 and not self.app.trained_for_lobby:
                    self.label_status.config(text="Status: Training Model...")
                    players = [p['summonerName'] for p in data['allPlayers']]
                    self.app.run_lobby_training(players)
                
                # Phase 2: Prediction at 15:00
                if self.app.prediction_made and self.app.last_prediction:
                    winner = self.app.last_prediction['winner']
                    conf = self.app.last_prediction['probability']
                    color = "#5dade2" if winner == "BLUE" else "#ec7063" # Blue or Red color
                    self.label_prediction.config(text=f"{winner} WIN ({conf:.1%})", fg=color)
                elif game_time >= 900 and not self.app.prediction_made:
                    self.app.make_live_prediction(data)
                    self.app.prediction_made = True

            else:
                self.label_status.config(text="Searching for League...")
                self.label_prediction.config(text="")
                self.app.trained_for_lobby = False
                self.app.prediction_made = False

            time.sleep(2)

if __name__ == "__main__":
    LoLOverlay()