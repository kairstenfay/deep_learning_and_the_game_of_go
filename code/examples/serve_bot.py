# Include the path to the local version of dlgo.
import os
import sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, '..'))

# tag::e2e_imports[]
import h5py

# from dlgo import goboard_fast as goboard
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.httpfrontend import get_web_app
# end::e2e_imports[]

# tag::e2e_load_agent[]
model_file = h5py.File("../agents/deep_bot.h5", "r")
bot_from_file = load_prediction_agent(model_file)
model_file.close()

"""
_, _, size = bot_from_file.encoder.shape()
board_size = size, size
game_state = goboard.GameState.new_game(board_size)
next_move = goboard.Move.play(goboard.Point(4, 4))
game_state = game_state.apply_move(next_move)

s = bot_from_file.select_move(game_state)
print(s)
"""

web_app = get_web_app({'predict': bot_from_file})
web_app.run()
# end::e2e_load_agent[]