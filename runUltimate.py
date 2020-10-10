import sys, os
import numpy as np
sys.path.insert(0, 'evoman')
from demo_controller import player_controller
from environment import Environment
from task1_constants import *

ultimate_file = open("Logs/Task1/UltimateChampion/UltimateChampion.txt", "r")

ultimateString = ultimate_file.read()
ultimateString = ultimateString.replace("[", "")
ultimateString = ultimateString.replace("]", "")
ultimate = np.fromstring(ultimateString, dtype=float, sep=' ')

print(ultimateString)
print(ultimate)

experiment_name = 'ultimateRun'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name, level=2,
                               player_controller=player_controller(N_HIDDEN_NEURONS),
                               enemies=[1],speed="normal")

enemies = [1,2,3,4,5,6,7,8]

for enemy in enemies:
    env.update_parameter('enemies', [enemy])
    env.play(ultimate)
