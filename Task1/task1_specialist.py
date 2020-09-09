################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment

from task1_controller import player_controller
from task1_selection import selection
from task1_mutation import mutation


experiment_name = 'task1_specialist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# TODO: write class 'player_controller' in file 'task1_controller.py'


# initializes environment with ai player, playing against static enemy
def initialize(enemy=1,speed="fastest", n_hidden_neurons = 10):
    global env
    env = Environment(experiment_name=experiment_name,
                      level=2,
                      playermode="ai",
                      speed=speed,
                      enemymode="static",
                      enemies=[enemy],
                      player_controller=player_controller(n_hidden_neurons))

    env.play()


initialize(enemy=1, speed="normal")
