import os
from collections import namedtuple

from rlcard.agents import CFRAgent
import json

import rlcard
from rlcard import models
from rlcard.utils import tournament, Logger, set_seed
import time
import datetime


class TrainConfig:
    def __init__(self, seed, num_episodes, num_eval_games, checkpoint, model_name, log_dir):
        self.seed = seed
        self.num_episodes = num_episodes
        self.num_eval_games = num_eval_games
        self.checkpoint = checkpoint
        self.model_name = model_name
        self.log_dir = log_dir

def train(trainConfig: TrainConfig):
    start = time.time()
    env = rlcard.make('leduc-holdem', config={'seed': 0, 'allow_step_back':True})
    eval_env = rlcard.make('leduc-holdem', config={'seed': 0})
    # Seed numpy, torch, random
    set_seed(trainConfig.seed)
    # Initilize CFR Agent
    agent = CFRAgent(env, os.path.join(trainConfig.log_dir, trainConfig.model_name))
    agent.load()  # If we have saved model, we first load the model

    # Evaluate my model CFR against pretrained model
    eval_env.set_agents([agent, models.load('leduc-holdem-cfr').agents[0]])

    # Start training
    with Logger(trainConfig.log_dir) as logger:
        for episode in range(trainConfig.num_episodes):
            agent.train()
            if episode % trainConfig.checkpoint == 0:
                agent.save() # Save model
        tournament_result = tournament(eval_env, trainConfig.num_eval_games)
        end = time.time()
        duration = str(datetime.timedelta(seconds=(end-start)))
        logger.log("For the config {}: \n"
                   "Result are model: {} vs pretrained model: {} \n"
                   "Duration of training is {}".
                   format(trainConfig, tournament_result[0], tournament_result[1], duration))


if __name__ == '__main__':
    with open('run_configs/runs_1.json') as f:
        trainConfigs = json.load(f)
        for configDict in trainConfigs:
            config = namedtuple("TrainConfig", configDict.keys())(*configDict.values())
            train(config)




