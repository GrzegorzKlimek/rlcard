import os
from collections import namedtuple

from rlcard.agents import CFRAgent, RandomAgent
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

def compare_models(env, agent1, agent2, num_eval_games, log_dir):
    env.set_agents([agent1,agent2])
    tournament_result = tournament(env, num_eval_games)
    with Logger(log_dir) as logger:
        logger.log( "Result are model1: {} vs  model2: {}".
                    format(tournament_result[0], tournament_result[1]))

def train(eval_env, agent, eval_agent, trainConfig):
    start = time.time()
    eval_env.set_agents([agent, eval_agent])
    # Start training
    with Logger(trainConfig.log_dir) as logger:
        for episode in range(trainConfig.num_episodes):
            agent.train()
            if episode % trainConfig.checkpoint == 0:
                agent.save() # Save model
        end = time.time()
        duration = str(datetime.timedelta(seconds=(end-start)))
        logger.log("Training for config {}. Took: {} \n".format(trainConfig, duration))


if __name__ == '__main__':
    with open('run_configs/runs_1.json') as f:
        trainConfigs = json.load(f)
        for configDict in trainConfigs:
            config = namedtuple("TrainConfig", configDict.keys())(*configDict.values())
            # Seed numpy, torch, random
            set_seed(config.seed)
            env = rlcard.make('leduc-holdem', config={'seed': 0, 'allow_step_back':True})
            eval_env = rlcard.make('leduc-holdem', config={'seed': 0})
            eval_agent = models.load('leduc-holdem-cfr').agents[0]
            eval_agent.load()
            random_eval_agent = RandomAgent(num_actions=env.num_actions)
            # Initilize CFR Agent
            agent = CFRAgent(env, os.path.join(config.log_dir, config.model_name))
            # agent.load()  # If we have saved model, we first load the model
            # Evaluate my model CFR against pretrained model
            train(eval_env, agent,eval_agent,config)
            compare_models(env, agent, eval_agent, config.num_eval_games, config.log_dir)