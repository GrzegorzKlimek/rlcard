import argparse
import os

import rlcard
from rlcard import models
from rlcard.agents import RandomAgent, CFRAgent
from rlcard.utils import tournament, Logger


def compare_models(env, agent1, agent2, num_eval_games, log_dir):
    env.set_agents([agent1,agent2])
    tournament_result = tournament(env, num_eval_games)
    with Logger(log_dir) as logger:
        logger.log( "Result are model1: {} vs  model2: {}".
                   format(tournament_result[0], tournament_result[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compare different agents")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_eval_games', type=int, default=2000)
    parser.add_argument('--model_name', type=str, default="cfr_model")
    parser.add_argument('--log_dir', type=str, default='experiments/leduc_holdem_cfr_results/')

    args = parser.parse_args()
    env = rlcard.make('leduc-holdem', config={'seed': 0, 'allow_step_back':True})
    agent1 = CFRAgent(env, os.path.join(args.log_dir, args.model_name))
    agent1.load()
    agent2 = models.load('leduc-holdem-cfr').agents[0]
    agent2.load()
    agent3 = RandomAgent(num_actions=env.num_actions)

    compare_models(env,agent1, agent2, args.num_eval_games, args.log_dir)
