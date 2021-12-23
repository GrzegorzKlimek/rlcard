import os
import argparse
from rlcard.agents import CFRAgent, RandomAgent
import datetime

import rlcard
from rlcard import models
from rlcard.utils import  tournament, Logger

def testModel(args):
    env = rlcard.make('leduc-holdem')
    # comparison_model = RandomAgent(num_actions=env.num_actions)
    comparison_agent = models.load('leduc-holdem-cfr').agents[0]
    tested_agent = CFRAgent(env, os.path.join(args.model_path, args.model_name))
    tested_agent.load()
    env.set_agents([ comparison_agent, tested_agent])
    with Logger(args.log_dir) as logger:
        rewards = tournament(env, args.num_eval_games)
        print(args.first_agent_name)
        logger.log_performance(env.timestep, rewards[0])
        print(args.second_agent_name)
        logger.log_performance(env.timestep, rewards[1])


if __name__ == '__main__':
    today_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser("CFR example in RLCard")
    parser.add_argument('--num_eval_games', type=int, default=1000)
    parser.add_argument('--model_path', type=str, default='experiments/leduc_holdem_cfr_result/')
    parser.add_argument('--model_name', type=str, default='cfr_model')
    parser.add_argument('--log_dir', type=str, default='experiments/leduc_holdem_compare_result/{}'.format(today_date))
    parser.add_argument('--first_agent_name', type=str, default='random agent')
    parser.add_argument('--second_agent_name', type=str, default='trained cfr agent')

    args = parser.parse_args()

    testModel(args)
