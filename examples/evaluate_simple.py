import os
import argparse
from rlcard.agents.cfr_agent import CFRAgent
import datetime

import rlcard
from rlcard import models
from rlcard.utils import  tournament, Logger

def testModel(args):
    env = rlcard.make('leduc-holdem')
    default_model = models.load('leduc-holdem-cfr').agents[0]
    cfr_agent = CFRAgent(env, os.path.join(args.model_path, args.model_name))
    cfr_agent.load()
    env.set_agents([default_model, cfr_agent])
    with Logger(args.log_dir) as logger:
        logger.log_performance(env.timestep, tournament(env, args.num_eval_games)[0])


if __name__ == '__main__':
    today_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser("CFR example in RLCard")
    parser.add_argument('--num_eval_games', type=int, default=1000)
    parser.add_argument('--model_path', type=str, default='../experiments/leduc_holdem_cfr_result/')
    parser.add_argument('--model_name', type=str, default='cfr_model')
    parser.add_argument('--log_dir', type=str, default='experiments/leduc_holdem_compare_result/{}'.format(today_date))

    args = parser.parse_args()

    testModel(args)
