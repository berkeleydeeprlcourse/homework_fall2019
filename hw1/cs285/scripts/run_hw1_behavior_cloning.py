import os
import time
import numpy as np
import tensorflow as tf

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.bc_agent import BCAgent
from cs285.policies.loaded_gaussian_policy import Loaded_Gaussian_Policy


class BC_Trainer(object):

    def __init__(self, params):

        #######################
        # AGENT PARAMS
        #######################

        agent_params = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'max_replay_buffer_size': params['max_replay_buffer_size'],
        }

        self.params = params
        # TODO: look in here and implement this
        self.params['agent_class'] = BCAgent
        self.params['agent_params'] = agent_params

        ################
        # RL TRAINER
        ################

        # TODO: look in here and implement this
        self.rl_trainer = RL_Trainer(self.params)

        #######################
        # LOAD EXPERT POLICY
        #######################

        print('Loading expert policy from...',
              self.params['expert_policy_file'])
        self.loaded_expert_policy = Loaded_Gaussian_Policy(
            self.rl_trainer.sess, self.params['expert_policy_file'])
        print('Done restoring expert policy...')

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            n_iter=self.params['n_iter'],
            initial_expertdata=self.params['expert_data'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
            relabel_with_expert=self.params['do_dagger'],
            expert_policy=self.loaded_expert_policy,
        )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # relative to where you're running this script from
    parser.add_argument('--expert_policy_file', '-epf',
                        type=str, required=True)
    # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=True)
    parser.add_argument('--env_name', '-env', type=str,
                        help='choices: Ant-v2, Humanoid-v2, Walker-v2, HalfCheetah-v2, Hopper-v2', required=True)
    parser.add_argument('--exp_name', '-exp', type=str,
                        default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int)

    # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--num_agent_train_steps_per_iter',
                        type=int, default=1000)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    # training data collected (in the env) during each iteration
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--eval_batch_size', type=int,
                        default=200)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    # depth, of policy to be learned
    parser.add_argument('--n_layers', type=int, default=2)
    # width of each layer, of policy to be learned
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=5e-3)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    ##################################
    # CREATE DIRECTORY FOR LOGGING
    ##################################

    logdir_prefix = 'bc_'
    if args.do_dagger:
        logdir_prefix = 'dagger_'
        assert args.n_iter > 1, (
            'DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        assert args.n_iter == 1, (
            'Vanilla behavior cloning collects expert data just once (n_iter=1)')

    # directory for logging
    data_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + \
        args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    # RUN TRAINING
    ###################

    trainer = BC_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
