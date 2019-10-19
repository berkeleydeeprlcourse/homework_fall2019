import tensorflow as tf
import numpy as np

from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.critics.dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, sess, env, agent_params):

        self.env = env
        self.sess = sess
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(sess, agent_params, self.optimizer_spec)
        self.actor = ArgMaxPolicy(sess, self.critic)

        lander = agent_params['env_name'] == 'LunarLander-v2'
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):

        """
            Step the env and store the transition

            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.

            Note that self.last_obs must always point to the new latest observation.
        """

        # TODO store the latest observation into the replay buffer
        # HINT: see replay buffer's function store_frame
        self.replay_buffer_idx = TODO

        eps = self.exploration.value(self.t)
        # TODO use epsilon greedy exploration when selecting action
        # HINT: take random action 
            # with probability eps (see np.random.random())
            # OR if your current step number (see self.t) is less that self.learning_starts
        perform_random_action = TODO

        if perform_random_action:
            action = TODO
        else:
            # TODO query the policy to select action
            # HINT: you cannot use "self.last_obs" directly as input
            # into your network, since it needs to be processed to include context
            # from previous frames. 
            # Check out the replay buffer, which has a function called
            # encode_recent_observation that will take the latest observation
            # that you pushed into the buffer and compute the corresponding
            # input that should be given to a Q network by appending some
            # previous frames.
            enc_last_obs = TODO
            enc_last_obs = enc_last_obs[None, :]

            # TODO query the policy with enc_last_obs to select action
            action = TODO
            action = action[0]

        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)
        TODO

        # TODO store the result of taking this action into the replay buffer
        # HINT1: see replay buffer's store_effect function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        TODO

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        TODO

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        """
            Here, you should train the DQN agent.
            This consists of training the critic, as well as periodically updating the target network.
        """

        loss = 0.0
        if (self.t > self.learning_starts and \
                self.t % self.learning_freq == 0 and \
                self.replay_buffer.can_sample(self.batch_size)):

            # TODO populate all placeholders necessary for calculating the critic's total_error
            # HINT: obs_t_ph, act_t_ph, rew_t_ph, obs_tp1_ph, done_mask_ph
            # feed_dict = {
            #     self.critic.learning_rate: self.optimizer_spec.lr_schedule.value(self.t),
            #     TODO,
            #     TODO,
            #     TODO,
            #     TODO,
            #     TODO,
            # }

            # TODO: create a LIST of tensors to run in order to 
            # train the critic as well as get the resulting total_error
            tensors_to_run = TODO
            loss, _ = self.sess.run(tensors_to_run, feed_dict=feed_dict)
            # Note: remember that the critic's total_error value is what you
            # created to compute the Bellman error in a batch, 
            # and the critic's train function performs a gradient step 
            # and update the network parameters to reduce that total_error.

            # TODO: use sess.run to periodically update the critic's target function
            # HINT: see update_target_fn
            if self.num_param_updates % self.target_update_freq == 0:
                TODO

            self.num_param_updates += 1

        self.t += 1
        return loss
