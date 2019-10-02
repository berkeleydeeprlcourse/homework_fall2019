
1) See hw1 if you'd like to see installation instructions. You do NOT have to redo them. But, you need to install OpenCV for this assignment:
`pip install opencv-python==3.4.0.12`

You also need to replace `<pathtogym>/gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file. To find the file:
$ locate lunar_lander.py
(or if there are multiple options there):
$ source activate cs285_env
$ ipython
$ import gym
$ gym.__file__
<pathtogym>/gym/__init__.py
##############################################
##############################################


2) Code:

-------------------------------------------

Files to look at, even though there are no explicit 'TODO' markings:
- scripts/run_hw3_dqn.py

-------------------------------------------

Blanks to be filled in by using your code from hw1 are marked with 'TODO: GETTHIS from HW1'

The following files have these:
- infrastructure/rl_trainer.py
- infrastructure/utils.py
- policies/MLP_policy.py

Blanks to be filled in by using your code from hw2 are marked with 'TODO: GETTHIS from HW2'

- infrastructure/utils.py
- policies/MLP_policy.py

-------------------------------------------

Blanks to be filled in now (for this assignment) are marked with 'TODO'

The following files have these:
- critics/dqn_critic.py
- agents/dqn_agent.py
- policies/argmax_policy.py

- critics/bootstrapped_continuous_critic.py
- agents/ac_agent.py

##############################################
##############################################


3) Run code with the following command: 

$ python cs285/scripts/run_hw3_dqn.py --env_name PongNoFrameskip-v4 --exp_name test_pong
$ python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name 100_1 -ntu 100 -ngsptu 1

Flags of relevance, when running the commands above (see pdf for more info):
-double_q Whether to use double Q learning or not.

##############################################


4) Visualize saved tensorboard event file:

$ cd cs285/data/<your_log_dir>
$ tensorboard --logdir .

Then, navigate to shown url to see scalar summaries as plots (in 'scalar' tab), as well as videos (in 'images' tab)
