1) See hw1 if you'd like to see installation instructions. You do NOT have to redo them.

##############################################
##############################################


2) Code:

-------------------------------------------

Files to look at, even though there are no explicit 'TODO' markings:
- scripts/run_hw4_mb.py

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
- critics/mb_agent.py
- models/ff_model.py
- policies/MPC_policy.py
- infrastructure/utils.py

##############################################
##############################################


3) Commands: 

Please refer to the PDF for the specific commands needed for different questions. 

##############################################


4) Visualize saved tensorboard event file:

$ cd cs285/data/<your_log_dir>
$ tensorboard --logdir .

Then, navigate to shown url to see scalar summaries as plots (in 'scalar' tab), as well as videos (in 'images' tab)