### Repo for hacking around re: the 'positivity bias'/asymmetric LR in CL-RL

RL in daily life often involves some form of categorisation, as we must learn not just option-outcome relationships (like in classic bandit tasks), but also the features that predict good vs. bad outcomes (i.e., hybrid category-learning-RL). Past work shows that a stubborn ‘learning trap’ arises in such situations – incorrect beliefs that persist due to discouraging the exploration of options that would offer corrective feedback; here, people learn incomplete rules based on fewer dimensions than is actually relevant for predicting outcomes, which result in the avoidance of potentially rewarding options and therefore suboptimal rewards. This proposal asks whether asymmetric learning rates (‘positivity bias’) – a phenomenon past work has found in human RL through the use of simple bandit tasks, and which has been shown to be adaptive for maximising rewards in these simple scenarios – could also exist in hybrid category-learning-RL problems, and how/whether asymmetric learning rates could be adaptive for gaining rewards. Findings would further our understanding of a commonly encountered class of problems, which sits at the intersection of category learning and RL and has implications for the formation of false/incomplete beliefs including stereotype formation.

#### File descriptions

- Existing implementations (without asymmetric learning rates): `fn_ALCOVE_RL.R` contains functions that are then called by `ALCOVE-RL-master.R`
    - then the script `sim-alcove-sequential.R` does some n_sim simulations based on `ALCOVE-RL-master.R`
    - `sim-alcove-parallel.R` also does simulations but with parallel processing (much faster)

- Modified implementation (with asymmetric learning rates):
  - `alcove-rl-asym.R` contains modified functions from `fn_ALCOVE_RL.R` and `ALCOVE-RL-master.R` (now all in one script)
  - `sim-alcove-asymmetric-sandbox.R` does simulations using `alcove-rl-asym.R`; in parallel
