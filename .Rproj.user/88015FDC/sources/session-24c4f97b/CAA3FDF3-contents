# =============================================================================
# ALCOVE-RL ASYMMETRIC LEARNING RATE SIMULATION
# =============================================================================
# 
# Purpose: Systematically explore the parameter space of asymmetric learning rates
#          in an approach-avoidance task with AND-rule categorisation
#
# Task Structure:
# - 3 binary dimensions creating 8 possible stimuli
# - AND rule: Dangerous if dim1=1 AND dim2=1, otherwise Friendly
# - Approach friendly: +1 point
# - Approach dangerous: -3 points  
# - Avoid: 0 points
#
# Key question:
# How do different combinations of learning rates for gains (lambda_w_plus) 
# vs losses (lambda_w_minus) affect overall task performance?
# =============================================================================

# Load required libraries
source("alcove-rl-asym.R")
library(DEoptim) 
library(tidyverse)
library(parallel)      # For parallel processing
library(foreach)       # For parallel loops
library(doParallel)    # Backend for parallel processing

# =============================================================================
# SECTION 1: INITIALISE MODEL STRUCTURE
# =============================================================================

## Create all possible stimuli in the 3D binary space ####
# This creates a 2^3 = 8 row dataframe with all combinations
all_stimuli <- expand.grid(
  dim1 = c(0, 1), # First dimension (binary)
  dim2 = c(0, 1), # Second dimension (binary)
  dim3 = c(0, 1)  # Third dimension (binary)
)

## Apply the AND rule to assign true categories ####
# The AND rule makes only stimuli with dim1=1 AND dim2=1 dangerous
# This creates a non-linear category boundary requiring selective attention
# Note: dim3 is irrelevant to the category, testing dimension weighting
all_stimuli$true_category <- ifelse(
  all_stimuli$dim1 == 1 & all_stimuli$dim2 == 1,
  "dangerous",
  "friendly"
)

# Create exemplar matrix (all 8 possible stimuli)
# Matrix structure: dimensions (rows) × exemplars (columns)
# Each column represents one of the 8 possible stimuli
h <- matrix(c(
  0,0,0,  # Stimulus 1: [0,0,0]
  1,0,0,  # Stimulus 2: [1,0,0]
  0,1,0,  # Stimulus 3: [0,1,0]
  1,1,0,  # Stimulus 4: [1,1,0] - dangerous
  0,0,1,  # Stimulus 5: [0,0,1]
  1,0,1,  # Stimulus 6: [1,0,1]
  0,1,1,  # Stimulus 7: [0,1,1]
  1,1,1   # Stimulus 8: [1,1,1] - dangerous
), nrow = 3, ncol = 8, byrow = FALSE)

# Initialise association weights (all zeros) ####
# Matrix: categories (rows) × exemplars (columns)
# Row 1: weights for "friendly" category
# Row 2: weights for "dangerous" category
w_init <- matrix(0, nrow = 2, ncol = 8)  # 2 categories × 8 exemplars

# Initialise attention weights ####
# Equal attention to all dimensions initially
# Model will learn that dim3 is irrelevant
alpha_init <- c(0.33, 0.33, 0.34)  # or c(1/3, 1/3, 1/3)

# Experiment configuration vector ####
exp_info <- c(
  3,  # d_num: number of stimulus dimensions
  2,  # k_num: number of categories (friendly/dangerous)
  8,  # stim_num: number of exemplars (all possible stimuli)
  1   # t_type: teacher type (1=strict, 2=contingent, 3=humble)
)

# =============================================================================
# SECTION 2: UTILITY FUNCTIONS
# =============================================================================

# Function to calculate points earned based on action and true category ####
# This implements the task's reward structure:
# - Approaching friendly: +1 point (correct approach)
# - Approaching dangerous: -3 points (costly error)
# - Avoiding anything: 0 points (safe but no reward)
calculate_pt_outcome <- function(model_predictions) {
  model_predictions$data$pt_outcome <- case_when(
    model_predictions$action == 1 & model_predictions$data$cat2 == 1 ~ -3,
    model_predictions$action == 1 & model_predictions$data$cat1 == 1 ~ 1,
    model_predictions$action == 2 ~ 0,
    TRUE ~ 0
  )
  
  model_predictions$data$total_pt <- sum(model_predictions$data$pt_outcome)
  return(model_predictions)
}

# =============================================================================
# SECTION 3: PARAMETER SPACE SETUP
# =============================================================================

# Define parameter ranges to explore ####
# We systematically vary both learning rates to understand their interaction
# for the moment, we fix lambda_alpha
lambda_w_plus <- seq(0.05, 0.75, by = 0.05)      
lambda_w_minus <- seq(0.05, 0.75, by = 0.05) 
lambda_alpha <- 0.3
n_simulations <- 200

# Set up parameter grid
results_grid <- expand.grid(lambda_w_plus = lambda_w_plus, lambda_w_minus = lambda_w_minus)
results_grid$mean_pts_per_trial <- NA

# Fixed parameters
fixed_params <- c(2, 1)  # c = 2, phi = 1 (Luce choice rule)

# =============================================================================
# SECTION 4: PARALLEL PROCESSING SETUP
# =============================================================================

# Detect number of cores and use most of them
n_cores <- detectCores() - 2  # Leave one core free
cat("Using", n_cores, "cores for parallel processing\n")

# Register parallel backend
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Export necessary objects to worker nodes
clusterEvalQ(cl, {
  source("alcove-rl-asym.R")
  library(tidyverse)
})

clusterExport(cl, c("all_stimuli", "h", "w_init", "alpha_init", "exp_info", 
                    "calculate_pt_outcome", "fixed_params", "n_simulations"))

cat("Running", nrow(results_grid), "parameter combinations with", n_simulations, "simulations each...\n")

# =============================================================================
# SECTION 5: MAIN SIMULATION LOOP (PARALLEL)
# =============================================================================

system.time({
  results_list <- foreach(combo_idx = 1:nrow(results_grid), 
                          .combine = 'c',
                          .packages = c('tidyverse')) %dopar% {
                            
                            lambda_w_plus <- results_grid$lambda_w_plus[combo_idx]
                            lambda_w_minus <- results_grid$lambda_w_minus[combo_idx]
                            sim_params <- c(fixed_params, lambda_w_plus, lambda_w_minus, lambda_alpha)
                            
                            # Run simulations for current parameter combination
                            mean_pts_per_trial <- numeric(n_simulations)
                            
                            for (sim in 1:n_simulations) {
                              # Generate new trial data
                              set.seed(456 + combo_idx * n_simulations + sim)
                              n_trials <- 96
                              trial_indices <- sample(1:8, n_trials, replace = TRUE)
                              trial_stimuli <- all_stimuli[trial_indices, ]
                              
                              # Format for ALCOVE
                              tr_data <- data.frame(
                                dim1 = trial_stimuli$dim1,
                                dim2 = trial_stimuli$dim2,
                                dim3 = trial_stimuli$dim3,
                                cat1 = ifelse(trial_stimuli$true_category == "friendly", 1, 0),
                                cat2 = ifelse(trial_stimuli$true_category == "dangerous", 1, 0),
                                ctr = rep(0, n_trials)
                              )
                              
                              # Run model
                              model_predictions <- ALCOVE_RL(
                                intl_par = sim_params,
                                tr_data = tr_data[, 1:6],
                                h = h,
                                w = w_init,
                                alpha = alpha_init,
                                q = 1,
                                r = 2,
                                exp_info = exp_info
                              )
                              
                              # Determine actions from response probabilities ####
                              # Action 1 = approach (higher probability for cat1)
                              # Action 2 = avoid (higher probability for cat2)
                              model_predictions$action <- apply(model_predictions$pr, 1, which.max)
                              model_predictions$data <- tr_data
                              
                              # Calculate points earned ####
                              model_predictions <- calculate_pt_outcome(model_predictions)
                              pts_per_trial <- unique(model_predictions$data$total_pt)
                              mean_pts_per_trial[sim] <- pts_per_trial/n_trials
                            }
                            
                            # Return mean for this parameter combination
                            mean(mean_pts_per_trial)
                          }
})

# Stop the cluster
stopCluster(cl)

# Store results
results_grid$mean_pts_per_trial <- results_list

cat("Parallel processing completed!\n")

# =============================================================================
# SECTION 6: VISUALISATION
# =============================================================================

library(ggplot2)

# Create the heatmap
p <- ggplot(results_grid, aes(x = lambda_w_minus, y = lambda_w_plus, fill = mean_pts_per_trial)) +
  geom_tile() +
  scale_fill_gradient2(low = "yellow", mid = "turquoise", high = "darkblue", 
                       midpoint = median(results_grid$mean_pts_per_trial, na.rm = TRUE),
                       name = "mean pts per trial") +
  scale_x_continuous(breaks = lambda_w_minus) +
  scale_y_continuous(breaks = lambda_w_plus) +
  labs(x = "lambda_w_minus (learning rate for weights, negative PE)", 
       y = "lambda_w_plus (learning rate for weights, positive PE)",
       subtitle = paste(n_simulations, "simulations per combination")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p)
