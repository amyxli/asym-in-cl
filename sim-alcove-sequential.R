# Load required libraries
source("ALCOVE-RL-master.R")
library(DEoptim) 
library(tidyverse)

# INITIALIZE MODEL STRUCTURE ####
## Create all possible stimuli ####
all_stimuli <- expand.grid(
  dim1 = c(0, 1),
  dim2 = c(0, 1), 
  dim3 = c(0, 1)
)

## Apply the AND rule to assign category (dangerous if dim1==1 AND dim2==1) ####
all_stimuli$true_category <- ifelse(
  all_stimuli$dim1 == 1 & all_stimuli$dim2 == 1,
  "dangerous",
  "friendly"
)

# Create exemplar matrix (all 8 possible stimuli)
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

# Initial weights (zeros)
w_init <- matrix(0, nrow = 2, ncol = 8)  # 2 categories × 8 exemplars

# Initial attention weights (equal)
alpha_init <- c(0.33, 0.33, 0.34)  # or c(1/3, 1/3, 1/3)

# Experiment info
exp_info <- c(3, 2, 8, 1)  # d_num, k_num, stim_num, t_type

# Function to calculate pt_outcome for a single simulation
calculate_pt_outcome <- function(model_predictions) {
  model_predictions$data$pt_outcome <- case_when(
    model_predictions$action == 1 & model_predictions$data$cat2 == 1 ~ -3,
    model_predictions$action == 1 & model_predictions$data$cat1 == 1 ~ 1,
    model_predictions$action == 2 ~ 0,  # action == 0 should be action == 2 based on your which.max logic
    TRUE ~ 0
  )
  
  model_predictions$data$total_pt <- sum(model_predictions$data$pt_outcome)
  return(model_predictions)
}

# Set up parameter grid
lambda_w_values <- seq(0.05, 0.95, by = 0.05)  # 19 values
lambda_alpha_values <- seq(0.05, 0.95, by = 0.05)  # 19 values

# Fixed parameters (c and phi)
fixed_params <- c(2, 1)  # c = 2, phi = 1

# Number of simulations per parameter combination
n_simulations <- 100

# Initialize results storage
results_grid <- expand.grid(lambda_w = lambda_w_values, lambda_alpha = lambda_alpha_values)
results_grid$mean_total_pt <- NA

set.seed(123)  # For reproducibility

# Loop through all parameter combinations
total_combinations <- nrow(results_grid)
cat("Running", total_combinations, "parameter combinations with", n_simulations, "simulations each...\n")

for (combo_idx in 1:total_combinations) {
  lambda_w <- results_grid$lambda_w[combo_idx]
  lambda_alpha <- results_grid$lambda_alpha[combo_idx]
  
  # Set current parameters
  sim_params <- c(fixed_params, lambda_w, lambda_alpha)
  
  # Storage for current combination
  total_pts <- numeric(n_simulations)
  
  # Run simulations for current parameter combination
  for (sim in 1:n_simulations) {
    # Generate new trial data for each simulation
    set.seed(456 + combo_idx * n_simulations + sim)  # Unique seed for each simulation
    n_trials <- 96
    trial_indices <- sample(1:8, n_trials, replace = TRUE)
    trial_stimuli <- all_stimuli[trial_indices, ]
    
    # Format for ALCOVE
    tr_data <- data.frame(
      # Stimulus dimensions
      dim1 = trial_stimuli$dim1,
      dim2 = trial_stimuli$dim2,
      dim3 = trial_stimuli$dim3,
      
      # Correct category feedback
      cat1 = ifelse(trial_stimuli$true_category == "friendly", 1, 0),
      cat2 = ifelse(trial_stimuli$true_category == "dangerous", 1, 0),
      
      # All learning trials
      ctr = rep(0, n_trials)
    )
    
    # Run model with current parameters
    model_predictions <- ALCOVE_RL(
      intl_par = sim_params,
      tr_data = tr_data[, 1:6],  # Exclude rep column
      h = h,
      w = w_init,
      alpha = alpha_init,
      q = 1,
      r = 2,
      exp_info = exp_info
    )
    
    # Get simulated actions (1 for approach/cat1, 2 for avoid/cat2)
    model_predictions$action <- apply(model_predictions$pr, 1, which.max)
    model_predictions$data <- tr_data
    
    # Calculate pt_outcome and total_pt
    model_predictions <- calculate_pt_outcome(model_predictions)
    
    # Store results
    total_pts[sim] <- unique(model_predictions$data$total_pt)
  }
  
  # Calculate mean total_pt for this parameter combination
  results_grid$mean_total_pt[combo_idx] <- mean(total_pts)
  
  # Print progress every 10 combinations
  if (combo_idx %% 10 == 0) {
    cat("Completed", combo_idx, "of", total_combinations, "parameter combinations\n")
    cat("lambda_w =", lambda_w, ", lambda_alpha =", lambda_alpha, 
        ", mean_total_pt =", round(results_grid$mean_total_pt[combo_idx], 2), "\n")
  }
}

# Create heatmap
cat("\nCreating heatmap...\n")

# Reshape data for heatmap
heatmap_matrix <- matrix(results_grid$mean_total_pt, 
                         nrow = length(lambda_alpha_values), 
                         ncol = length(lambda_w_values), 
                         byrow = FALSE)

# Create heatmap
library(ggplot2)

# Convert to long format for ggplot
heatmap_data <- expand.grid(lambda_w = lambda_w_values, lambda_alpha = lambda_alpha_values)
heatmap_data$mean_total_pt <- results_grid$mean_total_pt

# Create the heatmap
p <- ggplot(heatmap_data, aes(x = lambda_w, y = lambda_alpha, fill = mean_total_pt)) +
  geom_tile() +
  scale_fill_gradient2(low = "lightblue", mid = "white", high = "darkred", 
                       midpoint = median(heatmap_data$mean_total_pt, na.rm = TRUE),
                       name = "Mean\nTotal Points") +
  scale_x_continuous(breaks = seq(0.05, 0.95, by = 0.1)) +
  scale_y_continuous(breaks = seq(0.05, 0.95, by = 0.1)) +
  labs(x = "Lambda_w (Learning Rate for Weights)", 
       y = "Lambda_alpha (Learning Rate for Attention)",
       title = "Mean Total Points Across Parameter Space",
       subtitle = paste("Based on", n_simulations, "simulations per parameter combination")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p)

# Print summary statistics
cat("\nSummary of results:\n")
cat("Best parameter combination:\n")
best_idx <- which.max(results_grid$mean_total_pt)
cat("lambda_w =", results_grid$lambda_w[best_idx], 
    ", lambda_alpha =", results_grid$lambda_alpha[best_idx], 
    ", mean_total_pt =", round(results_grid$mean_total_pt[best_idx], 2), "\n")

cat("\nWorst parameter combination:\n")
worst_idx <- which.min(results_grid$mean_total_pt)
cat("lambda_w =", results_grid$lambda_w[worst_idx], 
    ", lambda_alpha =", results_grid$lambda_alpha[worst_idx], 
    ", mean_total_pt =", round(results_grid$mean_total_pt[worst_idx], 2), "\n")

cat("\nOverall range of mean_total_pt:", 
    round(min(results_grid$mean_total_pt), 2), "to", 
    round(max(results_grid$mean_total_pt), 2), "\n")