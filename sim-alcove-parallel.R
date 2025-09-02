# Load required libraries
source("ALCOVE-RL-master.R")
library(DEoptim) 
library(tidyverse)
library(parallel)      # For parallel processing
library(foreach)       # For parallel loops
library(doParallel)    # Backend for parallel processing

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
    model_predictions$action == 2 ~ 0,
    TRUE ~ 0
  )
  
  model_predictions$data$total_pt <- sum(model_predictions$data$pt_outcome)
  return(model_predictions)
}

# OPTION 1: PARALLEL WITH REDUCED PARAMETER SPACE ####
# Use coarser grid first to identify promising regions
coarse_lambda_w <- seq(0.05, 0.95, by = 0.05)      
coarse_lambda_alpha <- seq(0.05, 0.95, by = 0.05) 
n_simulations <- 100

# Set up parameter grid
results_grid <- expand.grid(lambda_w = coarse_lambda_w, lambda_alpha = coarse_lambda_alpha)
results_grid$mean_total_pt <- NA

# Fixed parameters
fixed_params <- c(2, 1)  # c = 2, phi = 1

# OPTION 2: PARALLEL PROCESSING SETUP ####
# Detect number of cores and use most of them
n_cores <- detectCores() - 2  # Leave one core free
cat("Using", n_cores, "cores for parallel processing\n")

# Register parallel backend
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Export necessary objects to worker nodes
clusterEvalQ(cl, {
  source("ALCOVE-RL-master.R")
  library(tidyverse)
})

clusterExport(cl, c("all_stimuli", "h", "w_init", "alpha_init", "exp_info", 
                    "calculate_pt_outcome", "fixed_params", "n_simulations"))

cat("Running", nrow(results_grid), "parameter combinations with", n_simulations, "simulations each...\n")

# PARALLEL LOOP OVER PARAMETER COMBINATIONS ####
system.time({
  results_list <- foreach(combo_idx = 1:nrow(results_grid), 
                          .combine = 'c',
                          .packages = c('tidyverse')) %dopar% {
                            
                            lambda_w <- results_grid$lambda_w[combo_idx]
                            lambda_alpha <- results_grid$lambda_alpha[combo_idx]
                            sim_params <- c(fixed_params, lambda_w, lambda_alpha)
                            
                            # Run simulations for current parameter combination
                            total_pts <- numeric(n_simulations)
                            
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
                              
                              # Get actions and calculate points
                              model_predictions$action <- apply(model_predictions$pr, 1, which.max)
                              model_predictions$data <- tr_data
                              model_predictions <- calculate_pt_outcome(model_predictions)
                              total_pts[sim] <- unique(model_predictions$data$total_pt)
                            }
                            
                            # Return mean for this parameter combination
                            mean(total_pts)
                          }
})

# Stop the cluster
stopCluster(cl)

# Store results
results_grid$mean_total_pt <- results_list

cat("Parallel processing completed!\n")

# OPTION 3: VECTORIZED HEATMAP CREATION ####
library(ggplot2)

# Create the heatmap
p <- ggplot(results_grid, aes(x = lambda_w, y = lambda_alpha, fill = mean_total_pt)) +
  geom_tile() +
  scale_fill_gradient2(low = "lightblue", mid = "white", high = "darkred", 
                       midpoint = median(results_grid$mean_total_pt, na.rm = TRUE),
                       name = "Mean\nTotal Points") +
  scale_x_continuous(breaks = coarse_lambda_w) +
  scale_y_continuous(breaks = coarse_lambda_alpha) +
  labs(x = "Lambda_w (Learning Rate for Weights)", 
       y = "Lambda_alpha (Learning Rate for Attention)",
       title = "Mean Total Points Across Parameter Space (Coarse Grid)",
       subtitle = paste("Based on", n_simulations, "simulations per combination, using", n_cores, "cores")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p)

# Print summary statistics
cat("\nSummary of results:\n")
best_idx <- which.max(results_grid$mean_total_pt)
cat("Best parameter combination:\n")
cat("lambda_w =", results_grid$lambda_w[best_idx], 
    ", lambda_alpha =", results_grid$lambda_alpha[best_idx], 
    ", mean_total_pt =", round(results_grid$mean_total_pt[best_idx], 2), "\n")

worst_idx <- which.min(results_grid$mean_total_pt)
cat("Worst parameter combination:\n")
cat("lambda_w =", results_grid$lambda_w[worst_idx], 
    ", lambda_alpha =", results_grid$lambda_alpha[worst_idx], 
    ", mean_total_pt =", round(results_grid$mean_total_pt[worst_idx], 2), "\n")

# OPTION 4: FINE-GRAINED SEARCH AROUND BEST REGION ####
# Based on the coarse results, do a fine search around the best parameters
best_lambda_w <- results_grid$lambda_w[best_idx]
best_lambda_alpha <- results_grid$lambda_alpha[best_idx]

cat("\nCoarse grid search completed. Use fine grid search code above if you want higher resolution around the best region.\n")
