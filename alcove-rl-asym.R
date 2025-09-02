#follow the model structure described in Alexander S. Rich, & Todd M. Gureckis
#Initial pars
#h = matrix(c(0,0,0,1,1,0,1,1), byrow = T, 4, 2) # Dim x Examplar
#w = matrix(c(0,0,0,0,0,0,0,0), byrow = T, 2, 4) # Cat x Exemplar
#alpha = c(0.5, 0.5)
#lamda_w = 0.75

#input_dim = tr[, 6:9]
#dim_imput = input_dim[1,]
#a_in = c(0,0)
#t = c(3, NA)
#=========Kruschke(1992): Equation 1===============================#
#activation of hidden node: a_hid
#r = 1
#c = 2
#free parameter: alpha(learnable); fixed parameter r,c; input:h-exemplar matrix (Sim x Dim), dim_imput: input dimension 

sim_distance = function(h, a_in, alpha, r = 2,q = 1, c = 2){
  a_hid = rep(0, dim(h)[2])
  for(j in 1: dim(h)[2]){ #j: stim index
    for(i in 1: length(a_in)){ #i: dimension index
      a_hid[j] = a_hid[j] + alpha[i]*((abs(h[i,j] - a_in[i]))^r)
    }
  }
  a_hid = exp(-c*a_hid^(q/r))
  return(a_hid)
}

#===========Kruschke(1992): Equation 2 modified with normalization=========#
#output nodes ao:
#free parameter: w (learnable)

out_put = function(a_hid, w){
  a_o = rep(0, dim(w)[1])
  for(k in 1: dim(w)[1]){ # Category index
    for(j in 1:length(a_hid)){
      a_o[k] = a_o[k] + w[k,j]*a_hid[j]
    }
  }
  a_o = a_o#/sum(a_hid) #normalization as shown in Gureckis paper; in original ALCOVE no normalization
  return(a_o)
}

#===========Kruschke(1992): Equation 3=============================#
#reponse probability: Luce choice rule
#phi = 1
#free parameter: phi (deterministic), non-negative, high value -> deterministic
rep_prob = function(a_o, phi){
  pr = exp (phi * a_o)/sum(exp(phi*a_o))
  return(pr)
}

#==========Contingent Teacher: Todd Gureckis =======================#
#Teaching parameter
#t = c(0, 1)
contingent_teacher = function(a_o, t){
  teacher = t 
  teacher[which(is.na(t))] = a_o[which(is.na(t))] #if feedback is not available, no update
  return(teacher)
}

#humble teacher
humble_teacher = function(a_o, t){
  teacher = rep(NA,length(t))
  for (k in 1:length(t)){
    if (t[k] == 1){
      teacher[k] = max(c(1,a_o[k]))
    }else {
      teacher[k] = min(c(-1,a_o[k]))
    }
  }
  return(teacher)
}


#=========Error term Kruschke(1992): Equation 4a=======================#
error = function(teacher, a_o){
  e = (teacher - a_o)
  return(e)
}

#=============Association Weight Update================#

# delta_w = function(lamda_w_plus, lamda_w_minus, error, a_hid){
#   del_w = matrix(0, length(error), length(a_hid))
#   for(k in 1:length(error)){
#     for(j in 1:length(a_hid)){
#       # Choose learning rate based on error sign
#       if(error[k] > 0){  # Positive feedback (gain)
#         lr = lamda_w_plus
#       } else {            # Negative feedback (loss)
#         lr = lamda_w_minus
#       }
#       del_w[k,j] = lr * error[k] * a_hid[j]
#     }
#   }
#   return(del_w)
# }

delta_w_with_outcome = function(lamda_w_plus, lamda_w_minus, error, a_hid, outcome_value){
  del_w = matrix(0, length(error), length(a_hid))
  
  for(k in 1:length(error)){
    for(j in 1:length(a_hid)){
      # Use outcome value for asymmetry
      if(outcome_value > 0){  # Positive outcome (gained points)
        lr = lamda_w_plus
      } else {                 # Negative outcome (lost points or zero)
        lr = lamda_w_minus
      }
      del_w[k,j] = lr * error[k] * a_hid[j]
    }
  }
  return(del_w)
}

#====Backward Propagate Error to Hidden Node==================#
#sum(t-a_o)*w 
bp_error = function(a_hid, error, w){
  bp = error %*% w
  for(j in 1: length(a_hid)){
    bp[j] = bp[j]*a_hid[j]
  }
  return(bp)
}
#=============Attention Weight Update================#
#Double check this equation
#lamda_alpha = 0.3
delta_alpha = function(lamda_alpha, c, h, a_in, bp){
  hmx = abs(h-a_in) # h(what's that?) minus dimensional inputs
  del_a = rep(0, length(a_in)) # initialise the weights associated with the dimensions
  for(j in 1:(dim(h)[2])){
    del_a = del_a + bp[j] * hmx[,j] *c
  }
  del_a = -lamda_alpha*del_a
  return(del_a)
}
#=================#
# Modified ALCOV_trial function - now takes 5 parameters instead of 4
ALCOV_trial = function(c, phi, lw_plus, lw_minus, la, alpha, w, h, q, r, a_in, fb, t_type){
  #1) performance of current trial
  a_hid = sim_distance(h, a_in, alpha, c)
  a_o = out_put(a_hid, w)
  pr =  rep_prob(a_o, phi)

  #2) learn from feedback
  if(t_type == 1){ #strict teacher
    t = fb
  } else if (t_type == 2){ #contingent teacher
    t = contingent_teacher(a_o, fb)
  } else if (t_type == 3){ #humble teacher
    t = humble_teacher(a_o, fb)
  }

  e = error(t, a_o)
  # Determine if outcome was positive or negative
  # Based on which action would be taken (highest probability)
  predicted_action = which.max(pr)
  
  if(predicted_action == 1){  # Would approach
    outcome_value = ifelse(fb[1] == 1, 1, -3)  # +1 if friendly, -3 if dangerous
  } else {  # Would avoid
    outcome_value = 0
  }
  
  # Pass outcome value to delta_w
  dw = delta_w_with_outcome(lw_plus, lw_minus, e, a_hid, outcome_value)
  
  # dw = delta_w(lw_plus, lw_minus, e, a_hid)  # Updated call
  
  bp = bp_error(a_hid, e, w)
  da = delta_alpha(la, c, h, a_in, bp)
  nw = w+dw
  nalpha = alpha + da
  nalpha[which(nalpha < 0)] = 0 #non-zero attention weight (Will, 1996)

  t_store = list()
  t_store$pr = pr
  t_store$w = nw
  t_store$alpha = nalpha
  t_store$e= e
  return(t_store)
}

# Modified ALCOVE_RL function - now expects 5 parameters instead of 4
ALCOVE_RL = function(intl_par, tr_data, h, w, alpha, q, r, exp_info){
  # Parse 5 parameters instead of 4
  c = intl_par[1]
  phi = intl_par[2]
  lw_plus = intl_par[3]   # Learning rate for positive errors
  lw_minus = intl_par[4]  # Learning rate for negative errors
  la = intl_par[5]        # Attention learning rate (moved to position 5)

  d_num = exp_info[1]
  k_num = exp_info[2]
  stim_num = exp_info[3]
  t_type = exp_info[4]

  trial_num = dim(tr_data)[1]
  pr_store = matrix(NA, trial_num, 2)
  alpha_store = matrix(NA, trial_num, d_num)
  e_store = matrix(NA, trial_num, k_num)
  w_store = matrix(NA, trial_num, (k_num*stim_num))

  for(n in 1: trial_num){
    trial = tr_data[n, ]
    a_in = as.numeric(trial[1:d_num])
    fb = as.numeric(trial[(d_num+1):(d_num+k_num)])
    ctr = as.numeric(trial[(d_num+k_num+1)])

    if (ctr == 0){
      alpha_store[n,] = alpha
      w_store[n, ] = as.numeric(t(w))
      # Updated call with 5 learning parameters
      trial_fit = ALCOV_trial(c, phi, lw_plus, lw_minus, la, alpha, w, h, q, r, a_in, fb, t_type)
      w = trial_fit$w
      alpha = trial_fit$alpha
      pr_store[n, 1:2] = trial_fit$pr
      e_store[n,] = trial_fit$e
    } else if (ctr == 2){
      alpha_store[n,] = alpha
      w_store[n, ] = as.numeric(t(w))
      # Updated call - set both weight learning rates to 0 for test trials
      trial_fit = ALCOV_trial(c, phi, lw_plus=0, lw_minus=0, la=0, alpha, w, h, q, r, a_in, fb, t_type)
      pr_store[n, 1:2] = trial_fit$pr
      e_store[n,] = trial_fit$e
    }
  }

  store = list()
  store$pr = pr_store
  store$w = w
  store$a = alpha_store
  store$e = e_store
  store$w_list = w_store

  return(store)
}
# ALCOV_trial = function(c, phi, lw_plus, lw_minus, la, alpha, w, h, q, r, a_in, fb, t_type){
#   #1) performance of current trial
#   a_hid = sim_distance(h, a_in, alpha, c)
#   a_o = out_put(a_hid, w)
#   pr =  rep_prob(a_o, phi)
# 
#   #2) learn from feedback
#   if(t_type == 1){ #strict teacher
#     t = fb
#   } else if (t_type == 2){ #contingent teacher
#     t = contingent_teacher(a_o, fb)
#   } else if (t_type == 3){ #humble teacher
#     t = humble_teacher(a_o, fb)
#   }
# 
#   e = error(t, a_o)
#   dw = delta_w(lw_plus, lw_minus, e, a_hid)  # Updated call
#   bp = bp_error(a_hid, e, w)
#   da = delta_alpha(la, c, h, a_in, bp)
#   nw = w+dw
#   nalpha = alpha + da
#   nalpha[which(nalpha < 0)] = 0 #non-zero attention weight (Will, 1996)
# 
#   t_store = list()
#   t_store$pr = pr
#   t_store$w = nw
#   t_store$alpha = nalpha
#   t_store$e= e
#   return(t_store)
# }
# 

# Modified ALCOVE_nlk function - now handles 5 parameters
ALCOVE_nlk = function(int_pars, data, h, w, alpha, q, r, exp_info, upperbound){
  d_num = exp_info[1] 
  k_num = exp_info[2]
  stim_num = exp_info[3]
  t_type = exp_info[4]
  
  tr_data = data[, 1:(d_num+k_num+1)]
  rep = data$rep
  new_param = rep(NA, length(int_pars))  # Now handles 5 parameters
  for(i in 1:length(int_pars)){
    new_param[i] = upperbound[i]/(1+exp(-int_pars[i]))
  }
  m_fit = ALCOVE_RL(new_param, tr_data, h, w, alpha, q = 1, r = 2, exp_info) 
  prob = m_fit$pr
  log_prob = log(prob)
  llk = 0
  nlk = 0 
  for(i in 1:length(rep)){
    llk = llk + log_prob[i, rep[i]]
    nlk = nlk - log_prob[i, rep[i]]
  }
  return(nlk)
}

# Modified ALCOVE_nlk_DEoptim function - now handles 5 parameters  
ALCOVE_nlk_DEoptim = function(int_pars, data, h, w, alpha, q, r, exp_info){
  d_num = exp_info[1] 
  k_num = exp_info[2]
  stim_num = exp_info[3]
  t_type = exp_info[4]
  
  col_n = d_num+k_num+1
  tr_data = data[, 1:col_n]
  rep = data$rep
  new_param = int_pars  # Now expects 5 parameters
  
  m_fit = ALCOVE_RL(new_param, tr_data, h, w, alpha, q = 1, r = 2, exp_info) 
  prob = m_fit$pr
  prob[which(prob==0, arr.ind = T)] = 10^-10
  log_prob = log(prob)
  llk = 0
  nlk = 0 
  for(i in 1:length(rep)){
    llk = llk + log_prob[i, rep[i]]
    nlk = nlk - log_prob[i, rep[i]]
  }
  
  return(nlk)
}

# Transform output function - now handles 5 parameters
transform_output = function(output_param, upperbound){
  new_param = rep(NA, length(output_param))  # Now handles 5 parameters
  for(i in 1:length(output_param)){
    new_param[i] = upperbound[i]/(1+exp(-output_param[i]))
  }
  return(new_param)
}
