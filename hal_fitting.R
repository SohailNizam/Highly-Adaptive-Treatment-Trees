# Highly Adaptive Regression Trees
# Journal of Evolutionary Intelligence
# Sohail Nizam, Emory University, sohail.nizam@emory.edu
# David Benkeser, Emory University, benkeser@emory.edu

# HAL fitting and prep for HART python script

library(hal9001)
library(dplyr)


for_py <- function(hal_fit, df, lambda_ind = 0){
  
  # make basis list a two column matrix
  basis_mat <- Reduce(rbind, hal_fit$basis_list)[as.numeric(names(hal_fit$copy_map)),]
  
  #make fit matrix
  if(lambda_ind == 0){
    fit_mat <- cbind(basis_mat, (hal_fit$coefs)[-1,])
  }else{
    fit_mat <- cbind(basis_mat, hal_fit$lasso_fit$glmnet.fit$beta[,lambda_ind])
  }
  
  
  #cast to data frame
  fit_df <- data.frame(fit_mat)
  
  #add a column with original variable names
  get_names <- function(row){
    return(names(df[,-1])[unlist(row[1])])
  }
  fit_df$var_name <- apply(fit_df, MARGIN = 1, FUN = get_names)
  
  #cast to character
  fit_df <- data.frame(apply(X = fit_df, MARGIN = 2, FUN = as.character))
  
  #rename the last col to "coeffs"
  names(fit_df)[3] <- "coeffs"
  
  return(fit_df)
}




