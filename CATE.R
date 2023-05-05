# General implementation of HTE (CATE and OTP) estimation
# With K = 2 arms




muhat_reg <- function(W, A, Y, muhat_learner){


	if(muhat_learner == 'HAL'){

		muhat <-  fit_hal(Y = Y, X = data.frame(W, A), 
		                  max_degree = min(ncol(W) + 1, 3), 
		                  family = 'gaussian', 
		                  smoothness_orders = 0, reduce_basis = NULL, 
		                  num_knots = nrow(W))

	}else if(muhat_learner == 'SL'){

		muhat <-  SuperLearner(Y = Y, X = data.frame(W, A), family = gaussian(),
                               cvControl = list(V = 5),
                               SL.library = c("SL.glmnet", "SL.ranger", "SL.xgboost", "SL.hal9001"))

	}else if(muhat_learner == 'XGB'){
		muhat <-  SuperLearner(Y = Y, X = data.frame(W, A), family = gaussian(),
                               cvControl = list(V = 5),
                               SL.library = c("SL.xgboost"))

	}else{
		stop('Not a valid learner')
	}

	return(muhat)

}

pihat_reg <- function(W, A, pihat_learner){

	K <- length(unique(A))


	if(pihat_learner == 'SL'){
		lib <- c("SL.mean", "SL.glmnet", "SL.ranger", "SL.xgboost", "SL.hal9001")

	}else if(pihat_learner == 'XGB'){

		print("test")

		lib <- c("SL.xgboost")

	}else{
		lib <- c('SL.mean')
	}


    if(K == 2){

    	pihat <-  SuperLearner(Y = A, X = data.frame(W), 
    		                   family = binomial(),
                               cvControl = list(V = 5),
                               SL.library = lib)

    	return(pihat)


    # If K > 2, hack together a multi-class super learner
	}else{

		pihat_list = list()
		for(i in 0:K - 2){
			if(i == 0){
				pihati <- SuperLearner(Y = (A == i), X = W,
				                       family = binomial(),
				                       cvControl = list(V = 5),
				                       SL.library = lib)

				pihat_list[paste0('pihat',i)] <- pihati

			}else{

				A_sub <- A[A <= i]
				W_sub <- W[A <= i, ]
				cond_pihat <- SuperLearner(Y = (A_sub == i), X = W_sub,
				                       family = binomial(),
				                       cvControl = list(V = 5),
				                       SL.library = lib)

				pihati <- Reduce(`+`, pihat_list)*cond_pihat

				pihat_list[paste0('pihat',i)] <- pihati

			}
			
		}
    }
}

po_reg <- function(W, pseudo_out, po_learner){

	if(po_learner == 'HAL'){

		CATE_fit <- fit_hal(Y = pseudo_out, X = W, 
		                    max_degree = min(ncol(W), 3), 
		                    family = 'gaussian', 
		                    smoothness_orders = 0, reduce_basis = NULL, 
		                    num_knots = nrow(W))

	}else if(po_learner == 'SL'){


		CATE_fit <-  SuperLearner(Y = pseudo_out, X = W, family = gaussian(),
                                  cvControl = list(V = 5),
                                  SL.library = c("SL.glmnet", "SL.ranger", "SL.xgboost", "SL.hal9001"))

	}else if(po_learner == 'XGB'){

		CATE_fit <-  SuperLearner(Y = pseudo_out, X = W, family = gaussian(),
                                  cvControl = list(V = 5),
                                  SL.library = c("SL.xgboost"))

	}else{
		stop('Not a valid learner')
	}
}

rename_fn <- function(df, ind, newname){

	names(df)[ind] <- newname
	return(df)

}

CATE <- function(W1, A1, Y1, W2 = NULL, A2 = NULL, Y2 = NULL, cond_set, method, muhat_learner, pihat_learner = NULL, po_learner = 'HAL'){

	if(is.null(Y2)){
		W2 <- W1
		A2 <- A1
		Y2 <- Y1
	}

	# Count the number of treatments
	K <- length(unique(A1))
	print(K)


	# CASE 1: Q-learning based on all covariates
	# Simplest case with only one stage regression
	# All other cases require a pseudo outcome regression
	# Also, no need for sample splitting here
	if(method == 'Q' & length(cond_set) == ncol(W1)){

		# Fit the outcome regression
	    muhat <- muhat_reg(W1, A1, Y1, muhat_learner)

	    return(muhat)
	}

	# CASE 2: Q-learning but we require a second stage regression 
	else if(method == 'Q'){


		# Fit the outcome regression
		muhat <- muhat_reg(W1, A1, Y1, muhat_learner)

		# Create modified data structure
		# Stack K-1 duplicates of the data with counterfac A values
		for(k in 1:(K - 1)){

			if(k == 1){
				mod <- data.frame(W2, A = rep(k, nrow(W2)))
			}else{
				mod <- rbind(mod, data.frame(W2, A = rep(k, nrow(W2))))
			}
			
		}

		# create another modified data structure with A = 0 for all rows
		mod0 <- mod
		mod0$A <- rep(0, nrow(mod))

		# Get the a pseudo outcome 
		# Here it's just muhat(a, w) - muhat(0, w)
		if(muhat_learner == 'SL') mod$muhat_aw <- predict(muhat, mod)$pred else mod$muhat_aw <- predict(muhat, mod)
		if(muhat_learner == 'SL') mod$muhat_0w <- predict(muhat, mod0)$pred else mod$muhat_0w <- predict(muhat, mod0)

		pseudo_out <- mod$muhat_aw - mod$muhat_0w

	    # Perform the pseudo-outcome regression
	    CATE_fit <- po_reg(mod[c(cond_set)], pseudo_out, po_learner)

	    return(CATE_fit)

	}
	
	# CASE 3: Doubly robust estimation
    else if(method == 'DR'){


    	# Fit the outcome regression
	    muhat <- muhat_reg(W1, A1, Y1, muhat_learner)
	    # Fit the propensity score
    	pihat <- pihat_reg(W1, A1, pihat_learner)

    	# Create modified data structure
		# Stack K-1 duplicates of the data with counterfac A values
		for(k in 1:(K - 1)){

			if(k == 1){
				mod <- data.frame(W = W2, A = A2, a = rep(k, nrow(W2)), zero = rep(0, nrow(W2)), Y = Y2)
			}else{
				mod <- rbind(mod, data.frame(W = W2, A = A2, a = rep(k, nrow(W2)), zero = rep(0, nrow(W2)), Y = Y2))
			}
			
		}

		names(mod)[1:ncol(W2)] <- names(W2)

    	# Create more columns to build the pseudo-outcome
    	mod$IAeqa <- ifelse(mod$A == mod$a, 1, 0)
    	mod$IAeq0 <- ifelse(mod$A == 0, 1, 0)
    	mod$pi_aw <- predict(pihat, mod[names(W2)], onlySL = TRUE)$pred # get the correct pred for a
    	mod$pi_0w <- 1 - mod$pi_aw # generalize for K > 2
    	mod$muhat_Aw <- predict(muhat, mod[c(names(W2), 'A')])$pred
    	mod$muhat_aw <- predict(muhat, rename_fn(mod[c(names(W2), 'a')], ncol(W2) + 1, 'A'))$pred
        mod$muhat_0w <- predict(muhat, rename_fn(mod[c(names(W2), 'zero')], ncol(W2) + 1, 'A'))$pred

        # construct the pseudo outcome
	    pseudo_out <- ((mod$IAeqa/mod$pi_aw) - (mod$IAeq0/mod$pi_0w))*(mod$Y - mod$muhat_Aw) + mod$muhat_aw - mod$muhat_0w

	    # Perform the pseudo-outcome regression
	    set.seed(1)
	    CATE_fit <- po_reg(W = mod[c(cond_set)], pseudo_out = pseudo_out, po_learner = po_learner)

	    return(CATE_fit) 
	}

}

