# apply these functions to the breast cancer data
#breast_cancer <- read.csv('./breast_cancer.csv')
#hal_fit <- fit_hal(Y = breast_cancer$outcome, X = breast_cancer[,-1], 
 #                  max_degree = 3, family = 'binomial', smoothness_orders = 0, 
  #                 reduce_basis = NULL, num_knots = nrow(breast_cancer))
#py_output <- for_py(hal_fit, breast_cancer)
#intercept <- hal_fit$coefs[1]

# Write files to be used for tree building
#write.csv(py_output, './breast_cancer_py.csv')
#write.csv(intercept, './breast_cancer_intercept.csv')