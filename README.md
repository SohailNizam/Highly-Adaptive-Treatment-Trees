# Highly-Adaptive-Treatment-Trees
Containing code to implement both HART and HATT


## Implementing HART

HART works in two stages. 

1. hal_fitting.R takes a HAL fit object and prepares it for tree creation in the next step.
2. build_HART.py takes the file produced by step 1 and produces trees in a variety of ways.

## Implementing HATT

To build a HATT, first use either CATE.R or multiarm_CATE.R to produce a CATE estimate. CATE.R only handles binary treatments, whereas multiarm_CATE.R can handle larger numbers of treatment arms. In either case, you must specify 'HAL' as the final stage learner and save the resulting fit. Next, follow the HART steps above using the CATE fit.


