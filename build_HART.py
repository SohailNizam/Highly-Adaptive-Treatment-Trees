# HART main script

import pandas as pd
import numpy as np
import copy
from anytree import Node
from anytree import AnyNode, RenderTree
from anytree.exporter import DotExporter
import random
import math


def create_model(df):
    '''
    Create the model object from the R hal object
    Input: A pandas data frame containing  3 columns 
          (col numbers, split values, beta coefficients)
          
    Output: A dictionary(keys = strings like 'x1_1', values = list(label, beta, split))
    Example: {'x1_1' : ['x1>5', 1.5, 5]}
    '''
    
    #initialize an empty dict
    model = {}
    
    #iterate over all rows of the input df
    #there will be one dict entry per row
    for index, row in df.iterrows():
        
        #First deal with interaction terms.
        #all int terms will have a "(" in them
        #a hold over from them being stored in R vectors
        
        if "(" in row["var_name"]:
            
            
            #int term "var_name" strings look like "c("var1", "var2")"
            #remove the leading "c(" and the trailing ")" and extra "s
            #leaves us with just "var1, var2"
            var = (row["var_name"].replace("c(", "")).replace(")", "").replace('"', "")
            
            #split on "," to get list ["var1", "var2"]
            var = var.split(",")
            
            
            #The "cutoff" values look the same. Long decimals instead of ints
            #again remove "c(" and ")"
            val = (row["cutoffs"].replace("c(", "")).replace(")", "")
            #split into list
            val = val.split(",")
            
            #initialize label and key strings
            #label will be displayed in the tree node (part of the dict val)
            label = ""
            #key will hold the corresponding value from hal_fit
            #this will link the two objects
            #this is the dict key
            key = ""
            
            
            #iterate over both lists
            #construct the label and key strings
            for name, cut in zip(var, val):
                label += name  + "<" + cut + ","

                key += name + "_" + cut + ","

            
            #remove extra trailing comma from each
            label = label[:-1]
            key = key[:-1]

            # also set the split value to 0
            # this will never be used
            split_val = 0


        #if not an interaction term 
        else:
            
            #same deal but only one value, so no iteration
            #want to turn 67.0 into 67
            #but keep 67.5 as 67.5
            if str(int(float(row["cutoffs"]))) == str(row["cutoffs"]):
                string = str(int(row["cutoffs"]))
            else:
                string = str(row["cutoffs"])
            
            
            label = row["var_name"] + "<" + string
            key = row["var_name"] + "_" + string

            # also get the split value
            split_val = row['cutoffs']
        
        
        #remove all spaces in strings for easier parsing in grow_tree()
        label = label.replace(" ", "")
        key = key.replace(" ", "")
            
        
        #add the label and beta value to the dict in a 

        #with key as the dict key
        model[key] = [label, row["V4"], split_val]
    
    return(model)


def create_split_candidates(model):
    '''
    Create a dict of all of the main effects values
    that need to be involved in the tree.
    Keys are the strings in hal_fit like "x1_1".
    Values are lists of all the interaction terms that cause this term
    to be in the dictionary. So the list may be empty
    if the term is only there because of its main effect beta.
    
    Items appear in this dict either because their 
    main effect coefficient is nonzero
    or because they appear in an interaction term whose
    coefficient is nonzero.
    
    Input: hal_fit object (list of lists of strings)
           hal_dict object
    Output: Dictionary
    '''
    
    
    #initialize the list to hold the important vals
    value_list = []
    
    #go through all of the model keys
    for val, attr in model.items():
        if "," in val and attr[1] != 0:
            value_list += val.split(",")
    
        elif "," not in val and attr[1] != 0:
            value_list.append(val)
    
    #initialize the final dictionary to be returned
    #with value_list as the keys
    split_candidates = dict.fromkeys(value_list, 0)
    
    
    #iterate over all of the model keys
    for key in model.keys():
        
        #if the key is an interaction term
        if "," in key and model[key][1] != 0:
            #split the term into a list
            key_split = key.split(",")

            #append the original interaction term
            #to the list corresponding to each main effect
            #involved in it
            for item in key_split:
                if item in split_candidates.keys():
                    if split_candidates[item] == 0:
                        split_candidates[item] = [key]
                    else:
                        split_candidates[item].append(key)

            
            #so "x1_1,x2_1" will be added to the list
            #corresponding to both "x1_1" and "x2_1"
            #Now we know why each term is actually in
            #split_candidates
    
    #also go through and add each of the values
    #themselves to the list of "reasons for being present"
    #if necessary
    for val in split_candidates.keys():
 
            
        if model[val][1] != 0 and split_candidates[val] == 0:
            split_candidates[val] = [val]
            
        elif model[val][1] != 0 and split_candidates[val] != 0:
            split_candidates[val].append(val)
    
    return(split_candidates)

def subset_df(df, R):

  '''
  Subset given data frame according to region R.
  E.g. if R tells us X1 > 5, we subset df to rows with X1 > 5x 
  '''
  # we'll subset the df as we iterate over the unique col names in lt + gt
  for colname in set(list(R['lt'].keys()) + list(R['gt'].keys())):
    if colname in R['lt']:
      min_num = float(min(R['lt'][colname]).split('_')[1])
      df = df.loc[df[colname] < min_num]
    if colname in R['gt']:
      max_num = float(max(R['gt'][colname]).split('_')[1])
      df = df.loc[df[colname] >= max_num]
  return(df)
 
def gini_impurity(df, val):
  '''
  Calculate gini impurity of a value in the data frame
  '''

  # use the string val to identify the 
  # correct row in df
  colname = val.split('_')[0]
  number = float(val.split('_')[1])
  
  # split the y col in df on that row
  lh_y = df.iloc[:, 0].loc[df[colname] < number]
  rh_y = df.iloc[:, 0].loc[df[colname] >= number]
  
  # calculate gini impurity for each side
  n_l = int(len(lh_y))
  n_r = int(len(rh_y))
  n = int(n_l + n_r)

  if n_l == 0:
    lh_impurity = 0
  else:
    lh_impurity = 1 - np.mean(lh_y)**2 - np.mean((np.ones(n_l) - lh_y))**2
    
    
  rh_impurity = 1 - np.mean(rh_y)**2 - np.mean((np.ones(n_r) - rh_y))**2
  # calculate the total impurity for this split
  gini = (n_l / n)*lh_impurity + (n_r / n)*rh_impurity
  return(gini)


def is_out_of_region_left(R, val):
  '''
  Is this potential split value out of the region we're considering
  on the left hand side (i.e. val is X1 = 5 and we know X1 < 5 based on R)
  '''
  colname = val.split('_')[0]
  if colname in R['lt']:
    if val in R['lt'][colname]:
      return(True)
  return(False)

def is_out_of_region_right(R, val):
  '''
  Is this potential split value out of the region we're considering
  on the right hand side (i.e. val is X1 = 5 and we know X1 >= 5 based on R)
  '''


  colname = val.split('_')[0]
  if colname in R['gt']:
    if val in R['gt'][colname]:
      return(True)

  return(False)

def is_out_of_region(R, val):
  return(is_out_of_region_left(R, val) or is_out_of_region_right(R, val))


def shrink_candidates_left(split_candidates, R):

  '''
  Remove values from our set of split candidates based on the region we're in.
  '''

  new_split_candidates = copy.deepcopy(split_candidates)

  # for every candidate split
  for cand, l in split_candidates.items():
    # first deal with the left hand (LH) subtree candidates
    # if the candidate appears in the list of
    # values we know we're 'less than' or we know we're 'greater than'
    if is_out_of_region(R, cand):
      # remove it from the candidates for the LH subtree
      del new_split_candidates[cand]
    # if we can't remove the whole candidate
    # we can still remove some int terms 
    # from its 'reasons for being in model' list
    else:
      # a new list that will exclude int terms with
      # with components that we know we're 'less than'
      new_l = []
      for val in l:
        # if it's not an int term, leave it in
        # obviously it's not out of region yet since we got to this else
        if ',' not in val:
          new_l.append(val)
        
        # if it is an int term
        else:
          # split into individual component terms
          terms = val.split(',')
          # if any of these individual components are in R['lt']
          # THIS NEEDS TO BE DIRECTIONAL
          # need a is_known_less_than
          if any(is_out_of_region_left(R, term) for term in terms):
            pass
          else:
            new_l.append(val)



			# if new_l is empty, it means the candidate was only in there
			# because of int terms that are now definitely 0
			# so we can remove the whole candidate
      if len(new_l) == 0:
        del new_split_candidates[cand]
			# if not, we can't remove this candidate just yet
      else:
        # make new_l the 'reason for being' list for this candidate
        new_split_candidates[cand] = new_l
  return(new_split_candidates)

def shrink_candidates_right(split_candidates, R):

  '''
  Remove values from our set of split candidates based on the region we're in.
  '''

  new_split_candidates = copy.deepcopy(split_candidates)
  # next deal with the right hand (RH) subtree candidates
  # if cand appears in the list of values we
  # know we're 'greater than'
  for cand, l in split_candidates.items():
    if is_out_of_region(R, cand):
      # remove it from the candidates for the RH subtree
      del new_split_candidates[cand]


	# No need to do more RH trimming
	# an int term won't become fixed after a RH move (and therefore need to be removed)
	# unless all the components are known to be 1, including cand
	# and if we know we're greater than cand, the above case handles things

  return(new_split_candidates)


def update_RH_region(R, split, model):
  '''
  Update region R based on split we've just added to the tree.
  Practically we add split and any value <= split within same feature to R['gt'][colname]
  '''
  colname = split.split('_')[0]
  number = model[split][2]
  gt_additions = []
  beta_additions = 0
  for term, info in model.items():
    if ',' not in term and term.split('_')[0] == colname and info[2] <= number:
      # TODO only add to gt_additions if not already in gt
      gt_additions.append(term)
      beta_additions += float(model[term][1])
    else:
      continue
  RH_R = copy.deepcopy(R)
  if colname not in RH_R['gt']:
    RH_R['gt'][colname] = gt_additions
  else:
    RH_R['gt'][colname] += gt_additions
  
  return(RH_R, beta_additions)

def update_LH_region(R, split, model):
  '''
  Update region R based on split we've just added to the tree.
  Practically we add split and any value >= split within same feature to R['lt'][colname]
  '''

  colname = split.split('_')[0]
  number = model[split][2]
  
  lt_additions = []
  for term, info in model.items():
    if ',' not in term and term.split('_')[0] == colname and info[2] >= number:
      lt_additions.append(term)
    else:
      continue
  LH_R = copy.deepcopy(R)
  if colname not in LH_R['lt']:
    LH_R['lt'][colname] = lt_additions
  else:
    LH_R['lt'][colname] += lt_additions
  
  return(LH_R)


def get_split(df, split_candidates):

  '''
  Get the value from split_candidates with the highest gini impurity in the data.
  '''

  # if there's no data left in df
  if len(df) == 0 or len(df) != 0: # overriding to always use this criteria
    # get the split with the biggest number of 'reasons' for being a candidate
    max_num = 0
    split = ''
    for cand, reasons in split_candidates.items():
      if len(reasons) > max_num:
        max_num = len(reasons)
        split = cand
    return(split)

  # first get gini_impurity for every value in value dict
  # based on df (which has been passed here already subsetted)
  split = ''
  min_gini = 100000
  for val in split_candidates.keys():
    gini = gini_impurity(df, val)
    if gini < min_gini:
      split = val
      min_gini = gini
  # if no splits have > 0 impurity
  # split still = ''
  # just set split = first candidate
  if split == '':
    split = list(split_candidates.keys())[0]
    
  #return the value with the highest impurity
  return(split)


def risk_tree(split_candidates, model, df, R, total_beta = 0):

  '''
  Grow a risk tree.
  '''

  # base case: no more split candidates
  if len(split_candidates) == 0:
    # required so anytree doesn't merge nodes
    idnum = str(random.uniform(0,1))
    # add approriate interaction term coeffs
    for term, coeff in model.items():
      if ',' in term and coeff[1] != 0:
        if all(is_out_of_region_right(R, val) for val in term.split(',')):
          total_beta += coeff[1]

    prob = math.exp(total_beta) / (math.exp(total_beta) + 1)
    
    # return the node
    return(Node(name = idnum,
                prob = prob,
                #display_name = str(round(prob, 3)),
                display_name = str(round(total_beta, 3)),
                risk_score = total_beta,
                R = R,
                candidates_left = []))
  
  # recursive case: still split candidates left
  else:

    # subset the df to rows falling within region R
    #df_subset = subset_df(df, R)
    # get the next split value
    split = get_split(df, split_candidates)
    
    # create a node out of the split
    node  = Node(name = str(random.uniform(0,1)),
                 display_name = model[split][0],
                 beta = model[split][1],
                 total_beta = total_beta,
                 R = R)
      
    # update region for left child
    left_R = update_LH_region(R, split, model)

    # update split_candidates for left child
    left_split_candidates = shrink_candidates_left(split_candidates, left_R)

    #left_df = subset_df(df, left_R)
      
    # grow subtree on the left hand side of the split
    left_child = risk_tree(left_split_candidates, model, df, left_R, total_beta)
    left_child.parent = node
      
    # update region for right child
    # this includes getting the betas corresponding to the region additions
    right_R, beta_additions = update_RH_region(R, split, model)

    # update split_candidates for right child
    right_split_candidates = shrink_candidates_right(split_candidates, right_R)

    #right_df = subset_df(df, right_R)
      
    # update running sum of betas
    total_beta += beta_additions
      
    # grow subtree on the right hand side of the split
    right_child = risk_tree(right_split_candidates, model, df, right_R, total_beta)
    right_child.parent = node

  return(node)

def subtree(splits, model, data, R, intercept):
  '''
  Grow a subsetted risk tree (restricted based on splits arg)
  '''
  R = {'gt' : {}, 'lt' : {}}
  split_candidates = create_split_candidates(model)
  for split_tup in splits:
    if split_tup[1] == 'lt':
      R = update_LH_region(R, split_tup[0], model)
      split_candidates = shrink_candidates_left(split_candidates, R)
    else:
      R = update_RH_region(R, split_tup[0], model)
      split_candidates = shrink_candidates_right(split_candidates, R)

  return(risk_tree(split_candidates, model, data, R, total_beta = intercept['x'][0]))

def decision_tree(risk_tree, K):

  '''
  Create a decision tree by thresholding a risk tree.
  '''

  # this will always be needed for the new node creation
  # since the new tree nodes also need unique ids
  idnum = risk_tree.name

	# Base case in three parts:

	# 1. we're at a leaf
  if risk_tree.is_leaf:
    pred = 0 if risk_tree.prob < K else 1
    return(Node(name = idnum, display_name = str(pred)))

	# 2. all leaves have prob >= K
  elif all([leaf.prob >= K for leaf in risk_tree.leaves]):
    return(Node(name = idnum, display_name = '1'))

	# 3. all leaves have prob < K
  elif all([leaf.prob < K for leaf in risk_tree.leaves]):
    return(Node(name = idnum, display_name = '0'))


	# Recursive case: repeat for both child nodes
  else:
    # create a node with same display name and id
		# (remember we're creating a new tree that's the same up until a certain point)
    node = Node(name = idnum, display_name = risk_tree.display_name)

		# call function for left hand side
    left_tree = decision_tree(risk_tree.children[0], K)
		# attach resulting subtree to parent node
    left_tree.parent = node

		# call function for right hand side
    right_tree = decision_tree(risk_tree.children[1], K)
		# attach resulting subtree to parent node
    right_tree.parent = node
    
    return(node)


def four_bin_risk_tree(risk_tree):


  '''
  Bin the predicted probabilities into K groups.
  Then prune the tree when adjacent nodes don't differ in binned pred.
  For now, hard coded at 4 bins

  K = 4
  Bins = [0, .25), [.25, .50), [.50, .75), [.75, 1.0]
  
  '''

  bin_dict = {'[0, .25)' : [0, .25],
              '[.25, .50)' : [.25, .50],
              '[.50, .75)' : [.50, .75],
              '[.75, 1.0]' : [.75, 1.0]}

  # this will always be needed for the new node creation
  # since the new tree nodes also need unique ids
  idnum = risk_tree.name

	# Base case in several parts:

	# 1. we're at a leaf
  if risk_tree.is_leaf:
    for label, bin in bin_dict.items():
      if risk_tree.prob >= bin[0] and risk_tree.prob< bin[1]:
        return(Node(name = idnum, display_name = label))

	# 2. all leaves have prob in bin 1
  elif all([leaf.prob < .25 for leaf in risk_tree.leaves]):
    return(Node(name = idnum, display_name = '[0, .25)'))

	# 3. all leaves have prob in bin 2
  elif all([leaf.prob >= .25 and leaf.prob < .50 for leaf in risk_tree.leaves]):
    return(Node(name = idnum, display_name = '[.25, .50)'))

  # 4. all leaves have prob in bin 3
  elif all([leaf.prob >= .50 and leaf.prob < .75 for leaf in risk_tree.leaves]):
    return(Node(name = idnum, display_name = '[.50, .75)'))
  
  # 5. all leaves have prob in bin 4
  elif all([leaf.prob >= .75 for leaf in risk_tree.leaves]):
    return(Node(name = idnum, display_name = '[.75, 1.0]'))


	# Recursive case: repeat for both child nodes
  else:
    # create a node with same display name and id
		# (remember we're creating a new tree that's the same up until a certain point)
    node = Node(name = idnum, display_name = risk_tree.display_name)

		# call function for left hand side
    left_tree = four_bin_risk_tree(risk_tree.children[0])
		# attach resulting subtree to parent node
    left_tree.parent = node

		# call function for right hand side
    right_tree = four_bin_risk_tree(risk_tree.children[1])
		# attach resulting subtree to parent node
    right_tree.parent = node
    
    return(node)


def three_bin_risk_tree(risk_tree):
  

  '''
  Bin the predicted probabilities into K groups.
  Then prune the tree when adjacent nodes don't differ in binned pred.
  For now, hard coded at 4 bins

  K = 3
  Bins = [0.0, 1/3), [1/3, 2/3), [2/3, 1.0), [.75, 1.0]
  
  '''

  bin_dict = {'[0.0, .333)' : [0, 1/3],
              '[.333, .667)' : [1/3, 2/3],
              '[.667, 1.0]' : [2/3, 1.0]}

  # this will always be needed for the new node creation
  # since the new tree nodes also need unique ids
  idnum = risk_tree.name

	# Base case in several parts:

	# 1. we're at a leaf
  if risk_tree.is_leaf:
    for label, bin in bin_dict.items():
      if risk_tree.prob >= bin[0] and risk_tree.prob< bin[1]:
        return(Node(name = idnum, display_name = label))

	# 2. all leaves have prob in bin 1
  elif all([leaf.prob < 1/3 for leaf in risk_tree.leaves]):
    return(Node(name = idnum, display_name = '[0.0, .333)'))

	# 3. all leaves have prob in bin 2
  elif all([leaf.prob >= 1/3 and leaf.prob < 2/3 for leaf in risk_tree.leaves]):
    return(Node(name = idnum, display_name = '[.333, .667)'))

  # 4. all leaves have prob in bin 3
  elif all([leaf.prob >= 2/3 for leaf in risk_tree.leaves]):
    return(Node(name = idnum, display_name = '[.667, 1.0]'))

	# Recursive case: repeat for both child nodes
  else:
    # create a node with same display name and id
		# (remember we're creating a new tree that's the same up until a certain point)
    node = Node(name = idnum, display_name = risk_tree.display_name)

		# call function for left hand side
    left_tree = three_bin_risk_tree(risk_tree.children[0])
		# attach resulting subtree to parent node
    left_tree.parent = node

		# call function for right hand side
    right_tree = three_bin_risk_tree(risk_tree.children[1])
		# attach resulting subtree to parent node
    right_tree.parent = node
    
    return(node)

def collapse_identical_subtrees(tree):

  '''
  After subsetting and binning, we may end up with large subtrees
  that are entirely identical. This is because certain splits were determined
  to be necessary early on, but, post binning, result in identical subtrees.

  This function collapses such sub trees (ie doesn't split on the unecessary value)
  '''

  # base case 1: at a terminal node
  if tree.is_leaf:
    return(Node(name = tree.name, display_name = tree.display_name))


  # base case 2: both children are identical subtrees
  elif identical_subtrees(tree.children[0], tree.children[1]):
    #TODO make a copy of the tree to return
    subtree = copy.deepcopy(tree.children[0])
    return(subtree)
  
  # recursive case: children are not identical subtrees
  else:
    node = Node(name = tree.name, display_name = tree.display_name)

    left_subtree = collapse_identical_subtrees(tree.children[0])
    left_subtree.parent = node
    right_subtree = collapse_identical_subtrees(tree.children[1])
    right_subtree.parent = node

    return(node)

def identical_subtrees(t1, t2):

  '''
  Determine if two subtrees are identical
  '''

  if t1.is_leaf and t2.is_leaf:
    return(t1.display_name == t2.display_name)
  
  elif (not t1.is_leaf) and t2.is_leaf:
    return(False)
  
  elif t1.is_leaf and (not t2.is_leaf):
    return(False)
  
  elif t1.display_name == t2.display_name:
    return(identical_subtrees(t1.children[0], t2.children[0]) and identical_subtrees(t1.children[1], t2.children[1]))
  
  else:

    return(False)


def group_similar_subtrees(tree, min_diff):

  '''
  Group sub tree whose terminal node predictions differ by less than min_diff
  '''
  # base case 1: at a terminal node
  if tree.is_leaf:
    return(Node(name = tree.name, display_name = tree.display_name))
  
  else:
    # base case 2: none of the terminal node preds differ by more than min_diff
    probs = [tree.leaves[i].prob for i in range(len(tree.leaves))]
    if max(probs) - min(probs) < min_diff:
      return(Node(name = tree.name, display_name = '[' + str(round(min(probs), 2)) + ',' + str(round(max(probs), 2)) + ']'))
    # recursive case
    else:
      node = Node(name = tree.name, display_name = tree.display_name)
      left_subtree = group_similar_subtrees(tree.children[0], min_diff)
      left_subtree.parent = node
      right_subtree = group_similar_subtrees(tree.children[1], min_diff)
      right_subtree.parent = node
      return(node)




