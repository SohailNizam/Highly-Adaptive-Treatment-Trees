# Highly Adaptive Regression Trees
# Journal of Evolutionary Intelligence
# Sohail Nizam, Emory University, sohail.nizam@emory.edu
# David Benkeser, Emory University, benkeser@emory.edu

# Building example trees from paper

# import tree building script
import build_HART

# import necessary libraries
import pandas as pd
import numpy as np
import copy
from anytree import Node
from anytree import AnyNode, RenderTree
from anytree.exporter import DotExporter
import random
import math


# Initialize data structures
hal_df = pd.read_csv('./breast_cancer_py.csv')
data = pd.read_csv('./breast_cancer.csv')
data = data.drop(data.columns[[0]], axis=1) 
intercept = pd.read_csv('./breast_cancer_intercept.csv')
model = create_model(hal_df)
R = {'gt' : {}, 'lt' : {}}


# Sub-Tree with binning (Figure 7 in paper)
# Create a sub-tree by defining a list of tuples indicating which regions to restrict.
# ('tumorsize_6', 'gt') indicates the region where tumor sizes are greater than 6
splits = [('age_3', 'lt'), ('age_2', 'gt'), ('tumorsize_6', 'gt')]
# subtree() createst he subtree
bc_subtree = subtree(splits, model, data, R, intercept)
# three_bin_risk_tree() takes any tree and bins each terminal node into one of three groups
# and collapses regions that no longer differ.
bc_binned_subtree = three_bin_risk_tree(bc_subtree)
DotExporter(bc_binned_subtree,
            nodeattrfunc
           = lambda node: 'label = "{}"'.format(node.display_name)).to_picture("bc_binned_subtree.png")


# Binary decision tree (Figure 8 in paper)
split_candidates = create_split_candidates(model)
# First create an unrestricted tree containing risk predictions in the terminal nodes
bc_tree = risk_tree(split_candidates, model, data, R, total_beta = intercept['x'][0])
# Convert the risk tree into a decision tree by thresholding risk scores at .5
bc_decision_tree = decision_tree(bc_tree, .50)
# Collaps sub-regions of the decision tree made identical by the thresholding
bc_decision_tree = collapse_identical_subtrees(collapse_identical_subtrees(bc_decision_tree))
DotExporter(bc_decision_tree,
            nodeattrfunc
           = lambda node: 'label = "{}"'.format(node.display_name)).to_picture("bc_decision_tree.png")


