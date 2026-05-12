import uproot
import awkward as ak
import vector
import numpy as np
import os

file_tt = uproot.open("4topLO_24March26_minus.root")
tree = file_tt["reco"]
print(tree.keys())
print(tree["met_met"].array())
tree = file_tt["truth"]
print(tree.keys())