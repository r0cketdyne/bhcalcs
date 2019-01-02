### bhcalcs/scripts/make_ZZ_dict.py
### Converts the measured ZZ values in the file bhcalcs/ZZ_values_measured.yaml to a dictionary of the actual energies of the states 11, 12, 21, and 22, and saves it in the file bhcalcs/ZZ_eigenergies.yaml
### run this from the base bhcalcs directory

import yaml
import os
import sys
from qnl_projects.BHDecoder.qutrit_utils import get_ZZ_parameters
import numpy as np

# check that this script is being run from the base directory
# (otherwise the file loading doesn't work properly)
curdir = os.getcwd()
if curdir[-7:] != 'bhcalcs':
    sys.exit('Please rerun this from the base directory. Aborting')

# load measured ZZ values
with open('ZZ_values_measured.yaml', 'r') as f:
    ZZ_values_measured = yaml.load(f)

calculated_ZZ_eigenergies = {}
# convert to energies
for qutrit_pair,qutrit_measured_ZZ in ZZ_values_measured.items():
    q0 = int(qutrit_pair[1])
    q1 = int(qutrit_pair[3])

    calculated_ZZ_eigenergies[qutrit_pair] = get_ZZ_parameters(q0, q1, qutrit_measured_ZZ)[0]

with open('ZZ_eigenergies.yaml', 'w') as f:
    ZZ_values_measured = yaml.dump(calculated_ZZ_eigenergies, f, default_flow_style=False)
