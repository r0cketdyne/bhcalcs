### bhcalcs/scripts/make_ZZ_dict.py
### Converts the measured ZZ values in the file bhcalcs/ZZ_values_measured.yaml to a dictionary of the actual energies of the states 11, 12, 21, and 22, and saves it in the file bhcalcs/ZZ_eigenergies.yaml
### run this from the base bhcalcs directory

import yaml
import os
import sys
from qnl_projects.BHDecoder.qutrit_utils import get_ZZ_parameters

# Define the file paths
measured_ZZ_file = os.path.join("bhcalcs", "ZZ_values_measured.yaml")
eigenergies_file = os.path.join("bhcalcs", "ZZ_eigenergies.yaml")

# Check if the script is being run from the base directory
if not os.path.basename(os.getcwd()) == 'bhcalcs':
    sys.exit('Please rerun this script from the base directory. Aborting')

# Load measured ZZ values
with open(measured_ZZ_file, 'r') as f:
    ZZ_values_measured = yaml.safe_load(f)

# Convert measured ZZ values to energies
calculated_ZZ_eigenergies = {}
for qutrit_pair, qutrit_measured_ZZ in ZZ_values_measured.items():
    q0, q1 = int(qutrit_pair[1]), int(qutrit_pair[3])
    calculated_ZZ_eigenergies[qutrit_pair] = get_ZZ_parameters(q0, q1, qutrit_measured_ZZ)[0]

# Save calculated ZZ energies to file
with open(eigenergies_file, 'w') as f:
    yaml.dump(calculated_ZZ_eigenergies, f, default_flow_style=False)
