"""Master script: generate data, then run NAS experiment.

Usage: python -m master
"""

from datagen.generate import main as generate_data
from NAS.silent_reasoning import main as run_nas


print('=' * 60)
print('STEP 1: DATA GENERATION')
print('=' * 60)
generate_data()

print('\n' + '=' * 60)
print('STEP 2: NAS EXPERIMENT')
print('=' * 60)
run_nas()
