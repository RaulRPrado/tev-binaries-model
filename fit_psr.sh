#!/bin/bash

source "/afs/ifh.de/group/cta/scratch/prado/set_python3.sh"
source "/afs/ifh.de/group/cta/scratch/prado/VTS/HESS_J0632/tev-binaries-model/set_tgb-lib.sh"

python3 applications/fit_pulsar_model.py $1
