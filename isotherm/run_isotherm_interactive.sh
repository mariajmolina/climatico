#!/bin/bash
set -euo pipefail

# ---------------------------
# 0. Load environment
# ---------------------------
module load conda
module load pyferret
# module load ncarenv

# Install simple env
conda env create -f env_isotherm.yml # <--- comment out after install!

# Activate your conda env (needs xarray, netCDF4, dask)
conda activate isotherming

# ---------------------------
# 1. Preprocessing
# ---------------------------
echo ">>> Running preprocessing..."
python <<'EOF'
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebook = "pyferret_file_creation.ipynb"
with open(notebook) as f:
    nb = nbformat.read(f, as_version=4)
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": "."}})
EOF

# ---------------------------
# 2. Ferret loop (for a few test files)
# ---------------------------
echo ">>> Running Ferret isotherm extraction..."

# Limit to first 2 files for quick testing
for file in $(ls file_*.nc | head -n 2); do
    outfile="iso20c_${file}"
    jnlfile="${file}.jnl"

    cat > "${jnlfile}" <<EOT
use "${file}"
let depth_of_20c = if kmt le 0 then (-1.e34) else missing(temp[k=@loc:20],0)
let depth_of_18c = if kmt le 0 then (-1.e34) else missing(temp[k=@loc:18],0)
let depth_of_15c = if kmt le 0 then (-1.e34) else missing(temp[k=@loc:15],0)
let depth_of_12c = if kmt le 0 then (-1.e34) else missing(temp[k=@loc:12],0)
let depth_of_10c = if kmt le 0 then (-1.e34) else missing(temp[k=@loc:10],0)
let depth_of_8c  = if kmt le 0 then (-1.e34) else missing(temp[k=@loc:8],0)
let depth_of_5c  = if kmt le 0 then (-1.e34) else missing(temp[k=@loc:5],0)
save/file="${outfile}" depth_of_20c, depth_of_18c, depth_of_15c, depth_of_12c, depth_of_10c, depth_of_8c, depth_of_5c
quit
EOT

    echo ">>> Processing ${file} -> ${outfile}"
    pyferret < "${jnlfile}"
    rm "${jnlfile}"
done

# ---------------------------
# 3. Merge test outputs
# ---------------------------
echo ">>> Merging outputs..."
python <<'EOF'
import xarray as xr

ds = xr.open_mfdataset("iso20c_file_*.nc", combine="by_coords")
ds.to_netcdf("iso20c_test.nc")
print("Wrote iso20c_test.nc")
EOF

echo ">>> Done (test run complete)!"
