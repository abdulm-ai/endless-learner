import os
import sys

# NumPy reads its thread count at import, so set it before anything imports NumPy.
threads = "1"
if "--threads" in sys.argv:
    threads = sys.argv[sys.argv.index("--threads") + 1]
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, threads)

from .cli import main  # noqa: E402

main()
