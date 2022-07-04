try:
    import cornet
except ModuleNotFoundError:
    import sys
    import subprocess
    python = sys.executable
    missing ["git+https://github.com/dicarlolab/CORnet"]
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

from cornet import cornet_z, cornet_r, cornet_rt, cornet_s
