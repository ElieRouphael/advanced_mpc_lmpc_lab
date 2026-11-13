from pathlib import Path
import os, subprocess, sys

root=Path(__file__).resolve().parents[1]
notebooks=sorted((root/'notebooks').glob('*.ipynb'))

# The QPs here have at most a few hundred variables. Multi-threaded BLAS gains nothing at that
# size and can make each linear solve 100x slower (thread start-up and contention), which shows
# up in the solve times printed by the notebooks. Use one BLAS thread unless the caller chose.
env=os.environ.copy()
for var in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    env.setdefault(var,'1')
env['PYTHONPATH']=str(root/'src')+os.pathsep+env.get('PYTHONPATH','')

for nb in notebooks:
    print(f'Executing {nb.name} ...', flush=True)
    cmd=[sys.executable,'-m','jupyter','nbconvert','--to','notebook','--execute',
         '--ExecutePreprocessor.timeout=600','--inplace',str(nb)]
    subprocess.run(cmd,cwd=root,check=True,env=env)
print('All notebooks executed successfully.')
