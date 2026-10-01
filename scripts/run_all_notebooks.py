from pathlib import Path
import subprocess, sys

root=Path(__file__).resolve().parents[1]
notebooks=sorted((root/'notebooks').glob('*.ipynb'))
for nb in notebooks:
    print(f'Executing {nb.name} ...', flush=True)
    cmd=[sys.executable,'-m','jupyter','nbconvert','--to','notebook','--execute',
         '--ExecutePreprocessor.timeout=180','--inplace',str(nb)]
    subprocess.run(cmd,cwd=root,check=True)
print('All notebooks executed successfully.')
