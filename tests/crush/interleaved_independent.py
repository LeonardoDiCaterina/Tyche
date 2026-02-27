"""Generate and interleave two streams from unrelated parents and run through PractRand."""
import subprocess, sys
import numpy as np

pycmd = r"""
import os, sys, numpy as np
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import io
_old = sys.stderr
sys.stderr = io.StringIO()
import jax, jax.numpy as jnp
from tyche import impl
sys.stderr = _old

parent1 = jax.random.key(1, impl=impl)
parent2 = jax.random.key(2, impl=impl)
child1 = jax.random.split(parent1)[0]
child2 = jax.random.split(parent2)[0]

count_total = {total}
count_half = {half}
chunk = 100000
out = sys.stdout.buffer
written = 0
while written < count_total:
    for k in (child1, child2):
        k, sub = jax.random.split(k)
        cnt = min(chunk, count_half - (written // 2))
        buf = np.array(jax.random.bits(sub, shape=(cnt,), dtype=jnp.uint32), dtype=np.uint32)
        out.write(buf.tobytes()); out.flush()
        written += cnt
""".format(total=1<<30, half=1<<29)

stream_proc = subprocess.Popen([sys.executable, "-c", pycmd], stdout=subprocess.PIPE)
pr = subprocess.Popen(["RNG_test", "stdin32", "-tlmax", "1GB"], stdin=stream_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stream_proc.stdout.close()
out, err = pr.communicate()
print(out.decode() + err.decode())
