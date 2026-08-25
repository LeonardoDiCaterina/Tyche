import pytest
import jax
import jax.numpy as jnp
import tyche.backend_cuda

@pytest.mark.skipif(
    not jax.default_backend() == "gpu",
    reason="Requires GPU backend"
)
def test_dummy_fill_ffi():
    # Try to execute the custom call
    # The dummy fill kernel fills the array with 42
    size = 100
    
    @jax.jit
    def run_dummy():
        return tyche.backend_cuda.dummy_fill(size)
    
    out = run_dummy()
    
    assert out.shape == (100,)
    assert out.dtype == jnp.uint32
    assert jnp.all(out == 42)
