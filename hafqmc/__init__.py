import scipy.signal as _signal
del _signal # just to avoid import failure of jax

from jax.config import config as _jax_config
_jax_config.update("jax_enable_x64", True)