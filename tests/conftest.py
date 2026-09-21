import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


@pytest.fixture(scope="session")
def key():
    return jax.random.PRNGKey(0)
