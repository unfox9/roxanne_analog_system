import numpy as np


class BasisProjector:

    def __init__(self, schema, basis_specs):
        self.action_dim = schema.action_dim
        self.basis_specs = basis_specs
        self.B = self._build_B(schema, basis_specs)

    def _build_B(self, schema, basis_specs):
        cols = []
        for gname, axis in basis_specs:
            v = np.zeros((schema.action_dim,), dtype=np.float32)
            for c in schema.get(gname):
                if axis is None or str(c.get("axis", "")).upper() == axis:
                    v[int(c["index"])] = 1.0
            n = np.linalg.norm(v)
            if n > 1e-6:
                v /= n
            cols.append(v)
        B = np.stack(cols, axis=1).astype(np.float32)
        return B

    @property
    def coeff_dim(self) -> int:
        return self.B.shape[1]

    def __call__(self, coeff: np.ndarray) -> np.ndarray:
        z = np.asarray(coeff, dtype=np.float32).reshape(-1)
        return (self.B @ z).astype(np.float32)
