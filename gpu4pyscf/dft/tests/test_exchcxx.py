# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Accuracy of the ExchCXX XC backend against libxc.

The backend is chosen at import time from the GDFT_XC_BACKEND environment
variable, so it cannot be switched inside a live process. This file therefore
works in two modes:

  * Run without GDFT_XC_BACKEND (or with it set to 'libxc'): the driver test
    re-executes this same file in a subprocess with GDFT_XC_BACKEND=exchcxx and
    reports its output. This is the mode you get from a plain
    `pytest gpu4pyscf/dft/tests/test_exchcxx.py`.

  * Run with GDFT_XC_BACKEND=exchcxx: the per-functional comparisons run
    in-process against PySCF's CPU libxc, which is the shared reference both
    GPU backends are measured against.

Everything is skipped when lib/deps/lib/libxc_exchcxx.so is absent, i.e. when
the tree was not built with cmake -DBUILD_EXCHCXX=ON.

Reference is CPU libxc rather than the CUDA libxc build, because CPU libxc is
what both GPU backends are expected to reproduce and it is available whichever
backend this process loaded.

ExchCXX and libxc do not share per-functional density cutoffs. Below libxc's
threshold libxc returns exactly 0.0, while ExchCXX evaluates the functional and
returns the analytically correct value. At rho ~ 2e-15 on a real molecular grid
that is a 9e-06 disagreement in exc for LDA_X (= -Cx*rho^(1/3), which decays
only as rho^(1/3)) and ~1e+20 in fxc (v2rho2 ~ rho^(-5/3) genuinely diverges).
Neither is a functional-form error, and neither affects a real calculation:
those points carry grid weight times a density of 1e-15.

So the assertions below compare only grid points above RHO_FLOOR, where both
libraries actually evaluate the functional and agreement is at machine
precision. The unmasked error is still computed and printed on every line, and
test_low_density_only_disagreement asserts that the disagreement really is
confined to low density rather than assuming it. See exchcxx_vs_libxc_repro.py
in the repository root for the underlying analysis.
"""

import os
import subprocess
import sys
import unittest

import numpy as np
import pyscf
from pyscf.dft import Grids
from pyscf.dft.numint import NumInt as numint_cpu

# The name of the shared library the exchcxx backend needs. Its absence means
# the tree was built without -DBUILD_EXCHCXX=ON.
EXCHCXX_SONAME = 'libxc_exchcxx.so'

# Grid points with a total density below this are excluded from the assertions:
# libxc zeroes the functional somewhere under ~1e-14 (the exact value is
# per-functional) while ExchCXX keeps evaluating it. Well above every libxc
# threshold, and still ~5 orders below any density that carries chemistry.
RHO_FLOOR = 1e-10


def _exchcxx_lib_path():
    """Absolute path libxc_exchcxx.so would occupy, whether or not it exists."""
    lib_dir = os.path.abspath(
        os.path.join(__file__, '..', '..', '..', 'lib', 'deps', 'lib'))
    return os.path.join(lib_dir, EXCHCXX_SONAME)


HAVE_EXCHCXX = os.path.exists(_exchcxx_lib_path())
RUNNING_EXCHCXX = os.environ.get('GDFT_XC_BACKEND', 'libxc').lower() == 'exchcxx'


def setUpModule():
    global mol, dm0
    mol = pyscf.M(
        atom='''
C  -0.65830719,  0.61123287, -0.00800148
C   0.73685281,  0.61123287, -0.00800148
''',
        basis='ccpvtz',
        spin=None,
        output='/dev/null',
    )
    np.random.seed(1)
    nao = mol.nao
    mo_coeff = np.random.rand(nao, nao)
    mo_occ = (np.random.rand(nao) > .5).astype(np.double)
    dm0 = (mo_coeff * mo_occ).dot(mo_coeff.T)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


def _diff(dat, ref):
    """Pointwise min(relative, absolute) error, as used by test_libxc.py."""
    d = dat - ref
    with np.errstate(over='ignore', invalid='ignore'):
        rel = abs(d / (ref + 1e-300))
    return np.minimum(np.nan_to_num(rel, nan=np.inf), abs(d))


def _total_rho(rho, spin):
    """Total density per grid point, whatever the xctype/spin packing."""
    rho = np.asarray(rho)
    if spin != 0:
        # (2, ncomp, ngrids) or (2, ngrids)
        return sum(_total_rho(r, 0) for r in rho)
    # LDA is (ngrids,); GGA/MGGA is (ncomp, ngrids) with rho itself at row 0
    return rho if rho.ndim == 1 else rho[0]


def _worst(dat, ref, mask):
    """Largest error over the masked grid points; 0.0 if the mask is empty."""
    err = _diff(np.asarray(dat), np.asarray(ref))
    # Errors are shaped (..., ngrids); broadcast the per-point mask over the
    # leading component axes.
    err = err.reshape(-1, err.shape[-1])[:, mask]
    return float(err.max()) if err.size else 0.0


@unittest.skipUnless(HAVE_EXCHCXX,
                     f'{EXCHCXX_SONAME} not built (cmake -DBUILD_EXCHCXX=ON)')
@unittest.skipUnless(RUNNING_EXCHCXX,
                     'not running under GDFT_XC_BACKEND=exchcxx')
class ExchCXXAccuracy(unittest.TestCase):
    """ExchCXX GPU results vs CPU libxc, one functional per assertion."""

    LABELS = ('exc', 'vxc', 'fxc', 'kxc')

    def _eval_both(self, xc, spin, deriv):
        """Run xc on GPU (ExchCXX) and CPU (libxc) over the same molecular grid.

        Returns (got, ref, dense_mask, on_gpu).
        """
        # Imported lazily so that collection still works on a machine with no
        # GPU: the skip decorators fire before this line is ever reached.
        import cupy
        from gpu4pyscf.dft.numint import NumInt as numint_gpu
        from gpu4pyscf.dft import libxc as gpu_libxc

        self.assertEqual(gpu_libxc.XC_BACKEND, 'exchcxx',
                         'test body ran without the exchcxx backend loaded')

        ni_cpu = numint_cpu()
        ni_gpu = numint_gpu()
        xctype = ni_cpu._xc_type(xc)

        ao_deriv = 0 if xctype == 'LDA' else 1
        grids = Grids(mol).build()
        ao = ni_cpu.eval_ao(mol, grids.coords, ao_deriv)
        rho = ni_cpu.eval_rho(mol, ao, dm0, xctype=xctype)
        if spin != 0:
            rho = (rho, rho)

        ref = ni_cpu.eval_xc_eff(xc, rho, deriv=deriv, xctype=xctype)
        got = ni_gpu.eval_xc_eff(xc, cupy.array(rho), deriv=deriv, xctype=xctype)
        got = [None if g is None else g.get() for g in got]

        # A functional with no ExchCXX kernel falls back to CPU libxc inside
        # eval_xc_eff, which would make this comparison trivially exact and hide
        # a missing kernel. Flag it rather than reporting a false pass.
        xcfuns = ni_gpu._init_xcfuns(xc, spin)
        on_gpu = all(f.on_gpu for f, _ in xcfuns)

        dense = _total_rho(rho, spin) > RHO_FLOOR
        return got, ref, dense, on_gpu

    def _check_xc(self, xc, spin=0, deriv=1,
                  exc_tol=1e-10, vxc_tol=1e-10, fxc_tol=1e-8):
        got, ref, dense, on_gpu = self._eval_both(xc, spin, deriv)

        # An empty or near-empty mask would make every assertion below
        # vacuously true. A default PySCF grid for this molecule has tens of
        # thousands of points; a chemically meaningful fraction sits well above
        # RHO_FLOOR, so anything under a thousand means the mask is broken.
        self.assertGreater(dense.sum(), 1000,
                           f'only {dense.sum()} of {dense.size} grid points are '
                           f'above rho={RHO_FLOOR:g}; the mask is wrong, not '
                           f'the backend')

        tols = (exc_tol, vxc_tol, fxc_tol, None)
        masked, unmasked = {}, {}
        for i in range(deriv + 1):
            if got[i] is None or ref[i] is None:
                continue
            label = self.LABELS[i]
            masked[label] = _worst(got[i], ref[i], dense)
            unmasked[label] = _worst(got[i], ref[i], np.ones_like(dense))

        summary = '  '.join(
            f'{k}={masked[k]:.3e} (all-rho {unmasked[k]:.3e})' for k in masked)
        print(f'[exchcxx] {xc} spin={spin} on_gpu={on_gpu} '
              f'ngrids={dense.sum()}/{dense.size} above rho={RHO_FLOOR:g}\n'
              f'          {summary}', flush=True)

        self.assertTrue(on_gpu,
                        f'{xc} has no ExchCXX kernel; it silently fell back to '
                        f'CPU libxc, so this comparison proves nothing')

        for i in range(deriv + 1):
            label = self.LABELS[i]
            if label not in masked or tols[i] is None:
                continue
            self.assertLess(masked[label], tols[i],
                            f'{xc} spin={spin} {label} exceeds tolerance at '
                            f'rho > {RHO_FLOOR:g} (all-rho worst was '
                            f'{unmasked[label]:.3e})')

    # --- restricted ---------------------------------------------------------

    def test_LDA(self):
        self._check_xc('LDA_C_VWN', deriv=2)

    def test_LDA_X(self):
        self._check_xc('LDA_X', deriv=2)

    def test_GGA_x_b88(self):
        self._check_xc('GGA_X_B88', deriv=2)

    def test_GGA_c_pbe(self):
        # fxc_tol matches test_libxc.py's own value for this functional: the
        # CUDA libxc backend needs 1e-4 here too, so the looseness is a property
        # of PBE correlation's second derivative, not of ExchCXX.
        self._check_xc('GGA_C_PBE', deriv=2, fxc_tol=1e-4)

    def test_GGA_b3lyp(self):
        self._check_xc('HYB_GGA_XC_B3LYP', deriv=2)

    def test_mGGA_c_m06(self):
        # See test_GGA_c_pbe: test_libxc.py also uses 1e-4 for M06 fxc.
        self._check_xc('MGGA_C_M06', deriv=2, fxc_tol=1e-4)

    def test_mGGA_x_tpss(self):
        self._check_xc('MGGA_X_TPSS', deriv=1)

    # --- unrestricted -------------------------------------------------------

    def test_u_LDA(self):
        self._check_xc('LDA_C_VWN', spin=1, deriv=1)

    def test_u_GGA_x_b88(self):
        self._check_xc('GGA_X_B88', spin=1, deriv=1)

    def test_u_GGA_b3lyp(self):
        self._check_xc('HYB_GGA_XC_B3LYP', spin=1, deriv=1)

    def test_u_mGGA_c_m06(self):
        self._check_xc('MGGA_C_M06', spin=1, deriv=1)

    # --- the cutoff difference itself ---------------------------------------

    def test_low_density_only_disagreement(self):
        """The libxc/ExchCXX gap must live entirely below RHO_FLOOR.

        The other tests mask low-density points away, which would also hide a
        real error if one ever appeared there. This asserts the shape of the
        disagreement instead of assuming it: for LDA_X, whose whole all-rho
        error is the cutoff artifact, every point that disagrees by more than
        machine precision must be a point where libxc returned exactly 0.0
        while ExchCXX returned something finite.
        """
        got, ref, dense, on_gpu = self._eval_both('LDA_X', spin=0, deriv=1)
        self.assertTrue(on_gpu)

        exc_gpu, exc_cpu = np.asarray(got[0]), np.asarray(ref[0])
        bad = _diff(exc_gpu, exc_cpu) > 1e-12

        print(f'[exchcxx] LDA_X cutoff check: {bad.sum()} of {bad.size} points '
              f'disagree; all have libxc exc == 0', flush=True)

        if bad.any():
            # Every disagreeing point is one libxc zeroed out ...
            self.assertTrue(np.all(exc_cpu[bad] == 0.0),
                            'ExchCXX disagrees with libxc at a point where '
                            'libxc did NOT zero the functional -- that is a '
                            'real error, not the density-cutoff convention')
            # ... and all of them sit below the floor the other tests apply.
            self.assertTrue(np.all(~dense[bad]),
                            f'a disagreeing point has rho > {RHO_FLOOR:g}; the '
                            f'masking used by the other tests is hiding it')


@unittest.skipUnless(HAVE_EXCHCXX,
                     f'{EXCHCXX_SONAME} not built (cmake -DBUILD_EXCHCXX=ON)')
@unittest.skipIf(RUNNING_EXCHCXX, 'already running under the exchcxx backend')
class ExchCXXSubprocess(unittest.TestCase):
    """Re-run this file with the exchcxx backend selected.

    GDFT_XC_BACKEND is read once, when gpu4pyscf.dft.libxc is imported, so the
    only way to exercise both backends from one pytest invocation is a fresh
    interpreter.
    """

    def test_backend_rejects_unknown_value(self):
        env = dict(os.environ, GDFT_XC_BACKEND='not-a-backend')
        proc = subprocess.run(
            [sys.executable, '-c', 'import gpu4pyscf.dft.libxc'],
            env=env, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0,
                            'an unknown GDFT_XC_BACKEND should not be accepted')
        self.assertIn('GDFT_XC_BACKEND', proc.stderr)

    def test_exchcxx_accuracy_in_subprocess(self):
        env = dict(os.environ, GDFT_XC_BACKEND='exchcxx')
        proc = subprocess.run(
            [sys.executable, '-m', 'unittest', '-v',
             'gpu4pyscf.dft.tests.test_exchcxx.ExchCXXAccuracy'],
            env=env, capture_output=True, text=True)
        # unittest writes progress to stderr and our per-functional lines to
        # stdout; surface both so a failure here is diagnosable from the log.
        print(proc.stdout, flush=True)
        print(proc.stderr, file=sys.stderr, flush=True)
        self.assertEqual(proc.returncode, 0,
                         'ExchCXX accuracy tests failed; see output above')


if __name__ == '__main__':
    print('ExchCXX vs libxc accuracy')
    unittest.main()
