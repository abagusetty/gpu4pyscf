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

Tolerances here are looser than in test_libxc.py. ExchCXX and libxc do not
share per-functional density cutoffs: below libxc's threshold libxc returns
exactly 0.0 while ExchCXX returns the analytically correct small value, so the
relative error at those grid points is 1.0 by construction. The error metric
below is min(relative, absolute), which turns those points into a small
absolute difference instead. See exchcxx_vs_libxc_repro.py in the repository
root for the full analysis.
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
    return np.min((abs(d / (ref + 1e-300)), abs(d)), axis=0)


@unittest.skipUnless(HAVE_EXCHCXX,
                     f'{EXCHCXX_SONAME} not built (cmake -DBUILD_EXCHCXX=ON)')
@unittest.skipUnless(RUNNING_EXCHCXX,
                     'not running under GDFT_XC_BACKEND=exchcxx')
class ExchCXXAccuracy(unittest.TestCase):
    """ExchCXX GPU results vs CPU libxc, one functional per assertion."""

    def _check_xc(self, xc, spin=0, deriv=1,
                  exc_tol=1e-5, vxc_tol=1e-5, fxc_tol=1e-3):
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

        # A functional with no ExchCXX kernel falls back to CPU libxc inside
        # eval_xc_eff, which would make this comparison trivially exact and hide
        # a missing kernel. Flag it rather than reporting a false pass.
        xcfuns = ni_gpu._init_xcfuns(xc, spin)
        on_gpu = all(f.on_gpu for f, _ in xcfuns)

        labels = ('exc', 'vxc', 'fxc', 'kxc')
        tols = (exc_tol, vxc_tol, fxc_tol, None)
        worst = {}
        for i in range(deriv + 1):
            if got[i] is None or ref[i] is None:
                continue
            err = _diff(got[i].get(), ref[i]).max()
            worst[labels[i]] = err

        summary = '  '.join(f'{k}={v:.3e}' for k, v in worst.items())
        print(f'[exchcxx] {xc} spin={spin} on_gpu={on_gpu}  {summary}',
              flush=True)

        self.assertTrue(on_gpu,
                        f'{xc} has no ExchCXX kernel; it silently fell back to '
                        f'CPU libxc, so this comparison proves nothing')

        for i in range(deriv + 1):
            if labels[i] not in worst or tols[i] is None:
                continue
            self.assertLess(worst[labels[i]], tols[i],
                            f'{xc} spin={spin} {labels[i]} exceeds tolerance')

    # --- restricted ---------------------------------------------------------

    def test_LDA(self):
        self._check_xc('LDA_C_VWN', deriv=2)

    def test_LDA_X(self):
        self._check_xc('LDA_X', deriv=2)

    def test_GGA_x_b88(self):
        self._check_xc('GGA_X_B88', deriv=2)

    def test_GGA_c_pbe(self):
        self._check_xc('GGA_C_PBE', deriv=2, fxc_tol=1e-2)

    def test_GGA_b3lyp(self):
        self._check_xc('HYB_GGA_XC_B3LYP', deriv=2, fxc_tol=1e-2)

    def test_mGGA_c_m06(self):
        self._check_xc('MGGA_C_M06', deriv=2, fxc_tol=1e-2)

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
