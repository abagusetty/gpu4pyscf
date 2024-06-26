# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

####################################################
#   Example of DFT with ECP
####################################################
import pyscf
from gpu4pyscf.dft import rks

atom = '''
I 0 0 0
I 1 0 0
'''

# def2-qzvpp contains ecp for heavy atoms
mol = pyscf.M(atom=atom, basis='def2-qzvpp', ecp='def2-qzvpp')
mf = rks.RKS(mol, xc='b3lyp').density_fit()
mf.grids.level = 6   # more grids are needed for heavy atoms
e_dft = mf.kernel()

# gradient and Hessian of ECP are also supported
# but ECP contributions are still calculated on CPU
g = mf.nuc_grad_method()
grad = g.kernel()

h = mf.Hessian()
hess = h.kernel()
