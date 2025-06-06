/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define BLKSIZEX        32
#define BLKSIZEY        32

__global__
void GDFTcontract_rho_kernel(double *rho, double *bra, double *ket, int ngrids, int nao);
__global__
void GDFTscale_ao_kernel(double *out, double *ket, double *wv,
                         int ngrids, int nao, int nvar);
