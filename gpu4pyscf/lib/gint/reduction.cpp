/* Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

template <int blockx, int blocky>
__attribute__((always_inline)) static void block_reduce_x(double val, double *addr, int tx, int ty, sycl::nd_item<2>& item){
    sycl::group thread_block = item.get_group();
    using tile_t             = double[blockx*blocky];
    tile_t& sdata = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);    
    sdata[tx*blocky+ty] = val; item.barrier(sycl::access::fence_space::local_space);
    if (blockx >= 32) if (tx < 16) sdata[tx*blocky+ty] += sdata[(tx+16)*blocky+ty]; item.barrier(sycl::access::fence_space::local_space);
    if (blockx >= 16) if (tx < 8)  sdata[tx*blocky+ty] += sdata[(tx+8)*blocky+ty];  item.barrier(sycl::access::fence_space::local_space);
    if (blockx >= 8)  if (tx < 4)  sdata[tx*blocky+ty] += sdata[(tx+4)*blocky+ty];  item.barrier(sycl::access::fence_space::local_space);
    if (blockx >= 4)  if (tx < 2)  sdata[tx*blocky+ty] += sdata[(tx+2)*blocky+ty];  item.barrier(sycl::access::fence_space::local_space);
    if (blockx >= 2)  if (tx < 1)  sdata[tx*blocky+ty] += sdata[(tx+1)*blocky+ty];  item.barrier(sycl::access::fence_space::local_space);
    if (tx == 0) {
	sycl::atomic_ref<double, sycl::memory_order::relaxed,
			 sycl::memory_scope::device, sycl::access::address_space::global_space> ref(val);
	ref.fetch_add(sdata[ty]);
    }
}

template <int blockx, int blocky>
__attribute__((always_inline)) static void block_reduce_y(double val, double *addr, int tx, int ty, sycl::nd_item<2>& item){
    /*
    if(blocky >= 32) sdata[tx*blocky+ty] += sdata[tx*blocky+ty+16];
    if(blocky >= 16) sdata[tx*blocky+ty] += sdata[tx*blocky+ty+8];
    if(blocky >= 8)  sdata[tx*blocky+ty] += sdata[tx*blocky+ty+4];
    if(blocky >= 4)  sdata[tx*blocky+ty] += sdata[tx*blocky+ty+2];
    if(blocky >= 2)  sdata[tx*blocky+ty] += sdata[tx*blocky+ty+1];
    */
    int stride = blocky + 1;

    sycl::group thread_block = item.get_group();
    using tile_t             = double[blockx*(blocky+1)];
    tile_t& sdata = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);    
    sdata[tx*stride+ty] = val; item.barrier(sycl::access::fence_space::local_space);
    if (blocky >= 32) if (ty < 16) sdata[tx*stride+ty] += sdata[tx*stride+ty+16]; item.barrier(sycl::access::fence_space::local_space);
    if (blocky >= 16) if (ty < 8)  sdata[tx*stride+ty] += sdata[tx*stride+ty+8];  item.barrier(sycl::access::fence_space::local_space);
    if (blocky >= 8)  if (ty < 4)  sdata[tx*stride+ty] += sdata[tx*stride+ty+4];  item.barrier(sycl::access::fence_space::local_space);
    if (blocky >= 4)  if (ty < 2)  sdata[tx*stride+ty] += sdata[tx*stride+ty+2];  item.barrier(sycl::access::fence_space::local_space);
    if (blocky >= 2)  if (ty < 1)  sdata[tx*stride+ty] += sdata[tx*stride+ty+1];  item.barrier(sycl::access::fence_space::local_space);
    if (ty == 0) {
	sycl::atomic_ref<double, sycl::memory_order::relaxed,
			 sycl::memory_scope::device, sycl::access::address_space::global_space> ref(val);
	ref.fetch_add(sdata[tx*stride]);
    }
 }
