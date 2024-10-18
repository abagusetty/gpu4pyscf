/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
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

#include "sycl_device.hpp"

template <typename T>
void MALLOC(void*& var, size_t size) {
    var = sycl::malloc_device<T>(size, *(sycl_get_queue()));
}

void FREE(void* ptr) {
    sycl::free(ptr, *sycl_get_queue());
}
void MEMSET(void* addr, int value, size_t size) {
    sycl_get_queue()->memset(addr, value, size).wait();
}

template <typename T>
void DEVICE_INIT(void*& dst, const void* src, size_t size) {
    MALLOC<T>(dst, size);
    sycl_get_queue()->memcpy(dst, src, sizeof(T) * (size)).wait();
}
