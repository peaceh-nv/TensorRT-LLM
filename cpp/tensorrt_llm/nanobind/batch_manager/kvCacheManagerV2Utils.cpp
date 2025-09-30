/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kvCacheManagerV2Utils.h"
#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

void KVCacheManagerV2UtilsBindings::initBindings(nb::module_& module)
{
    // Bind DiskAddress struct
    nb::class_<DiskAddress>(module, "DiskAddress")
        .def(nb::init<int, ssize_t>(), nb::arg("fd"), nb::arg("pos"))
        .def_rw("fd", &DiskAddress::fd)
        .def_rw("pos", &DiskAddress::pos);

    // Bind Task template instantiations
    nb::class_<Task<DiskAddress, DiskAddress>>(module, "DiskToDiskTask")
        .def(nb::init<DiskAddress, DiskAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<DiskAddress, DiskAddress>::dst)
        .def_rw("src", &Task<DiskAddress, DiskAddress>::src);

    nb::class_<Task<MemAddress, DiskAddress>>(module, "DiskToHostTask")
        .def(nb::init<MemAddress, DiskAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<MemAddress, DiskAddress>::dst)
        .def_rw("src", &Task<MemAddress, DiskAddress>::src);

    nb::class_<Task<DiskAddress, MemAddress>>(module, "HostToDiskTask")
        .def(nb::init<DiskAddress, MemAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<DiskAddress, MemAddress>::dst)
        .def_rw("src", &Task<DiskAddress, MemAddress>::src);

    // Bind copy functions
    module.def(
        "copy_disk_to_disk",
        [](std::vector<Task<DiskAddress, DiskAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDiskToDisk(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"),
        "Copy data from disk to disk using CUDA host function");

    module.def(
        "copy_disk_to_host",
        [](std::vector<Task<MemAddress, DiskAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDiskToHost(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"),
        "Copy data from disk to host using CUDA host function");

    module.def(
        "copy_host_to_disk",
        [](std::vector<Task<DiskAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToDisk(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"),
        "Copy data from host to disk using CUDA host function");
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
