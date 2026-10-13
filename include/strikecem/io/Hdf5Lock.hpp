#pragma once
#include <mutex>

namespace strikecem {

// Global recursive mutex to serialize HDF5 C/C++ library operations in multi-threaded contexts,
// protecting global hid_t tables, memory managers, and error stacks on platforms where
// libhdf5 is not built with --enable-threadsafe.
inline std::recursive_mutex& hdf5_global_mutex() {
    static std::recursive_mutex mtx;
    return mtx;
}

} // namespace strikecem
