import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Optional

class ScemQueryResult(ctypes.Structure):
    _fields_ = [
        ("scattering_real", ctypes.c_double),
        ("scattering_imag", ctypes.c_double),
        ("rcs_sqm", ctypes.c_double),
        ("rcs_dbsm", ctypes.c_double),
    ]

class ScemQueryRequest(ctypes.Structure):
    _fields_ = [
        ("azimuth_deg", ctypes.c_double),
        ("elevation_deg", ctypes.c_double),
        ("frequency_hz", ctypes.c_double),
        ("polarization", ctypes.c_char * 8),
    ]

def find_library(custom_path: Optional[str] = None) -> str:
    """Finds the StrikeCEM shared library."""
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    env_path = os.environ.get("STRIKECEM_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Look relative to repository root if working from source
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        repo_root / "build" / "release" / "libStrikeCEM.so",
        repo_root / "build" / "libStrikeCEM.so",
        repo_root / "build" / "release" / "libStrikeCEM.dylib",
        repo_root / "build" / "StrikeCEM.dll",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    # Standard system lookup
    system_lib = ctypes.util.find_library("StrikeCEM")
    if system_lib:
        return system_lib

    raise FileNotFoundError(
        "Could not find libStrikeCEM shared library. Set STRIKECEM_LIB_PATH environment variable."
    )

def load_bindings(lib_path: Optional[str] = None) -> ctypes.CDLL:
    path = find_library(lib_path)
    lib = ctypes.CDLL(path)

    # Function prototypes
    lib.scem_database_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.scem_database_open.restype = ctypes.c_int

    lib.scem_database_close.argtypes = [ctypes.c_void_p]
    lib.scem_database_close.restype = None

    lib.scem_database_get_design_id.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.scem_database_get_design_id.restype = ctypes.c_int

    lib.scem_database_get_design_revision.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.scem_database_get_design_revision.restype = ctypes.c_int

    lib.scem_database_verify_design.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.scem_database_verify_design.restype = ctypes.c_int

    lib.scem_database_get_frequencies.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.scem_database_get_frequencies.restype = ctypes.c_int

    # char out_pols[][8] -> pointer to array of (c_char * 8)
    lib.scem_database_get_polarizations.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.scem_database_get_polarizations.restype = ctypes.c_int

    lib.scem_database_query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_char_p,
        ctypes.POINTER(ScemQueryResult),
    ]
    lib.scem_database_query.restype = ctypes.c_int

    lib.scem_database_query_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ScemQueryRequest),
        ctypes.c_size_t,
        ctypes.POINTER(ScemQueryResult),
    ]
    lib.scem_database_query_batch.restype = ctypes.c_int

    lib.scem_get_last_error.argtypes = []
    lib.scem_get_last_error.restype = ctypes.c_char_p

    return lib
