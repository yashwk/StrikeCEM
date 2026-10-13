from __future__ import annotations
import ctypes
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union, Any

from .bindings import load_bindings, ScemQueryResult, ScemQueryRequest
from .errors import raise_for_status, DesignMismatchError

@dataclass(frozen=True)
class QueryResult:
    scattering_real: float
    scattering_imag: float
    rcs_sqm: float
    rcs_dbsm: float

    @property
    def scattering(self) -> complex:
        return complex(self.scattering_real, self.scattering_imag)

@dataclass(frozen=True)
class QueryRequest:
    azimuth_deg: float
    elevation_deg: float
    frequency_hz: float
    polarization: str

class Database:
    """High-performance StrikeCEM RCS database reader and query engine."""

    def __init__(self, handle: ctypes.c_void_p, lib: ctypes.CDLL, path: str):
        self._handle = handle
        self._lib = lib
        self._path = path
        self._closed = False

    @classmethod
    def open(cls, path: str, lib_path: Optional[str] = None) -> Database:
        """Opens a .scem, .scemdb, or .h5 database file."""
        lib = load_bindings(lib_path)
        handle = ctypes.c_void_p()
        status = lib.scem_database_open(path.encode("utf-8"), ctypes.byref(handle))
        if status != 0:
            err_msg = lib.scem_get_last_error()
            msg = err_msg.decode("utf-8") if err_msg else f"Failed to open database {path}"
            raise_for_status(status, msg)
        return cls(handle, lib, path)

    def close(self) -> None:
        """Closes database and releases native memory."""
        if not self._closed and self._handle:
            self._lib.scem_database_close(self._handle)
            self._handle = ctypes.c_void_p()
            self._closed = True

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _check_open(self) -> None:
        if self._closed or not self._handle:
            raise RuntimeError("Database is closed.")

    @property
    def design_id(self) -> str:
        self._check_open()
        buf = ctypes.create_string_buffer(256)
        status = self._lib.scem_database_get_design_id(self._handle, buf, len(buf))
        raise_for_status(status)
        return buf.value.decode("utf-8")

    @property
    def design_revision(self) -> str:
        self._check_open()
        buf = ctypes.create_string_buffer(256)
        status = self._lib.scem_database_get_design_revision(self._handle, buf, len(buf))
        raise_for_status(status)
        return buf.value.decode("utf-8")

    def verify_design(self, expected_id: str, expected_rev: str) -> bool:
        """Verifies design ID and revision. Returns True on match, raises DesignMismatchError on mismatch."""
        self._check_open()
        status = self._lib.scem_database_verify_design(
            self._handle,
            expected_id.encode("utf-8"),
            expected_rev.encode("utf-8")
        )
        if status == 5:
            raise DesignMismatchError(
                f"Design mismatch: expected ({expected_id}, {expected_rev}), found ({self.design_id}, {self.design_revision})"
            )
        raise_for_status(status)
        return True

    @property
    def frequencies(self) -> List[float]:
        self._check_open()
        count = ctypes.c_size_t(0)
        # First call gets count
        self._lib.scem_database_get_frequencies(self._handle, None, ctypes.byref(count))
        if count.value == 0:
            return []
        arr = (ctypes.c_double * count.value)()
        status = self._lib.scem_database_get_frequencies(self._handle, arr, ctypes.byref(count))
        raise_for_status(status)
        return [float(arr[i]) for i in range(count.value)]

    @property
    def polarizations(self) -> List[str]:
        self._check_open()
        count = ctypes.c_size_t(0)
        # First call gets count
        self._lib.scem_database_get_polarizations(self._handle, None, ctypes.byref(count))
        if count.value == 0:
            return []
        ElemType = ctypes.c_char * 8
        ArrType = ElemType * count.value
        arr = ArrType()
        status = self._lib.scem_database_get_polarizations(self._handle, ctypes.cast(arr, ctypes.c_void_p), ctypes.byref(count))
        raise_for_status(status)
        return [arr[i].value.decode("utf-8") for i in range(count.value)]

    def query(self, az_deg: float, el_deg: float, freq_hz: float, pol: str) -> QueryResult:
        """Queries interpolated RCS at a given angle, frequency, and polarization channel."""
        self._check_open()
        res = ScemQueryResult()
        status = self._lib.scem_database_query(
            self._handle,
            float(az_deg),
            float(el_deg),
            float(freq_hz),
            pol.encode("utf-8"),
            ctypes.byref(res),
        )
        if status != 0:
            err = self._lib.scem_get_last_error()
            msg = err.decode("utf-8") if err else f"Query failed for pol '{pol}'"
            raise_for_status(status, msg)
        return QueryResult(
            scattering_real=res.scattering_real,
            scattering_imag=res.scattering_imag,
            rcs_sqm=res.rcs_sqm,
            rcs_dbsm=res.rcs_dbsm,
        )

    def query_batch(self, requests: Sequence[Union[QueryRequest, tuple]]) -> List[QueryResult]:
        """Queries multiple trajectory points in a single high-throughput vectorized C call."""
        self._check_open()
        n = len(requests)
        if n == 0:
            return []

        ReqArray = ScemQueryRequest * n
        ResArray = ScemQueryResult * n
        c_reqs = ReqArray()
        c_res = ResArray()

        for i, r in enumerate(requests):
            if isinstance(r, QueryRequest):
                c_reqs[i].azimuth_deg = float(r.azimuth_deg)
                c_reqs[i].elevation_deg = float(r.elevation_deg)
                c_reqs[i].frequency_hz = float(r.frequency_hz)
                c_reqs[i].polarization = r.polarization.encode("utf-8")[:7]
            else:
                c_reqs[i].azimuth_deg = float(r[0])
                c_reqs[i].elevation_deg = float(r[1])
                c_reqs[i].frequency_hz = float(r[2])
                c_reqs[i].polarization = str(r[3]).encode("utf-8")[:7]

        status = self._lib.scem_database_query_batch(
            self._handle,
            c_reqs,
            n,
            c_res,
        )
        if status != 0:
            err = self._lib.scem_get_last_error()
            msg = err.decode("utf-8") if err else "Batch query failed"
            raise_for_status(status, msg)

        return [
            QueryResult(
                scattering_real=c_res[i].scattering_real,
                scattering_imag=c_res[i].scattering_imag,
                rcs_sqm=c_res[i].rcs_sqm,
                rcs_dbsm=c_res[i].rcs_dbsm,
            )
            for i in range(n)
        ]

    def query_batch_numpy(
        self,
        az_deg_array: Any,
        el_deg_array: Any,
        freq_hz_array: Any,
        polarization: Union[str, Sequence[str]]
    ) -> dict:
        """Batch query interfacing directly with NumPy arrays or sequences."""
        try:
            import numpy as np
        except ImportError:
            np = None

        if hasattr(az_deg_array, "__len__"):
            n = len(az_deg_array)
        else:
            raise TypeError("az_deg_array must be a sequence or numpy array")

        if isinstance(polarization, str):
            pols = [polarization] * n
        else:
            pols = list(polarization)

        reqs = [
            QueryRequest(
                azimuth_deg=float(az_deg_array[i]),
                elevation_deg=float(el_deg_array[i]),
                frequency_hz=float(freq_hz_array[i] if hasattr(freq_hz_array, "__getitem__") else freq_hz_array),
                polarization=pols[i]
            )
            for i in range(n)
        ]
        results = self.query_batch(reqs)

        sqm = [r.rcs_sqm for r in results]
        dbsm = [r.rcs_dbsm for r in results]
        real = [r.scattering_real for r in results]
        imag = [r.scattering_imag for r in results]

        if np is not None:
            return {
                "rcs_sqm": np.array(sqm, dtype=np.float64),
                "rcs_dbsm": np.array(dbsm, dtype=np.float64),
                "scattering_real": np.array(real, dtype=np.float64),
                "scattering_imag": np.array(imag, dtype=np.float64),
                "scattering": np.array(real, dtype=np.float64) + 1j * np.array(imag, dtype=np.float64),
            }
        return {
            "rcs_sqm": sqm,
            "rcs_dbsm": dbsm,
            "scattering_real": real,
            "scattering_imag": imag,
        }
