from __future__ import annotations

from collections.abc import Callable
import ctypes
import errno
import os
from pathlib import Path
import sys
import threading
from typing import Protocol


class MemoryReadError(RuntimeError):
    """Raised when a requested process-memory span cannot be read exactly."""


class ProcessIdentityError(MemoryReadError):
    """Raised when a PID does not identify the expected process lifetime."""


class AddressReader(Protocol):
    @property
    def access_trace(self) -> tuple[tuple[int, int], ...]: ...

    def pread_exact(self, address: int, size: int) -> bytes: ...

    def pread_many_exact(
        self, spans: tuple[tuple[int, int], ...]
    ) -> tuple[bytes, ...]: ...


class _VectorProcessReader(Protocol):
    def read_many_exact(
        self,
        pid: int,
        spans: tuple[tuple[int, int], ...],
    ) -> tuple[bytes, ...]: ...


class _ProcessVmReadvReader:
    def __init__(
        self,
        *,
        read_batch: Callable[[int, tuple[tuple[int, int], ...]], bytes],
        max_iov_count: int,
    ) -> None:
        self._read_batch = read_batch
        self._max_iov_count = max_iov_count

    def read_many_exact(
        self,
        pid: int,
        spans: tuple[tuple[int, int], ...],
    ) -> tuple[bytes, ...]:
        values: list[bytes] = []
        for offset in range(0, len(spans), self._max_iov_count):
            batch = spans[offset : offset + self._max_iov_count]
            payload = self._read_batch(pid, batch)
            expected_size = sum(size for _, size in batch)
            if len(payload) != expected_size:
                raise MemoryReadError(
                    "process memory vector short read: "
                    f"expected {expected_size}, got {len(payload)}"
                )
            payload_offset = 0
            for _, size in batch:
                next_offset = payload_offset + size
                values.append(payload[payload_offset:next_offset])
                payload_offset = next_offset
        return tuple(values)


class _IOVec(ctypes.Structure):
    _fields_ = (
        ("iov_base", ctypes.c_void_p),
        ("iov_len", ctypes.c_size_t),
    )


class _ProcessVmReadvCall:
    def __init__(self, *, function) -> None:
        self._function = function

    def __call__(
        self,
        pid: int,
        spans: tuple[tuple[int, int], ...],
    ) -> bytes:
        total_size = sum(size for _, size in spans)
        local_buffer = ctypes.create_string_buffer(total_size)
        local_iov = (_IOVec * 1)(
            _IOVec(ctypes.addressof(local_buffer), total_size)
        )
        remote_iov = (_IOVec * len(spans))(
            *(_IOVec(address, size) for address, size in spans)
        )
        transferred = self._function(
            pid,
            local_iov,
            1,
            remote_iov,
            len(spans),
            0,
        )
        if transferred < 0:
            error_number = ctypes.get_errno() or errno.EIO
            raise OSError(error_number, os.strerror(error_number))
        if transferred > total_size:
            raise MemoryReadError("process memory vector read exceeded its buffer")
        return local_buffer.raw[:transferred]


class ProcessMemoryReader:
    """Read verified spans from one stable Linux process identity."""

    def __init__(
        self,
        *,
        pid: int,
        expected_start_time_ticks: int,
        proc_root: Path | str = Path("/proc"),
    ) -> None:
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ProcessIdentityError("pid must be a positive integer")
        if (
            isinstance(expected_start_time_ticks, bool)
            or not isinstance(expected_start_time_ticks, int)
            or expected_start_time_ticks <= 0
        ):
            raise ProcessIdentityError(
                "expected_start_time_ticks must be a positive integer"
            )
        self.pid = pid
        self.expected_start_time_ticks = expected_start_time_ticks
        self.proc_root = Path(proc_root)
        self._process_root = self.proc_root / str(pid)
        self._descriptor = -1
        self._fallback_lock = threading.Lock()
        self._access_trace: list[tuple[int, int]] = []
        self._vector_reader = _load_process_vm_readv_reader(self.proc_root)

        self._verify_identity()
        flags = os.O_RDONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self._descriptor = os.open(self._process_root / "mem", flags)
        except OSError as error:
            raise MemoryReadError("process memory cannot be opened") from error
        try:
            self._verify_identity()
        except Exception:
            self.close()
            raise

    def __enter__(self) -> ProcessMemoryReader:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def pread_exact(self, address: int, size: int) -> bytes:
        return self.pread_many_exact(((address, size),))[0]

    def pread_many_exact(
        self, spans: tuple[tuple[int, int], ...]
    ) -> tuple[bytes, ...]:
        if self._descriptor < 0:
            raise MemoryReadError("process memory reader is closed")
        if not isinstance(spans, tuple) or not spans:
            raise MemoryReadError("read round must contain at least one span")

        normalized: list[tuple[int, int, int]] = []
        for item in spans:
            if not isinstance(item, tuple) or len(item) != 2:
                raise MemoryReadError("read span must be an address-size pair")
            address, size = item
            if (
                isinstance(address, bool)
                or not isinstance(address, int)
                or address < 0
            ):
                raise MemoryReadError("address must be a non-negative integer")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise MemoryReadError("read size must be positive")
            end = address + size
            if end <= address:
                raise MemoryReadError("read span overflows the address space")
            normalized.append((address, size, end))

        self._verify_identity()
        mappings = self._read_readable_mappings()
        if any(
            not _span_is_within_readable_mapping(address, end, mappings)
            for address, _, end in normalized
        ):
            raise MemoryReadError("read span is not within one readable mapping")

        if self._vector_reader is not None:
            read_trace = tuple((address, size) for address, size, _ in normalized)
            self._access_trace.extend(read_trace)
            try:
                values = self._vector_reader.read_many_exact(self.pid, read_trace)
            except OSError as error:
                raise MemoryReadError("process memory vector read failed") from error
            if len(values) != len(read_trace) or any(
                len(value) != size
                for value, (_, size) in zip(values, read_trace, strict=True)
            ):
                raise MemoryReadError("process memory vector read was incomplete")
            self._verify_identity()
            return tuple(values)

        values: list[bytes] = []
        for address, size, _ in normalized:
            try:
                value = self._pread(address, size)
            except OSError as error:
                raise MemoryReadError("process memory pread failed") from error
            if len(value) != size:
                raise MemoryReadError(
                    f"process memory short read: expected {size}, got {len(value)}"
                )
            values.append(value)
        self._verify_identity()
        return tuple(values)

    @property
    def access_trace(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._access_trace)

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def _pread(self, address: int, size: int) -> bytes:
        self._access_trace.append((address, size))
        if hasattr(os, "pread"):
            return os.pread(self._descriptor, size, address)
        # Windows uses this path only for portable fake-/proc unit tests.
        with self._fallback_lock:
            os.lseek(self._descriptor, address, os.SEEK_SET)
            return os.read(self._descriptor, size)

    def _verify_identity(self) -> None:
        actual = _read_process_start_time(self._process_root / "stat")
        if actual != self.expected_start_time_ticks:
            raise ProcessIdentityError(
                "process start time does not match the expected identity"
            )

    def _read_readable_mappings(self) -> tuple[tuple[int, int], ...]:
        try:
            lines = (self._process_root / "maps").read_text(
                encoding="utf-8", errors="strict"
            ).splitlines()
        except (OSError, UnicodeError) as error:
            raise MemoryReadError("process maps cannot be read") from error
        mappings: list[tuple[int, int]] = []
        for line in lines:
            fields = line.split(maxsplit=2)
            if len(fields) < 2 or "r" not in fields[1][:1]:
                continue
            try:
                range_start_text, range_end_text = fields[0].split("-", 1)
                range_start = int(range_start_text, 16)
                range_end = int(range_end_text, 16)
            except (ValueError, IndexError) as error:
                raise MemoryReadError("process maps contains an invalid row") from error
            mappings.append((range_start, range_end))
        return tuple(mappings)


def _load_process_vm_readv_reader(
    proc_root: Path,
) -> _VectorProcessReader | None:
    if sys.platform != "linux" or proc_root != Path("/proc"):
        return None
    try:
        library = ctypes.CDLL(None, use_errno=True)
        function = library.process_vm_readv
    except (AttributeError, OSError):
        return None
    function.argtypes = (
        ctypes.c_int,
        ctypes.POINTER(_IOVec),
        ctypes.c_ulong,
        ctypes.POINTER(_IOVec),
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    function.restype = ctypes.c_ssize_t
    try:
        max_iov_count = int(os.sysconf("SC_IOV_MAX"))
    except (AttributeError, OSError, TypeError, ValueError):
        max_iov_count = 1024
    if max_iov_count <= 0:
        max_iov_count = 1024
    max_iov_count = min(max_iov_count, 1024)
    return _ProcessVmReadvReader(
        read_batch=_ProcessVmReadvCall(function=function),
        max_iov_count=max_iov_count,
    )


def _span_is_within_readable_mapping(
    start: int,
    end: int,
    mappings: tuple[tuple[int, int], ...],
) -> bool:
    return any(
        range_start <= start and end <= range_end
        for range_start, range_end in mappings
    )


def _read_process_start_time(stat_path: Path) -> int:
    try:
        text = stat_path.read_text(encoding="utf-8", errors="strict").strip()
    except (OSError, UnicodeError) as error:
        raise ProcessIdentityError("process stat cannot be read") from error
    closing_parenthesis = text.rfind(")")
    if closing_parenthesis < 0:
        raise ProcessIdentityError("process stat has no command terminator")
    fields_from_state = text[closing_parenthesis + 1 :].split()
    if len(fields_from_state) <= 19:
        raise ProcessIdentityError("process stat is missing start time")
    try:
        start_time_ticks = int(fields_from_state[19])
    except ValueError as error:
        raise ProcessIdentityError("process start time is invalid") from error
    if start_time_ticks <= 0:
        raise ProcessIdentityError("process start time must be positive")
    return start_time_ticks
