from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
from dataclasses import dataclass
from typing import Protocol


_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_BINARY = getattr(os, "O_BINARY", 0)
_PROC_PERMISSION_PATTERN = re.compile(r"[r-][w-][x-][ps]")
_ELF_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
_PROGRAM_HEADER = struct.Struct("<IIQQQQQQ")
_SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")
_SYMBOL = struct.Struct("<IBBHQQ")
_NOTE_HEADER = struct.Struct("<III")
_PT_LOAD = 1
_SHT_NOTE = 7
_SHT_DYNSYM = 11
_STB_GLOBAL = 1
_STT_FUNC = 2
_SHN_UNDEF = 0
_NT_GNU_BUILD_ID = 3


class ElfInspectionError(RuntimeError):
    """Raised when a target ELF cannot be identified unambiguously."""


class ProcessInspectionError(RuntimeError):
    """Raised when live process provenance cannot be pinned safely."""


@dataclass(frozen=True)
class FileIdentity:
    device_major: int
    device_minor: int
    inode: int
    path_display: str
    digest: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "device_major": self.device_major,
            "device_minor": self.device_minor,
            "inode": self.inode,
            "path_display": self.path_display,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class ProcessIdentity:
    host_boot_id: str
    pid: int
    start_time_ticks: int
    live_executable: FileIdentity


@dataclass(frozen=True)
class ElfSymbolIdentity:
    name: str
    symbol_value: int
    resolved_file_offset: int
    elf_build_id: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "symbol_value": self.symbol_value,
            "resolved_file_offset": self.resolved_file_offset,
        }


@dataclass(frozen=True)
class MappedElfIdentity:
    owner_pid: int
    owner_start_time_ticks: int
    device_major: int
    device_minor: int
    inode: int
    path_display: str
    attachment_path: str
    digest: str
    elf_build_id: str
    llama_decode_symbol: ElfSymbolIdentity

    def file_mapping(self) -> dict[str, object]:
        return {
            "device_major": self.device_major,
            "device_minor": self.device_minor,
            "inode": self.inode,
            "path_display": self.path_display,
            "digest": self.digest,
            "elf_build_id": self.elf_build_id,
        }


@dataclass(frozen=True)
class NativeDependency:
    role: str
    identity: FileIdentity
    elf_build_id: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "role": self.role,
            **self.identity.to_mapping(),
            "elf_build_id": self.elf_build_id,
        }


@dataclass(frozen=True)
class ProcMapEntry:
    start: int
    end: int
    permissions: str
    file_offset: int
    device_major: int
    device_minor: int
    inode: int
    path_display: str

    @property
    def length_bytes(self) -> int:
        return self.end - self.start

    @property
    def identity_key(self) -> tuple[int, int, int]:
        return (self.device_major, self.device_minor, self.inode)


@dataclass(frozen=True)
class PinnedFile:
    fd: int
    identity: FileIdentity


class ProcessInspectionBackend(Protocol):
    def read_boot_id(self) -> str: ...
    def read_start_time_ticks(self, pid: int) -> int: ...
    def open_executable(self, pid: int) -> PinnedFile: ...
    def read_maps(self, pid: int) -> tuple[ProcMapEntry, ...]: ...
    def open_mapping(self, pid: int, mapping: ProcMapEntry) -> PinnedFile: ...
    def list_tids(self, pid: int) -> tuple[int, ...]: ...
    def read_thread_state(self, pid: int, tid: int) -> tuple[int, str]: ...


def _device_numbers(device: int) -> tuple[int, int]:
    if hasattr(os, "major") and hasattr(os, "minor"):
        return (os.major(device), os.minor(device))
    return (0, int(device))


def _sha256_fd(fd: int) -> str:
    try:
        original_offset = os.lseek(fd, 0, os.SEEK_CUR)
    except OSError:
        original_offset = 0
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(fd, original_offset, os.SEEK_SET)
    return "sha256:" + digest.hexdigest()


def _pin_fd(fd: int, path_display: str) -> PinnedFile:
    result = os.fstat(fd)
    if not stat.S_ISREG(result.st_mode):
        raise ProcessInspectionError("pinned process object is not a regular file")
    device_major, device_minor = _device_numbers(result.st_dev)
    return PinnedFile(
        fd=fd,
        identity=FileIdentity(
            device_major=device_major,
            device_minor=device_minor,
            inode=int(result.st_ino),
            path_display=path_display,
            digest=_sha256_fd(fd),
        ),
    )


def _parse_start_time(stat_text: str) -> int:
    closing = stat_text.rfind(")")
    if closing < 0:
        raise ProcessInspectionError("malformed proc stat record")
    fields = stat_text[closing + 1 :].strip().split()
    if len(fields) <= 19:
        raise ProcessInspectionError("proc stat record lacks start_time_ticks")
    try:
        value = int(fields[19], 10)
    except ValueError as error:
        raise ProcessInspectionError("invalid process start_time_ticks") from error
    if value <= 0:
        raise ProcessInspectionError("process start_time_ticks must be positive")
    return value


def parse_proc_maps(text: str) -> tuple[ProcMapEntry, ...]:
    entries: list[ProcMapEntry] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        parts = line.split(maxsplit=5)
        if len(parts) < 5:
            raise ProcessInspectionError(
                f"malformed proc maps record at line {line_number}"
            )
        address, permissions, offset, device, inode_text = parts[:5]
        path_display = parts[5] if len(parts) == 6 else ""
        if _PROC_PERMISSION_PATTERN.fullmatch(permissions) is None:
            raise ProcessInspectionError(
                f"invalid proc maps permissions at line {line_number}"
            )
        try:
            start_text, end_text = address.split("-", 1)
            major_text, minor_text = device.split(":", 1)
            start = int(start_text, 16)
            end = int(end_text, 16)
            file_offset = int(offset, 16)
            device_major = int(major_text, 16)
            device_minor = int(minor_text, 16)
            inode = int(inode_text, 10)
        except (ValueError, TypeError) as error:
            raise ProcessInspectionError(
                f"invalid proc maps number at line {line_number}"
            ) from error
        if start < 0 or end <= start or file_offset < 0 or inode < 0:
            raise ProcessInspectionError(
                f"invalid proc maps range at line {line_number}"
            )
        entries.append(
            ProcMapEntry(
                start=start,
                end=end,
                permissions=permissions,
                file_offset=file_offset,
                device_major=device_major,
                device_minor=device_minor,
                inode=inode,
                path_display=path_display,
            )
        )
    if not entries:
        raise ProcessInspectionError("proc maps record is empty")
    return tuple(entries)


class SystemProcessInspectionBackend:
    def __init__(self, proc_root: Path | str = "/proc") -> None:
        self.proc_root = Path(proc_root)

    def read_boot_id(self) -> str:
        value = (self.proc_root / "sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        if not value:
            raise ProcessInspectionError("host boot ID is empty")
        return value

    def read_start_time_ticks(self, pid: int) -> int:
        value = (self.proc_root / str(pid) / "stat").read_text(
            encoding="ascii"
        )
        return _parse_start_time(value)

    def open_executable(self, pid: int) -> PinnedFile:
        path = self.proc_root / str(pid) / "exe"
        display = os.readlink(path)
        fd = os.open(path, os.O_RDONLY | _O_CLOEXEC | _O_BINARY)
        try:
            return _pin_fd(fd, display)
        except BaseException:
            os.close(fd)
            raise

    def read_maps(self, pid: int) -> tuple[ProcMapEntry, ...]:
        text = (self.proc_root / str(pid) / "maps").read_text(
            encoding="utf-8", errors="strict"
        )
        return parse_proc_maps(text)

    def open_mapping(self, pid: int, mapping: ProcMapEntry) -> PinnedFile:
        path = (
            self.proc_root
            / str(pid)
            / "map_files"
            / f"{mapping.start:x}-{mapping.end:x}"
        )
        fd = os.open(path, os.O_RDONLY | _O_CLOEXEC | _O_BINARY)
        try:
            return _pin_fd(fd, mapping.path_display)
        except BaseException:
            os.close(fd)
            raise

    def list_tids(self, pid: int) -> tuple[int, ...]:
        task_root = self.proc_root / str(pid) / "task"
        tids = tuple(
            sorted(
                int(item.name)
                for item in task_root.iterdir()
                if item.name.isdecimal()
            )
        )
        if not tids:
            raise ProcessInspectionError("target thread group is empty")
        return tids

    def read_thread_state(self, pid: int, tid: int) -> tuple[int, str]:
        text = (self.proc_root / str(pid) / "task" / str(tid) / "stat").read_text(
            encoding="ascii"
        )
        closing = text.rfind(")")
        if closing < 0:
            raise ProcessInspectionError("malformed thread stat record")
        fields = text[closing + 1 :].strip().split()
        if len(fields) <= 19 or len(fields[0]) != 1:
            raise ProcessInspectionError("thread stat record is incomplete")
        return (_parse_start_time(text), fields[0])


def trusted_file_key(identity: FileIdentity) -> tuple[int, int, int, str]:
    return (
        identity.device_major,
        identity.device_minor,
        identity.inode,
        identity.digest,
    )


def _require_pid(pid: int) -> None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("pid must be a positive integer")


def read_process_identity(
    pid: int,
    *,
    backend: ProcessInspectionBackend | None = None,
) -> ProcessIdentity:
    _require_pid(pid)
    active_backend = backend or SystemProcessInspectionBackend()
    before = active_backend.read_start_time_ticks(pid)
    pinned = active_backend.open_executable(pid)
    try:
        executable = pinned.identity
    finally:
        os.close(pinned.fd)
    confirmation = active_backend.open_executable(pid)
    try:
        confirmed_executable = confirmation.identity
    finally:
        os.close(confirmation.fd)
    after = active_backend.read_start_time_ticks(pid)
    if before != after:
        raise ProcessInspectionError("process identity changed during inspection")
    if trusted_file_key(executable) != trusted_file_key(confirmed_executable):
        raise ProcessInspectionError("process executable changed during inspection")
    return ProcessIdentity(
        host_boot_id=active_backend.read_boot_id(),
        pid=pid,
        start_time_ticks=before,
        live_executable=executable,
    )


def _read_fd(fd: int) -> bytes:
    try:
        original_offset = os.lseek(fd, 0, os.SEEK_CUR)
    except OSError:
        original_offset = 0
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    os.lseek(fd, original_offset, os.SEEK_SET)
    return b"".join(chunks)


def _slice(image: bytes, offset: int, size: int, subject: str) -> bytes:
    if offset < 0 or size < 0 or offset > len(image) or size > len(image) - offset:
        raise ElfInspectionError(f"{subject} lies outside the ELF image")
    return image[offset : offset + size]


def _parse_build_id(image: bytes, sections: tuple[tuple[int, ...], ...]) -> str:
    build_ids: set[bytes] = set()
    for section in sections:
        section_type = section[1]
        if section_type != _SHT_NOTE:
            continue
        data = _slice(image, section[4], section[5], "ELF note section")
        cursor = 0
        while cursor < len(data):
            if len(data) - cursor < _NOTE_HEADER.size:
                raise ElfInspectionError("truncated ELF note header")
            name_size, description_size, note_type = _NOTE_HEADER.unpack_from(
                data, cursor
            )
            cursor += _NOTE_HEADER.size
            name = _slice(data, cursor, name_size, "ELF note name")
            cursor += (name_size + 3) & ~3
            description = _slice(
                data, cursor, description_size, "ELF note description"
            )
            cursor += (description_size + 3) & ~3
            if note_type == _NT_GNU_BUILD_ID and name.rstrip(b"\x00") == b"GNU":
                if not description:
                    raise ElfInspectionError("ELF GNU build ID is empty")
                build_ids.add(description)
    if len(build_ids) != 1:
        raise ElfInspectionError("exactly one ELF GNU build ID is required")
    return next(iter(build_ids)).hex()


def _cstring(table: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(table):
        raise ElfInspectionError("ELF string offset is outside its table")
    end = table.find(b"\x00", offset)
    if end < 0:
        raise ElfInspectionError("unterminated ELF string")
    try:
        return table[offset:end].decode("ascii")
    except UnicodeDecodeError as error:
        raise ElfInspectionError("ELF symbol name is not ASCII") from error


def _parse_elf(
    fd: int,
    *,
    symbol_name: str | None,
) -> tuple[str, ElfSymbolIdentity | None]:
    image = _read_fd(fd)
    if len(image) < _ELF_HEADER.size:
        raise ElfInspectionError("truncated ELF header")
    header = _ELF_HEADER.unpack_from(image)
    ident = header[0]
    if ident[:4] != b"\x7fELF" or ident[4:7] != bytes((2, 1, 1)):
        raise ElfInspectionError("only little-endian ELF64 version 1 is supported")
    if header[2] != 62:
        raise ElfInspectionError("only x86_64 ELF objects are supported")
    program_offset, section_offset = header[5], header[6]
    header_size, program_size, program_count = header[8], header[9], header[10]
    section_size, section_count = header[11], header[12]
    if header_size != _ELF_HEADER.size:
        raise ElfInspectionError("unexpected ELF header size")
    if program_size != _PROGRAM_HEADER.size or section_size != _SECTION_HEADER.size:
        raise ElfInspectionError("unexpected ELF table entry size")
    programs = tuple(
        _PROGRAM_HEADER.unpack(
            _slice(
                image,
                program_offset + index * program_size,
                program_size,
                "ELF program header",
            )
        )
        for index in range(program_count)
    )
    sections = tuple(
        _SECTION_HEADER.unpack(
            _slice(
                image,
                section_offset + index * section_size,
                section_size,
                "ELF section header",
            )
        )
        for index in range(section_count)
    )
    build_id = _parse_build_id(image, sections)
    if symbol_name is None:
        return (build_id, None)
    candidates: list[tuple[int, int]] = []
    for section in sections:
        if section[1] != _SHT_DYNSYM:
            continue
        linked_index = section[6]
        entry_size = section[9]
        if linked_index >= len(sections) or entry_size != _SYMBOL.size:
            raise ElfInspectionError("invalid ELF dynamic symbol section")
        symbol_bytes = _slice(
            image, section[4], section[5], "ELF dynamic symbol section"
        )
        if len(symbol_bytes) % entry_size:
            raise ElfInspectionError("misaligned ELF dynamic symbol section")
        string_section = sections[linked_index]
        string_table = _slice(
            image, string_section[4], string_section[5], "ELF string table"
        )
        for offset in range(0, len(symbol_bytes), entry_size):
            symbol = _SYMBOL.unpack_from(symbol_bytes, offset)
            if _cstring(string_table, symbol[0]) != symbol_name:
                continue
            binding = symbol[1] >> 4
            symbol_type = symbol[1] & 0x0F
            visibility = symbol[2] & 0x03
            if (
                binding == _STB_GLOBAL
                and symbol_type == _STT_FUNC
                and visibility == 0
                and symbol[3] != _SHN_UNDEF
            ):
                candidates.append((symbol[4], symbol[5]))
    if len(candidates) != 1:
        raise ElfInspectionError(
            f"exactly one eligible {symbol_name} dynamic symbol is required"
        )
    symbol_value, _ = candidates[0]
    resolved_offsets: list[int] = []
    for program in programs:
        program_type, _, file_offset, virtual_address, _, file_size, memory_size, _ = program
        if program_type != _PT_LOAD:
            continue
        if virtual_address <= symbol_value < virtual_address + memory_size:
            resolved = file_offset + symbol_value - virtual_address
            if resolved < file_offset + file_size and resolved < len(image):
                resolved_offsets.append(resolved)
    if len(resolved_offsets) != 1:
        raise ElfInspectionError("ELF symbol file offset is ambiguous or unmapped")
    return (
        build_id,
        ElfSymbolIdentity(
            name=symbol_name,
            symbol_value=symbol_value,
            resolved_file_offset=resolved_offsets[0],
            elf_build_id=build_id,
        ),
    )


def inspect_elf(fd: int, symbol_name: str = "llama_decode") -> ElfSymbolIdentity:
    if isinstance(fd, bool) or not isinstance(fd, int) or fd < 0:
        raise ValueError("fd must be a non-negative integer")
    if not isinstance(symbol_name, str) or not symbol_name:
        raise ValueError("symbol_name must be non-empty")
    _, symbol = _parse_elf(fd, symbol_name=symbol_name)
    assert symbol is not None
    return symbol


def _basename(path_display: str) -> str:
    cleaned = path_display.removesuffix(" (deleted)")
    return PurePosixPath(cleaned).name


def _single_object_entries(
    entries: tuple[ProcMapEntry, ...],
    *,
    predicate: object,
    description: str,
) -> tuple[ProcMapEntry, ...]:
    selected = tuple(
        entry
        for entry in entries
        if entry.inode > 0 and callable(predicate) and predicate(_basename(entry.path_display))
    )
    object_keys = {entry.identity_key for entry in selected}
    if len(object_keys) != 1:
        raise ProcessInspectionError(f"exactly one {description} object is required")
    return tuple(entry for entry in selected if entry.identity_key in object_keys)


def _open_verified_mapping(
    backend: ProcessInspectionBackend,
    pid: int,
    entries: tuple[ProcMapEntry, ...],
) -> PinnedFile:
    executable = tuple(entry for entry in entries if entry.permissions[2] == "x")
    selected = min(executable or entries, key=lambda entry: entry.start)
    pinned = backend.open_mapping(pid, selected)
    if trusted_file_key(pinned.identity)[:3] != selected.identity_key:
        os.close(pinned.fd)
        raise ProcessInspectionError(
            "opened mapping identity does not match proc maps device and inode"
        )
    return pinned


def inspect_single_mapped_libllama(
    pid: int,
    *,
    backend: ProcessInspectionBackend | None = None,
) -> MappedElfIdentity:
    _require_pid(pid)
    active_backend = backend or SystemProcessInspectionBackend()
    before = active_backend.read_start_time_ticks(pid)
    entries = _single_object_entries(
        active_backend.read_maps(pid),
        predicate=lambda name: (
            name == "libllama.so" or name.startswith("libllama.so.")
        ),
        description="mapped libllama",
    )
    executable = tuple(entry for entry in entries if entry.permissions[2] == "x")
    attachment_mapping = min(executable or entries, key=lambda entry: entry.start)
    pinned = _open_verified_mapping(active_backend, pid, entries)
    try:
        symbol = inspect_elf(pinned.fd)
        identity = pinned.identity
    finally:
        os.close(pinned.fd)
    confirmed_mappings = active_backend.read_maps(pid)
    expected_key = (
        identity.device_major,
        identity.device_minor,
        identity.inode,
    )
    if not any(
        entry.identity_key == expected_key and entry.permissions[2] == "x"
        for entry in confirmed_mappings
    ):
        raise ProcessInspectionError("mapped libllama mapping changed during inspection")
    after = active_backend.read_start_time_ticks(pid)
    if before != after:
        raise ProcessInspectionError("process identity changed during inspection")
    return MappedElfIdentity(
        owner_pid=pid,
        owner_start_time_ticks=before,
        device_major=identity.device_major,
        device_minor=identity.device_minor,
        inode=identity.inode,
        path_display=identity.path_display,
        attachment_path=(
            f"/proc/{pid}/map_files/"
            f"{attachment_mapping.start:x}-{attachment_mapping.end:x}"
        ),
        digest=identity.digest,
        elf_build_id=symbol.elf_build_id,
        llama_decode_symbol=symbol,
    )


def _dependency_role(name: str) -> str | None:
    if name.startswith("libbpf.so"):
        return "libbpf"
    if name.startswith("libelf.so") or re.fullmatch(
        r"libelf-[0-9]+(?:\.[0-9]+)*\.so", name
    ):
        return "libelf"
    if name.startswith("libz.so") and not name.startswith("libzstd"):
        return "zlib"
    if name.startswith("libc.so"):
        return "libc"
    if name.startswith("ld-linux"):
        return "loader"
    return None


def collect_native_dependencies(
    pid: int,
    *,
    backend: ProcessInspectionBackend | None = None,
) -> tuple[NativeDependency, ...]:
    _require_pid(pid)
    active_backend = backend or SystemProcessInspectionBackend()
    before = active_backend.read_start_time_ticks(pid)
    grouped: dict[str, dict[tuple[int, int, int], list[ProcMapEntry]]] = {}
    for entry in active_backend.read_maps(pid):
        role = _dependency_role(_basename(entry.path_display))
        if role is not None and entry.inode > 0:
            grouped.setdefault(role, {}).setdefault(entry.identity_key, []).append(entry)
    required = ("libbpf", "libelf", "zlib", "libc", "loader")
    missing = tuple(role for role in required if role not in grouped)
    if missing:
        raise ProcessInspectionError(
            "missing dependency roles: " + ", ".join(missing)
        )
    ambiguous = tuple(role for role in required if len(grouped[role]) != 1)
    if ambiguous:
        raise ProcessInspectionError(
            "ambiguous dependency roles: " + ", ".join(ambiguous)
        )
    dependencies: list[NativeDependency] = []
    for role in required:
        entries = tuple(next(iter(grouped[role].values())))
        pinned = _open_verified_mapping(active_backend, pid, entries)
        try:
            build_id, _ = _parse_elf(pinned.fd, symbol_name=None)
            identity = pinned.identity
        finally:
            os.close(pinned.fd)
        dependencies.append(
            NativeDependency(role=role, identity=identity, elf_build_id=build_id)
        )
    after = active_backend.read_start_time_ticks(pid)
    if before != after:
        raise ProcessInspectionError("process identity changed during inspection")
    return tuple(dependencies)
