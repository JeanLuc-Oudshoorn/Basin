"""List and read millions of small source files without seeking for each one.

A corpus of small Parquet files on a hard disk is seek-bound: visited one by one
in name order, every table costs several head movements, and name order need not
follow the order in which the files were written. The two helpers here keep the
drive streaming instead.

`list_directory` reports, where the platform offers it for free (Windows), each
entry's position in the volume's file table. Visiting files in that order turns
millions of scattered metadata lookups into one forward sweep.

`read_files` reads a whole window of files at once. On Windows it asks the file
system where each file's pieces are stored and requests them in ascending disk
position from a few threads. Elsewhere it reads the files concurrently.

Only the order of the requests differs. Every file is read completely, through
its own handle and by file offset, so the returned bytes are the file contents
whatever the file system reports about their placement.
"""

import os
import sys
import threading

_DONE = object()


def list_directory(path):
    """Yield (name, is_directory, record) for each entry of a directory.

    `is_directory` does not follow links, like `DirEntry.is_dir(follow_symlinks=
    False)`. `record` orders entries by their place in the volume's file table,
    or is None when the platform does not report it while listing.
    """
    path = os.fspath(path)
    if sys.platform == "win32":
        entries = _list_with_records(path)
        try:
            first = next(entries, _DONE)
        except OSError:
            first = None  # this file system cannot list with records
        if first is not None:
            if first is not _DONE:
                yield first
                yield from entries
            return
    with os.scandir(path) as entries:
        for entry in entries:
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError:
                is_directory = False
            yield entry.name, is_directory, None


def read_files(paths, threads=16, order=None):
    """Return the contents of every path, in the order given.

    `order` optionally holds one sortable value per path (the `record` of its
    folder from `list_directory`); files are then opened in that order.
    """
    paths = [os.fspath(path) for path in paths]
    if not paths:
        return []
    sequence = range(len(paths))
    if order is not None:
        sequence = sorted(sequence, key=order.__getitem__)
    if sys.platform == "win32":
        return _read_in_disk_order(paths, sequence, threads)
    contents = [None] * len(paths)

    def read(index):
        with open(paths[index], "rb") as stream:
            contents[index] = stream.read()

    visit(read, sequence, threads)
    return contents


def visit(function, items, threads):
    """Call function on every item from a few threads, dispatching in order."""
    iterator, lock, errors = iter(items), threading.Lock(), []

    def work():
        try:
            while not errors:
                with lock:
                    item = next(iterator, _DONE)
                if item is _DONE:
                    return
                function(item)
        except BaseException as error:  # re-raised in the calling thread
            errors.append(error)

    workers = [
        threading.Thread(target=work, daemon=True)
        for _ in range(max(1, min(threads, len(items))))
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if errors:
        raise errors[0]


if sys.platform == "win32":
    import ctypes
    import struct
    from ctypes import wintypes

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.GetFileSizeEx.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_longlong),
    ]
    _kernel32.GetFileSizeEx.restype = wintypes.BOOL
    _kernel32.GetFileInformationByHandleEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
    _kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(_Overlapped),
    ]
    _kernel32.ReadFile.restype = wintypes.BOOL
    _kernel32.DeviceIoControl.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    _kernel32.DeviceIoControl.restype = wintypes.BOOL
    _kernel32.GetVolumePathNameW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    _kernel32.GetVolumePathNameW.restype = wintypes.BOOL
    _kernel32.GetDiskFreeSpaceW.argtypes = [wintypes.LPCWSTR] + [
        ctypes.POINTER(wintypes.DWORD)
    ] * 4
    _kernel32.GetDiskFreeSpaceW.restype = wintypes.BOOL

    _INVALID_HANDLE = wintypes.HANDLE(-1).value
    _GENERIC_READ = 0x80000000
    _FILE_LIST_DIRECTORY = 0x1
    _SHARE_ALL = 0x7
    _OPEN_EXISTING = 3
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000  # required to open a directory
    # No read-ahead: the requests below already arrive in disk order.
    _FILE_FLAG_RANDOM_ACCESS = 0x10000000
    _FILE_ATTRIBUTE_DIRECTORY = 0x10
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x400
    _NAME_SURROGATE = 0x20000000  # reparse tags of links and junctions
    _FILE_ID_BOTH_DIRECTORY_INFO = 10
    _FSCTL_GET_RETRIEVAL_POINTERS = 0x00090073
    _ERROR_NO_MORE_FILES = 18
    _ERROR_HANDLE_EOF = 38
    _LISTING_BUFFER = 1 << 20
    _EXTENT_BUFFER = 4096
    _MAX_REQUEST = 1 << 30
    # FILE_ID_BOTH_DIR_INFO: next entry, attributes, name bytes, reparse tag,
    # file id; the name follows at byte 104.
    _ENTRY = struct.Struct("<I52xIII28xq")
    _RECORD_MASK = (1 << 48) - 1  # the upper bits count reuses of the record

    def _failure(path):
        error = ctypes.WinError(ctypes.get_last_error())
        error.filename = path
        return error

    def _extended(path):
        """Lift the legacy 260-character limit for long absolute paths."""
        if len(path) < 248 or path.startswith("\\\\?\\"):
            return path
        path = os.path.abspath(path)
        if path.startswith("\\\\"):
            return "\\\\?\\UNC\\" + path[2:]
        return "\\\\?\\" + path

    def _open(path, access=_GENERIC_READ, flags=_FILE_FLAG_RANDOM_ACCESS):
        handle = _kernel32.CreateFileW(
            _extended(path), access, _SHARE_ALL, None, _OPEN_EXISTING, flags, None
        )
        if handle == _INVALID_HANDLE:
            raise _failure(path)
        return handle

    def _list_with_records(path):
        handle = _open(path, _FILE_LIST_DIRECTORY, _FILE_FLAG_BACKUP_SEMANTICS)
        try:
            buffer = ctypes.create_string_buffer(_LISTING_BUFFER)
            while True:
                if not _kernel32.GetFileInformationByHandleEx(
                    handle, _FILE_ID_BOTH_DIRECTORY_INFO, buffer, _LISTING_BUFFER
                ):
                    if ctypes.get_last_error() == _ERROR_NO_MORE_FILES:
                        return
                    raise _failure(path)
                raw, offset = buffer.raw, 0
                while True:
                    following, attributes, length, tag, record = _ENTRY.unpack_from(
                        raw, offset
                    )
                    name = raw[offset + 104 : offset + 104 + length].decode(
                        "utf-16-le", "surrogatepass"
                    )
                    if name not in (".", ".."):
                        link = (
                            attributes & _FILE_ATTRIBUTE_REPARSE_POINT
                            and tag & _NAME_SURROGATE
                        )
                        yield (
                            name,
                            bool(attributes & _FILE_ATTRIBUTE_DIRECTORY and not link),
                            record & _RECORD_MASK,
                        )
                    if not following:
                        break
                    offset += following
        finally:
            _kernel32.CloseHandle(handle)

    def _cluster_size(path):
        """Bytes per allocation unit of the volume holding path, or None."""
        volume = ctypes.create_unicode_buffer(1024)
        if not _kernel32.GetVolumePathNameW(os.path.abspath(path), volume, 1024):
            return None
        sectors, sector_bytes = wintypes.DWORD(), wintypes.DWORD()
        free, total = wintypes.DWORD(), wintypes.DWORD()
        if not _kernel32.GetDiskFreeSpaceW(
            volume.value,
            ctypes.byref(sectors),
            ctypes.byref(sector_bytes),
            ctypes.byref(free),
            ctypes.byref(total),
        ):
            return None
        return sectors.value * sector_bytes.value or None

    def _pieces(handle, size, cluster):
        """(disk position, file offset, length) requests that tile [0, size)."""
        whole = [(-1, 0, size)]
        if cluster is None:
            return whole
        start = ctypes.c_longlong(0)
        output = ctypes.create_string_buffer(_EXTENT_BUFFER)
        returned = wintypes.DWORD()
        if not _kernel32.DeviceIoControl(
            handle,
            _FSCTL_GET_RETRIEVAL_POINTERS,
            ctypes.byref(start),
            8,
            output,
            _EXTENT_BUFFER,
            ctypes.byref(returned),
            None,
        ):
            # Stored inside its file record, too fragmented for the buffer, or
            # not a file system that reports placement: one ordinary read.
            return whole
        raw = output.raw
        count, first = struct.unpack_from("<I4xq", raw, 0)
        pieces, position = [], 0
        for index in range(count):
            following, location = struct.unpack_from("<qq", raw, 16 + 16 * index)
            offset, stop = first * cluster, min(following * cluster, size)
            first = following
            if offset >= size:
                break
            if location < 0 or offset != position or stop <= offset:
                return whole  # sparse or compressed: placement is not usable
            pieces.append((location, offset, stop - offset))
            position = stop
        return pieces if position == size else whole

    def _read_piece(handle, buffer, offset, length, path):
        target = (ctypes.c_char * length).from_buffer(buffer, offset)
        received, done = wintypes.DWORD(), 0
        while done < length:
            position = _Overlapped()
            position.Offset = (offset + done) & 0xFFFFFFFF
            position.OffsetHigh = (offset + done) >> 32
            if not _kernel32.ReadFile(
                handle,
                ctypes.byref(target, done),
                min(length - done, _MAX_REQUEST),
                ctypes.byref(received),
                ctypes.byref(position),
            ):
                if ctypes.get_last_error() == _ERROR_HANDLE_EOF:
                    break
                raise _failure(path)
            if not received.value:
                break
            done += received.value
        if done != length:
            raise OSError(f"File changed while it was being read: {path}")

    def _read_in_disk_order(paths, sequence, threads):
        count = len(paths)
        cluster = _cluster_size(paths[0])
        handles, buffers, requests = [None] * count, [None] * count, [None] * count

        def prepare(index):
            handle = handles[index] = _open(paths[index])
            size = ctypes.c_longlong()
            if not _kernel32.GetFileSizeEx(handle, ctypes.byref(size)):
                raise _failure(paths[index])
            buffers[index] = bytearray(size.value)
            requests[index] = [
                (location, index, offset, length)
                for location, offset, length in _pieces(handle, size.value, cluster)
                if length
            ]

        def read(request):
            _, index, offset, length = request
            _read_piece(handles[index], buffers[index], offset, length, paths[index])

        try:
            visit(prepare, sequence, threads)
            visit(
                read, sorted(piece for pieces in requests for piece in pieces), threads
            )
        finally:
            for handle in handles:
                if handle is not None:
                    _kernel32.CloseHandle(handle)
        return buffers
