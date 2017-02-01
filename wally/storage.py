"""
This module contains interfaces for storage classes
"""

import os
import re
import abc
import shutil
import sqlite3
import logging
from typing import Any, TypeVar, Type, IO, Tuple, cast, List, Dict, Iterable, Iterator

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper  # type: ignore
except ImportError:
    from yaml import Loader, Dumper  # type: ignore
import numpy

from .common_types import IStorable
from .utils import shape2str, str2shape


logger = logging.getLogger("wally")


class ISimpleStorage(metaclass=abc.ABCMeta):
    """interface for low-level storage, which doesn't support serialization
    and can operate only on bytes"""

    @abc.abstractmethod
    def put(self, value: bytes, path: str) -> None:
        pass

    @abc.abstractmethod
    def get(self, path: str) -> bytes:
        pass

    @abc.abstractmethod
    def rm(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def sync(self) -> None:
        pass

    @abc.abstractmethod
    def __contains__(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def get_fd(self, path: str, mode: str = "rb+") -> IO:
        pass

    @abc.abstractmethod
    def sub_storage(self, path: str) -> 'ISimpleStorage':
        pass

    @abc.abstractmethod
    def list(self, path: str) -> Iterator[Tuple[bool, str]]:
        pass


class ISerializer(metaclass=abc.ABCMeta):
    """Interface for serialization class"""
    @abc.abstractmethod
    def pack(self, value: IStorable) -> bytes:
        pass

    @abc.abstractmethod
    def unpack(self, data: bytes) -> Any:
        pass


class DBStorage(ISimpleStorage):

    create_tb_sql = "CREATE TABLE IF NOT EXISTS wally_storage (key text, data blob, type text)"
    insert_sql = "INSERT INTO wally_storage VALUES (?, ?, ?)"
    update_sql = "UPDATE wally_storage SET data=?, type=? WHERE key=?"
    select_sql = "SELECT data, type FROM wally_storage WHERE key=?"
    contains_sql = "SELECT 1 FROM wally_storage WHERE key=?"
    rm_sql = "DELETE FROM wally_storage WHERE key LIKE '{}%'"
    list2_sql = "SELECT key, length(data), type FROM wally_storage"
    SQLITE3_THREADSAFE = 1

    def __init__(self, db_path: str = None, existing: bool = False,
                 prefix: str = None, db: sqlite3.Connection = None) -> None:

        assert not prefix or "'" not in prefix, "Broken sql prefix {!r}".format(prefix)

        if db_path:
            self.existing = existing
            if existing:
                if not os.path.isfile(db_path):
                    raise IOError("No storage found at {!r}".format(db_path))

            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            if sqlite3.threadsafety != self.SQLITE3_THREADSAFE:
                raise RuntimeError("Sqlite3 compiled without threadsafe support, can't use DB storage on it")

            try:
                self.db = sqlite3.connect(db_path, check_same_thread=False)
            except sqlite3.OperationalError as exc:
                raise IOError("Can't open database at {!r}".format(db_path)) from exc

            self.db.execute(self.create_tb_sql)
        else:
            if db is None:
                raise ValueError("Either db or db_path parameter must be passed")
            self.db = db

        if prefix is None:
            self.prefix = ""
        elif not prefix.endswith('/'):
            self.prefix = prefix + '/'
        else:
            self.prefix = prefix

    def put(self, value: bytes, path: str) -> None:
        c = self.db.cursor()
        fpath = self.prefix + path
        c.execute(self.contains_sql, (fpath,))
        if len(c.fetchall()) == 0:
            c.execute(self.insert_sql, (fpath, value, 'yaml'))
        else:
            c.execute(self.update_sql, (value, 'yaml', fpath))

    def get(self, path: str) -> bytes:
        c = self.db.cursor()
        c.execute(self.select_sql, (self.prefix + path,))
        res = cast(List[Tuple[bytes, str]], c.fetchall())  # type: List[Tuple[bytes, str]]
        if not res:
            raise KeyError(path)
        assert len(res) == 1
        val, tp = res[0]
        assert tp == 'yaml'
        return val

    def rm(self, path: str) -> None:
        c = self.db.cursor()
        path = self.prefix + path
        assert "'" not in path, "Broken sql path {!r}".format(path)
        c.execute(self.rm_sql.format(path))

    def __contains__(self, path: str) -> bool:
        c = self.db.cursor()
        path = self.prefix + path
        c.execute(self.contains_sql, (self.prefix + path,))
        return len(c.fetchall()) != 0

    def print_tree(self):
        c = self.db.cursor()
        c.execute(self.list2_sql)
        data = list(c.fetchall())
        data.sort()
        print("------------------ DB ---------------------")
        for key, data_ln, type in data:
            print(key, data_ln, type)
        print("------------------ END --------------------")

    def sub_storage(self, path: str) -> 'DBStorage':
        return self.__class__(prefix=self.prefix + path, db=self.db)

    def sync(self):
        self.db.commit()

    def get_fd(self, path: str, mode: str = "rb+") -> IO[bytes]:
        raise NotImplementedError("SQLITE3 doesn't provide fd-like interface")

    def list(self, path: str) -> Iterator[Tuple[bool, str]]:
        raise NotImplementedError("SQLITE3 doesn't provide list method")


DB_REL_PATH = "__db__.db"


class FSStorage(ISimpleStorage):
    """Store all data in files on FS"""

    def __init__(self, root_path: str, existing: bool) -> None:
        self.root_path = root_path
        self.existing = existing
        self.ignored = {self.j(DB_REL_PATH), '.', '..'}

    def j(self, path: str) -> str:
        return os.path.join(self.root_path, path)

    def put(self, value: bytes, path: str) -> None:
        jpath = self.j(path)
        os.makedirs(os.path.dirname(jpath), exist_ok=True)
        with open(jpath, "wb") as fd:
            fd.write(value)

    def get(self, path: str) -> bytes:
        try:
            with open(self.j(path), "rb") as fd:
                return fd.read()
        except FileNotFoundError as exc:
            raise KeyError(path) from exc

    def rm(self, path: str) -> None:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.unlink(path)

    def __contains__(self, path: str) -> bool:
        return os.path.exists(self.j(path))

    def get_fd(self, path: str, mode: str = "rb+") -> IO[bytes]:
        jpath = self.j(path)

        if "cb" == mode:
            create_on_fail = True
            mode = "rb+"
            os.makedirs(os.path.dirname(jpath), exist_ok=True)
        elif "ct" == mode:
            create_on_fail = True
            mode = "rt+"
            os.makedirs(os.path.dirname(jpath), exist_ok=True)
        else:
            create_on_fail = False

        try:
            fd = open(jpath, mode)
        except IOError:
            if not create_on_fail:
                raise

            if 't' in mode:
                fd = open(jpath, "wt")
            else:
                fd = open(jpath, "wb")

        return cast(IO[bytes], fd)

    def sub_storage(self, path: str) -> 'FSStorage':
        return self.__class__(self.j(path), self.existing)

    def sync(self):
        pass

    def list(self, path: str) -> Iterator[Tuple[bool, str]]:
        path = self.j(path)

        if not os.path.exists(path):
            return

        if not os.path.isdir(path):
            raise OSError("{!r} is not a directory".format(path))

        for fobj in os.scandir(path):
            if fobj.path not in self.ignored:
                if fobj.is_dir():
                    yield False, fobj.name
                else:
                    yield True, fobj.name


class YAMLSerializer(ISerializer):
    """Serialize data to yaml"""
    def pack(self, value: IStorable) -> bytes:
        try:
            return yaml.dump(value, Dumper=Dumper, encoding="utf8")
        except Exception as exc:
            raise ValueError("Can't pickle object {!r} to yaml".format(type(value))) from exc

    def unpack(self, data: bytes) -> Any:
        return yaml.load(data, Loader=Loader)


class SAFEYAMLSerializer(ISerializer):
    """Serialize data to yaml"""
    def pack(self, value: IStorable) -> bytes:
        try:
            return yaml.safe_dump(value, encoding="utf8")
        except Exception as exc:
            raise ValueError("Can't pickle object {!r} to yaml".format(type(value))) from exc

    def unpack(self, data: bytes) -> Any:
        return yaml.safe_load(data)


ObjClass = TypeVar('ObjClass', bound=IStorable)


class _Raise:
    pass


csv_file_encoding = 'ascii'


class Storage:
    """interface for storage"""

    def __init__(self, fs_storage: ISimpleStorage, db_storage: ISimpleStorage, serializer: ISerializer) -> None:
        self.fs = fs_storage
        self.db = db_storage
        self.serializer = serializer

    def sub_storage(self, *path: str) -> 'Storage':
        fpath = "/".join(path)
        return self.__class__(self.fs.sub_storage(fpath), self.db.sub_storage(fpath), self.serializer)

    def put(self, value: Any, *path: str) -> None:
        dct_value = cast(IStorable, value).raw() if isinstance(value, IStorable) else value
        serialized = self.serializer.pack(dct_value)  # type: ignore
        fpath = "/".join(path)
        self.db.put(serialized, fpath)
        self.fs.put(serialized, fpath)

    def put_list(self, value: Iterable[IStorable], *path: str) -> None:
        serialized = self.serializer.pack([obj.raw() for obj in value])  # type: ignore
        fpath = "/".join(path)
        self.db.put(serialized, fpath)
        self.fs.put(serialized, fpath)

    def get(self, path: str, default: Any = _Raise) -> Any:
        try:
            vl = self.db.get(path)
        except:
            if default is _Raise:
                raise
            return default

        return self.serializer.unpack(vl)

    def rm(self, *path: str) -> None:
        fpath = "/".join(path)
        self.fs.rm(fpath)
        self.db.rm(fpath)

    def __contains__(self, path: str) -> bool:
        return path in self.fs or path in self.db

    def put_raw(self, val: bytes, *path: str) -> str:
        fpath = "/".join(path)
        self.fs.put(val, fpath)
        # TODO: dirty hack
        return self.resolve_raw(fpath)

    def resolve_raw(self, fpath) -> str:
        return cast(FSStorage, self.fs).j(fpath)

    def get_raw(self, *path: str) -> bytes:
        return self.fs.get("/".join(path))

    def append_raw(self, value: bytes, *path: str) -> None:
        with self.fs.get_fd("/".join(path), "rb+") as fd:
            fd.seek(0, os.SEEK_END)
            fd.write(value)

    def get_fd(self, path: str, mode: str = "r") -> IO:
        return self.fs.get_fd(path, mode)

    def put_array(self, header: List[str], value: numpy.array, *path: str) -> None:
        for val in header:
            assert isinstance(val, str) and ',' not in val, \
                "Can't convert {!r} to array header, as it's values contains comma".format(header)

        fpath = "/".join(path)
        with self.get_fd(fpath, "wb") as fd:
            self.do_append(fd, header, value, fpath)

    def get_array(self, *path: str) -> Tuple[List[str], numpy.array]:
        path_s = "/".join(path)
        with self.get_fd(path_s, "rb") as fd:
            header = fd.readline().decode(csv_file_encoding).rstrip().split(",")
            type_code, second_axis = header[-2:]
            res = numpy.genfromtxt(fd, dtype=type_code, delimiter=',')

        if '0' == second_axis:
            res.shape = (len(res),)

        return header[:-2], res

    def append(self, header: List[str], value: numpy.array, *path: str) -> None:
        for val in header:
            assert isinstance(val, str) and ',' not in val, \
                "Can't convert {!r} to array header, as it's values contains comma".format(header)

        fpath = "/".join(path)
        with self.get_fd(fpath, "cb") as fd:
            self.do_append(fd, header, value, fpath, maybe_append=True)

    def do_append(self, fd, header: List[str], value: numpy.array, path: str, fmt="%lu",
                  maybe_append: bool = False) -> None:

        if len(value.shape) == 1:
            second_axis = 0
        else:
            second_axis = value.shape[1]
        header += [value.dtype.name, str(second_axis)]

        write_header = False

        if maybe_append:
            fd.seek(0, os.SEEK_END)
            if fd.tell() != 0:
                fd.seek(0, os.SEEK_SET)
                # check header match
                curr_header = fd.readline().decode(csv_file_encoding).rstrip().split(",")
                assert header == curr_header, \
                    "Path {!r}. Expected header ({!r}) and current header ({!r}) don't match"\
                        .format(path, header, curr_header)
                fd.seek(0, os.SEEK_END)
            else:
                write_header = True
        else:
            write_header = True

        if write_header:
            fd.write((",".join(header) + "\n").encode(csv_file_encoding))

        if len(value.shape) == 1:
            # make array vertical to simplify reading
            vw = value.view().reshape((value.shape[0], 1))
        else:
            vw = value
        numpy.savetxt(fd, vw, delimiter=',', newline="\n", fmt=fmt)

    def load_list(self, obj_class: Type[ObjClass], *path: str) -> List[ObjClass]:
        path_s = "/".join(path)
        raw_val = cast(List[Dict[str, Any]], self.get(path_s))
        assert isinstance(raw_val, list)
        return [cast(ObjClass, obj_class.fromraw(val)) for val in raw_val]

    def load(self, obj_class: Type[ObjClass], *path: str) -> ObjClass:
        path_s = "/".join(path)
        return cast(ObjClass, obj_class.fromraw(self.get(path_s)))

    def sync(self) -> None:
        self.db.sync()
        self.fs.sync()

    def __enter__(self) -> 'Storage':
        return self

    def __exit__(self, x: Any, y: Any, z: Any) -> None:
        self.sync()

    def list(self, *path: str) -> Iterator[Tuple[bool, str]]:
        return self.fs.list("/".join(path))

    def _iter_paths(self,
                    root: str,
                    path_parts: List[str],
                    groups: Dict[str, str]) -> Iterator[Tuple[bool, str, Dict[str, str]]]:

        curr = path_parts[0]
        rest = path_parts[1:]

        for is_file, name in self.list(root):
            if rest and is_file:
                continue

            rr = re.match(pattern=curr + "$", string=name)
            if rr:
                if root:
                    path = root + "/" + name
                else:
                    path = name

                new_groups = rr.groupdict().copy()
                new_groups.update(groups)

                if rest:
                    yield from self._iter_paths(path, rest, new_groups)
                else:
                    yield is_file, path, new_groups


def make_storage(url: str, existing: bool = False) -> Storage:
    return Storage(FSStorage(url, existing),
                   DBStorage(os.path.join(url, DB_REL_PATH)),
                   SAFEYAMLSerializer())

