# -*- coding: utf-8 -*-
"""
Classes associated with the creation, keeping and editing of log entries for a
model run.
"""
import os
import errno
import time

class Log:
    """
    Class to handle creation, modification and storage of log entries
    """

    def __init__(self, fname, verbose=True):
        """
        Parameters
        ----------
        filename : str
            Full path to log file
        verbose : bool
            Whether to print log entries verbosely.
        """
        self._entries = {}
        self._filename = fname
        self._verbose = verbose

    def __str__(self):
        es = []
        for entry_num in range(1, len(self.entries) + 1):
            es.append(self.entries[entry_num].__str__)

        return '\n'.join(es)

    @property
    def filename(self):
        return self._filename

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, new_verbosity):
        self._verbose = new_verbosity

    @property
    def entries(self):
        return self._entries

    @entries.setter
    def entries(self, new_entries):
        self._entries = new_entries

    def add_entry(self, mtype, entry):
        """
        Add entry to log

        Parameters
        ----------
        mtype : str
            Log entry type (e.g. info, error etc.)
        entry: str
            Message to enter into log

        Returns
        -------
        None.

        """
        if not os.path.exists(os.path.dirname(self.filename)):
            # Raise FileNotFoundError (subclass of builtin OSError) correctly
            raise FileNotFoundError(errno.ENOTDIR,
                                    os.strerror(errno.ENOTDIR),
                                    os.path.dirname(self.filename))

        if not os.path.exists(self.filename):
            # os.mknod(self.filename) --> fails with PermissionError
            open(self.filename, 'w').close()

        new_entry = Entry(mtype, entry)
        new_entries = self.entries
        new_entries[len(self._entries) + 1] = new_entry
        self.entries = new_entries

        if self.verbose:
            print(new_entry)

        with open(self.filename, 'at') as f:
            f.write(('\n' if len(self.entries) != 1 else "") +
                    new_entry.__str__())


class Entry:
    """
    Entry class for use with Log class
    """
    _valid_mtypes = ("INFO", "ERROR", "WARNING")
    _mtype_max_len = max([len(_) for _ in _valid_mtypes])

    @classmethod
    def valid_mtypes(cls):
        return cls._valid_mtypes

    @classmethod
    def mtype_max_len(cls):
        return cls._mtype_max_len

    def __init__(self, mtype, entry):
        """
        Parameters
        ----------
        mtype : str
            Message type. One of 'INFO', 'ERROR' or 'WARNING' (any case)
        entry: str
            Entry message

        Returns
        -------
        None.

        """

        if not isinstance(mtype, str):
            raise TypeError("mtype must be a str")

        if not isinstance(entry, str):
            raise TypeError("entry must be a str")

        if mtype.upper() not in Entry.valid_mtypes():
            raise TypeError("mtype must be one of '" +
                            "', '".join(self._valid_mtypes[:-1]) + "' or '" +
                            self._valid_mtypes[-1] + "'")

        self._mtime = time.localtime()
        self._mtype = mtype
        self._message = entry

    def __repr__(self):
        s = "Entry(mtype='" + self.mtype + "', entry='" + self.message + "')"
        return s
            
    def __str__(self):
        return ':: '.join([self.time_str(), self.mtype, self.message])
    
    @property
    def message(self):
        return self._message
    
    @property
    def mtype(self):
        return self._mtype

    @property
    def mtime(self):
        return self._mtime

    def time_str(self, fmt='%d%B%Y-%H:%M:%S'):
        return time.strftime(fmt, self.mtime).upper()
