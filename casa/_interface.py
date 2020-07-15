"""
Contains any classes responsible for the scripting of casa tasks and pipelines
which are subsequently executed on the command line using 'casa -c [script]'
"""

import os


class Script(object):
    """
    Class to handle a collection of coherent list of casa tasks/tool and execute
    that collection with casa, in the order in which it is given.
    """
    def __init__(self):
        self._tasklist = []

        # Must always add e-MERLIN's primary beam response to CASA's vpmanager
        from VaJePy.casa.tasks import AddGaussPBresponse

        fwhm_str = '{:.3f}deg'.format(1.71768e10 / (1e9 * 25.))
        maxrad_str = '{:.3f}deg'.format(3.43537e10 / (1e9 * 25.))

        self.add_task(AddGaussPBresponse(telescope='MERLIN2',
                                         halfwidth=fwhm_str,
                                         maxrad=maxrad_str,
                                         reffreq='1GHz'))

    @property
    def tasklist(self):
        return self._tasklist

    @tasklist.setter
    def tasklist(self, new_tasklist):
        self._tasklist = new_tasklist

    def add_task(self, new_task):
        from collections.abc import Iterable
        if not isinstance(new_task, Iterable):
            self.tasklist.append(new_task)
        else:
            for task in new_task:
                self.tasklist.append(task)

    def execute(self, dcy=os.getcwd(), dryrun=False):
        import shutil
        import subprocess
        from datetime import datetime as dt

        pwd = dcy

        if dcy != os.getcwd():
            os.chdir(dcy)
        #
        # tmp_dcy = pwd + os.sep + 'tmp'
        # os.mkdir(tmp_dcy)
        # os.chdir(tmp_dcy)

        prefix = dt.now().strftime("%d%m%Y_%H%M%S")
        logfile = dcy + os.sep + prefix + '.log'
        casafile = dcy + os.sep + prefix + '.py'

        with open(casafile, 'a+') as lf:
            # Necessary imports within CASA environment
            lf.write('import shutil\n')
            for task in self.tasklist:
                lf.write(str(task) + '\n')

        cmd = "casa --nogui --nologger --agg --logfile {} -c {}"

        if dryrun:
            print(cmd.format(logfile, casafile))
            print("Contents of {}:".format(casafile))
            with open(casafile, 'rt') as lf: print(lf.read())

            # shutil.copy(logfile, dcy + os.sep + logfile)
            # shutil.copy(casafile, dcy + os.sep + os.path.basename(casafile))

            # print("Cleaning up by removing {}\n".format(tmp_dcy))
            # shutil.rmtree(tmp_dcy)

        else:
            op = subprocess.run(cmd.format(logfile, casafile), shell=True)
            # print("Cleaning up by removing {}\n".format(tmp_dcy))
            # shutil.rmtree(tmp_dcy)