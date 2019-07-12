
from multiprocessing import Process, Value


class AccessComponent(object):

    def __init__(self, target_func, cmd_args, **kwargs):
        self.target_func = target_func
        self.kill_proc = Value('i', 0)
        kwargs['kill_proc'] = self.kill_proc
        kwargs['cmd_args'] = cmd_args
        self.cmd_args = cmd_args
        self.kwargs = kwargs
        self.process = Process(target=target_func, kwargs=kwargs)

    def __del__(self):
        self.stop()

    def start(self):
        self.process.daemon = True
        self.process.start()

    def stop(self):
        self.kill_proc.value = 1
