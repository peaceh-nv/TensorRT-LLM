import cuda.bindings.driver as drv


class OutOfMemory(MemoryError):
    pass


class OutOfGpuMemory(OutOfMemory):
    __slots__ = ('previous', 'goal', 'actual')
    previous: int | None
    goal: int | None
    actual: int | None

    def __init__(self,
                 previous: int | None = None,
                 goal: int | None = None,
                 actual: int | None = None):
        self.previous = previous
        self.goal = goal
        self.actual = actual if actual is not None else goal
        super().__init__(
            f"Out of GPU memory: previous={previous}, goal={goal}, actual={actual}"
        )


class OutOfHostMemory(OutOfMemory):
    pass


class OutOfDiskSpace(OutOfMemory):
    pass


class LogicError(Exception):
    '''
    This exception indicates a bug in the code.
    '''

    def __init__(self, message: str):
        super().__init__(message)


class CudaDriverError(RuntimeError):
    error_code: drv.CUresult

    def __init__(self, error_code: drv.CUresult):
        self.error_code = error_code
        super().__init__(f"CUDA driver error: {error_code}")


class ResourceBusy(RuntimeError):
    pass
