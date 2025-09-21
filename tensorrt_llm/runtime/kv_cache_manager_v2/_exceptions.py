import cuda.bindings.driver as drv


class OutOfMemoryError(MemoryError):
    pass


class HostOOMError(OutOfMemoryError):
    pass


class DiskOOMError(OutOfMemoryError):
    pass


class LogicError(Exception):
    '''
    This exception indicates a bug in the code.
    '''

    def __init__(self, message: str):
        super().__init__(message)


class CuError(RuntimeError):
    error_code: drv.CUresult

    def __init__(self, error_code: drv.CUresult):
        self.error_code = error_code
        err, err_str = drv.cuGetErrorString(error_code)
        if err != drv.CUresult.CUDA_SUCCESS:
            err_str = "<Failed to get error string with cuGetErrorString>"
        super().__init__(f"CUDA driver error: {error_code} ({err_str})")


class CuOOMError(CuError, OutOfMemoryError):

    def __init__(self):
        super().__init__(drv.CUresult.CUDA_ERROR_OUT_OF_MEMORY)


class ResourceBusyError(RuntimeError):
    pass


class OutOfPagesError(RuntimeError):
    pass
