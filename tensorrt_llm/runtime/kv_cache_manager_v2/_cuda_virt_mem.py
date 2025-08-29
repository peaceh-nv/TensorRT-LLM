import cuda.bindings.driver as drv

from ._utils import _unwrap


# Physical memory
class PhysMem:
    SIZE: int = 32 << 20
    __slots__ = ('device_id', 'handle')

    device_id: int
    handle: drv.CUmemGenericAllocationHandle

    def __init__(self):
        self.device_id = _unwrap(
            drv.cuCtxGetDevice())  # type: ignore[attr-defined]
        prop = drv.CUmemAllocationProp()
        prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device_id
        self.handle = _unwrap(drv.cuMemCreate(PhysMem.SIZE, prop, 0))

    def destroy(self):
        if self.handle == drv.CUmemGenericAllocationHandle(0):
            return
        _unwrap(drv.cuMemFree(self.handle))
        self.handle = drv.CUmemGenericAllocationHandle(0)

    def __del__(self):
        self.destroy()


# Virtual memory
class VirtMem:
    __slots__ = ('device_id', 'address', 'vm_size', 'pm_stack', '_access_desc')
    device_id: int
    address: drv.CUdeviceptr
    vm_size: int
    pm_stack: list[PhysMem]

    _access_desc: drv.CUmemAccessDesc

    def __init__(self, vm_size: int):
        assert vm_size % PhysMem.SIZE == 0
        self.device_id = _unwrap(
            drv.cuCtxGetDevice())  # type: ignore[attr-defined]
        self.address = _unwrap(drv.cuMemAddressReserve(vm_size, 0, 0, 0))
        self.vm_size = vm_size
        self.pm_stack = []
        self._access_desc = drv.CUmemAccessDesc()
        self._access_desc.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        self._access_desc.location.id = self.device_id
        self._access_desc.flags = drv.CUmemAccessFlags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    def destroy(self):
        if self.vm_size == 0:
            return
        while self.pm_stack:
            self.pm_stack.pop()
        _unwrap(drv.cuMemAddressFree(self.address, self.vm_size))
        self.address = drv.CUdeviceptr(0)
        self.vm_size = 0

    def __del__(self):
        self.destroy()

    def push(self, phy_mem: PhysMem):
        assert PhysMem.SIZE * (
            len(self.pm_stack) +
            1) <= self.vm_size and self.device_id == phy_mem.device_id
        vm_ptr = drv.CUdeviceptr(
            int(self.address.value()) + PhysMem.SIZE * len(self.pm_stack))
        _unwrap(drv.cuMemMap(vm_ptr, PhysMem.SIZE, 0, phy_mem.handle, 0))
        _unwrap(drv.cuMemSetAccess(vm_ptr, PhysMem.SIZE, self._access_desc))
        self.pm_stack.append(phy_mem)

    def pop(self) -> PhysMem:
        assert self.pm_stack
        vm_ptr = drv.CUdeviceptr(
            int(self.address.value()) + PhysMem.SIZE * (len(self.pm_stack) - 1))
        _unwrap(drv.cuMemUnmap(vm_ptr, PhysMem.SIZE))
        return self.pm_stack.pop()

    def mapped_size(self) -> int:
        return PhysMem.SIZE * len(self.pm_stack)

    def virtual_size(self) -> int:
        return self.vm_size
