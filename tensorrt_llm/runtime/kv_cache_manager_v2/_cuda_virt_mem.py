import cuda.bindings.driver as drv

from ._utils import ItemHolderWithSharedPool, PooledFactoryBase, _unwrap, div_up


# Physical memory
class NativePhysMemAllocator:
    __slots__ = ('_device_id', '_size', '_prop')

    _device_id: int
    _size: int
    _prop: drv.CUmemAllocationProp

    def __init__(self, size: int):
        self._device_id = _unwrap(
            drv.cuCtxGetDevice())  # type: ignore[attr-defined]
        self._size = size
        prop = drv.CUmemAllocationProp()
        prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self._device_id
        self._prop = prop

    def allocate(self) -> drv.CUmemGenericAllocationHandle:
        return _unwrap(drv.cuMemCreate(self._size, self._prop, 0))

    def release(self, handle: drv.CUmemGenericAllocationHandle):
        if handle == drv.CUmemGenericAllocationHandle(0):
            return
        _unwrap(drv.cuMemFree(handle))

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def size(self) -> int:
        return self._size


class PhysMem(ItemHolderWithSharedPool[drv.CUmemGenericAllocationHandle]):
    SIZE: int = 32 << 20
    __slots__ = ()


class PooledPhysMemAllocator(PooledFactoryBase[drv.CUmemGenericAllocationHandle,
                                               PhysMem]):
    _Holder = PhysMem
    __slots__ = ("device_id", )
    device_id: int

    def __init__(self):
        raw_alloc = NativePhysMemAllocator(PhysMem.SIZE)
        self.device_id = raw_alloc.device_id
        super().__init__(lambda: raw_alloc.allocate(),
                         lambda handle: raw_alloc.release(handle))


# Virtual memory
class VirtMem:
    __slots__ = ('_vm_size', '_allocator', '_address', '_pm_stack',
                 '_access_desc')
    _vm_size: int
    _allocator: PooledPhysMemAllocator
    _address: drv.CUdeviceptr
    _pm_stack: list[PhysMem]
    _access_desc: drv.CUmemAccessDesc

    def __init__(self,
                 vm_size: int,
                 phys_mem_allocator: PooledPhysMemAllocator,
                 init_num_phys_mem: int = 0):
        assert vm_size % PhysMem.SIZE == 0
        self._allocator = phys_mem_allocator
        device_id = phys_mem_allocator.device_id
        self._address = _unwrap(drv.cuMemAddressReserve(vm_size, 0, 0, 0))
        self._vm_size = vm_size
        self._pm_stack = []
        self._access_desc = drv.CUmemAccessDesc()
        self._access_desc.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        self._access_desc.location.id = device_id
        self._access_desc.flags = drv.CUmemAccessFlags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.extend(init_num_phys_mem)

    def destroy(self):
        if self._vm_size == 0:
            return
        while self._pm_stack:
            self._pm_stack.pop().close()
        _unwrap(drv.cuMemAddressFree(self._address, self._vm_size))
        self._address = drv.CUdeviceptr(0)
        self._vm_size = 0

    def __del__(self):
        self.destroy()

    def extend(self, num_phys_mem: int):
        old_num_phys_mem = self.num_phys_mem
        try:
            for _ in range(num_phys_mem):
                self._push(self._allocator.create())
        except Exception:  # to make realloc behave like normal realloc, we need to rollback if out of memory
            while self.num_phys_mem > old_num_phys_mem:
                self._pop().close()
            raise

    def shrink(self, num_phys_mem: int):
        for _ in range(num_phys_mem):
            self._pop().close()

    # Different from normal realloc, this function never changes the pointer.
    def realloc(self, num_bytes: int):
        required_num_phys_mem = div_up(num_bytes, PhysMem.SIZE)
        if required_num_phys_mem > self.num_phys_mem:
            self.extend(required_num_phys_mem - self.num_phys_mem)
        elif required_num_phys_mem < self.num_phys_mem:
            self.shrink(self.num_phys_mem - required_num_phys_mem)

    def _push(self, phy_mem: PhysMem):
        assert PhysMem.SIZE * (len(self._pm_stack) + 1) <= self._vm_size
        vm_ptr = drv.CUdeviceptr(
            int(self._address.value()) + PhysMem.SIZE * len(self._pm_stack))
        _unwrap(drv.cuMemMap(vm_ptr, PhysMem.SIZE, 0, phy_mem.handle, 0))
        _unwrap(drv.cuMemSetAccess(vm_ptr, PhysMem.SIZE, self._access_desc))
        self._pm_stack.append(phy_mem)

    def _pop(self) -> PhysMem:
        assert self._pm_stack
        vm_ptr = drv.CUdeviceptr(
            int(self._address.value()) + PhysMem.SIZE *
            (len(self._pm_stack) - 1))
        _unwrap(drv.cuMemUnmap(vm_ptr, PhysMem.SIZE))
        return self._pm_stack.pop()

    @property
    def mapped_bytes(self) -> int:
        return PhysMem.SIZE * self.num_phys_mem

    @property
    def virtual_bytes(self) -> int:
        return self._vm_size

    @property
    def num_phys_mem(self) -> int:
        return len(self._pm_stack)

    @property
    def address(self) -> drv.CUdeviceptr:
        return self._address
