const _: () = assert!(haem_worker::MEMORY_LIMIT_EXIT == haem_frames::embedding::MEMORY_LIMIT_EXIT);

pub fn prepare() -> std::io::Result<haem_worker::setup::Resources> {
    haem_worker::setup::prepare()
}

pub fn arm(bytes: usize) {
    haem_worker::allocator::arm(bytes);
}
