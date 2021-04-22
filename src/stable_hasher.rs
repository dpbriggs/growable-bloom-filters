use std::hash::Hasher;

/// Wrapper over a hasher that provides stable output across platforms
/// Based on https://github.com/rust-lang/rust/blob/c0955a34bcb17f0b31d7b86522a520ebe7fa93ac/src/librustc_data_structures/stable_hasher.rs#L78-L166
///
/// To that end we always convert integers to little-endian format before
/// hashing and the architecture dependent `isize` and `usize` types are
/// extended to 64 bits if needed.
pub struct StableHasher {
    /// Using xxh3-64 with default seed/secret as the portable hasher.
    state: xxhash_rust::xxh3::Xxh3,
}

impl StableHasher {
    #[inline]
    pub fn new() -> Self {
        Self {
            state: xxhash_rust::xxh3::Xxh3::new(),
        }
    }
}

impl Hasher for StableHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.state.finish()
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.state.write(bytes);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.state.write_u8(i);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.state.write_u16(i.to_le());
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.state.write_u32(i.to_le());
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.state.write_u64(i.to_le());
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.state.write_u128(i.to_le());
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        // Always treat usize as u64 so we get the same results on 32 and 64 bit
        // platforms. This is important for symbol hashes when cross compiling,
        // for example.
        self.state.write_u64((i as u64).to_le());
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.state.write_i8(i);
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.state.write_i16(i.to_le());
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.state.write_i32(i.to_le());
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.state.write_i64(i.to_le());
    }

    #[inline]
    fn write_i128(&mut self, i: i128) {
        self.state.write_i128(i.to_le());
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        // Always treat isize as i64 so we get the same results on 32 and 64 bit
        // platforms. This is important for symbol hashes when cross compiling,
        // for example.
        self.state.write_i64((i as i64).to_le());
    }
}
