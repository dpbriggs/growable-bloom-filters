#![deny(unsafe_code)]
#![feature(test)]
// Impl of Scalable Bloom Filters
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.62.7953&rep=rep1&type=pdf
extern crate test;
use bitvec::prelude::BitVec;
use seahash::SeaHasher;
use serde_derive::{Deserialize, Serialize};
use std::{
    hash::{Hash, Hasher},
    iter::Iterator,
};

/// Base Bloom Filter
#[derive(Deserialize, Serialize, PartialEq, Clone, Debug)]
struct Bloom {
    /// The actual bit field. Set to 0 with `Bloom::new`.
    field: BitVec,
    /// The number of slices in the bloom filter.
    /// A single insertion will result in a single bit being set in each slice.
    num_slices: usize,
    /// The _bit_ length of each slice.
    slice_len: usize,
    /// The seed used in the hash function.
    seed: u64,
}

impl Bloom {
    /// Create a new Bloom filter
    ///
    /// # Arguments
    ///
    /// * `num_slices` - The number of slices used in the bloom filter.
    /// * `slice_len` - The actual _bit_ length of each slide.
    /// * `seed` - A pseudo random seed; used in hashing each slice.
    fn new(num_slices: usize, slice_len: usize, seed: u64) -> Bloom {
        debug_assert!(slice_len >= 1);
        debug_assert!(num_slices > 0);
        let bitvec_size = num_slices * slice_len;
        let mut field = BitVec::with_capacity(bitvec_size);
        for _ in 0..bitvec_size {
            field.push(false);
        }
        field.shrink_to_fit();
        Bloom {
            field,
            num_slices,
            slice_len,
            seed,
        }
    }

    /// Create an index iterator for a given item.
    ///
    /// This creates a stream of indices corresponding to a single index
    /// per slice.
    ///
    /// # Arguments
    ///
    /// * `item` - The item to hash.
    fn index_iterator<'a, T: Hash>(&self, item: &'a T) -> impl Iterator<Item = usize> + 'a {
        let slice_len = self.slice_len;
        let (k1, k2, k3, k4) = generate_seed(self.seed);
        let mut hasher = SeaHasher::with_seeds(k1, k2, k3, k4);
        (0..self.num_slices).map(move |curr_slice| {
            item.hash(&mut hasher);
            let hash = hasher.finish();
            hasher.write_u64(hash);
            (hash as usize % slice_len) + curr_slice * slice_len
        })
    }

    /// Insert an `item` into the Bloom.
    ///
    /// # Arguments
    ///
    /// * `item` - The item to insert
    ///
    /// # Example
    ///
    ///
    /// use growable_bloom_filter::Bloom;
    /// let bloom = Bloom::new(2, 128);
    ///
    /// let item = 0;
    /// bloom.insert(&item);
    ///
    /// let item = "Hello World".to_owned();
    /// bloom.insert(&item);
    ///
    fn insert<T: Hash>(&mut self, item: &T) {
        for index in self.index_iterator(item) {
            self.field.set(index, true)
        }
    }

    /// Test if `item` is in the Bloom.
    ///
    /// # Arguments
    ///
    /// * `item` - The item to test
    ///
    /// # Example
    ///
    ///
    /// let bloom = Bloom:new(2, 128);
    /// let item = 0;
    ///
    /// bloom.insert(&item);
    /// assert!(bloom.contains(&item));
    ///
    fn contains<T: Hash>(&self, item: &T) -> bool {
        self.index_iterator(item)
            .all(|index| self.field.get(index).unwrap())
    }

    /// Test the fill ratio of the Bloom
    ///
    /// # Arguments
    ///
    /// * `lower_bound` - The minimum fill ratio
    ///
    /// # Example
    ///
    ///
    /// let bloom = Bloom::new(0.05, 10);
    /// let item = 0;
    ///
    /// bloom.insert(&item);
    /// assert!(!bloom.fill_ratio_gte(0.01));
    ///
    fn fill_ratio_gte(&self, lower_bound: f64) -> bool {
        let len = (self.slice_len * self.num_slices) as f64;
        (self.field.count_ones() as f64 / len) >= lower_bound
    }
}

/// Convenience function to hash a u64
///
/// # Arguments
///
/// * `i` - The u64 to hash
fn hash_u64(i: u64) -> u64 {
    seahash::hash(&i.to_be_bytes())
}

/// Stretch a u64 into four u64s
///
/// # Arguments
///
/// * `base_seed` - The seed to stretch
#[inline]
fn generate_seed(base_seed: u64) -> (u64, u64, u64, u64) {
    let h1 = hash_u64(base_seed);
    let h2 = hash_u64(h1);
    let h3 = hash_u64(h2);
    let mut hasher = SeaHasher::with_seeds(base_seed, h1, h2, h3);
    hasher.write_u64(0);
    let a = hasher.finish();
    hasher.write_u64(a);
    let b = hasher.finish();
    hasher.write_u64(b);
    let c = hasher.finish();
    hasher.write_u64(c);
    let d = hasher.finish();
    (a, b, c, d)
}

/// A Growable Bloom Filter
///
/// # Overview
///
/// Implementation of [Scalable Bloom Filters](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.62.7953&rep=rep1&type=pdf)
/// which also provides serde serialization and deserialize.
///
/// A bloom filter lets you `insert` items, and then test association with `contains`.
/// It's space and time efficient, at the cost of false positives.
/// In particular, if `contains` returns `true`, it may be in filter.
/// But if `contains` returns false, it's definitely not in the bloom filter.
///
/// You can control the failure rate by setting `desired_error_prob` and `est_insertions` appropriately.
///
/// # Applications
/// Bloom filters are typically used as a pre-cache to avoid expensive operations.
/// For example, if you need to ask ten thousand servers if they have data XYZ,
/// you could use GrowableBloom to figure out which ones do NOT have XYZ.
///
/// # Example
///
/// ```rust
/// use serde_json;
/// use growable_bloom_filter::GrowableBloom;
///
/// let mut gbloom = GrowableBloom::new(0.05, 1000);
/// gbloom.insert(&0);
/// assert!(gbloom.contains(&0));
///
/// let s = serde_json::to_string(&gbloom).unwrap();
/// let des_gbloom: GrowableBloom = serde_json::from_str(&s).unwrap();
/// assert!(des_gbloom.contains(&0));
/// ```
#[derive(Deserialize, Serialize, PartialEq, Clone, Debug)]
pub struct GrowableBloom {
    /// The constituent bloom filters
    blooms: Vec<Bloom>,
    /// The current number of slices in the bloom filter
    curr_num_slice: usize,
    /// The current size of each slice in the bloom filter
    slice_size: usize,
    /// The current seed
    curr_seed: u64,
}

impl GrowableBloom {
    const GROWTH_FACTOR: usize = 2;

    /// Create a new GrowableBloom filter.
    ///
    /// # Arguments
    ///
    /// * `desired_error_prob` - The desired error probability (eg. 0.05, 0.01)
    /// * `est_insertions` - The estimated number of insertions (eg. 100, 1000)
    ///
    /// NOTE: You really don't need to be accurate with est_insertions.
    ///       Power of 10 granularity should be fine.
    /// # Example
    /// ```rust
    /// // 5% failure rate, estimated 100 elements to insert
    /// use growable_bloom_filter::GrowableBloom;
    /// let mut gbloom = GrowableBloom::new(0.05, 100);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if desired_error_prob is less then 0 or greater than 1
    pub fn new(desired_error_prob: f64, est_insertions: usize) -> GrowableBloom {
        assert!(0.0 < desired_error_prob && desired_error_prob <= 1.0);
        // directly from paper: k ~ log_2(1/desired_error_prob)
        let opt_num_slices = ((1.0 / desired_error_prob).log2()).ceil();
        // re-arrange est_insertions ~ M(ln(2)^2 / ln(desired_error_prob))
        let opt_total_bits = (desired_error_prob.ln().abs() * est_insertions as f64
            / 2f64.ln().powi(2))
        .ceil() as usize;
        let opt_num_slices = opt_num_slices as usize;
        let slice_size = opt_total_bits / opt_num_slices;
        let curr_seed = 0;
        let first_bloom = Bloom::new(opt_num_slices, slice_size, curr_seed);
        debug_assert!(opt_num_slices > 0);
        GrowableBloom {
            blooms: vec![first_bloom],
            curr_num_slice: opt_num_slices,
            slice_size,
            curr_seed,
        }
    }

    /// Test if `item` in the Bloom filter.
    ///
    /// If `true` is returned, it's _may_ be in the filter.
    /// If `false` is returned, it's NOT in the filter.
    ///
    /// # Arguments
    ///
    /// * `item` - The item to test
    ///
    /// # Example
    ///
    /// ```rust
    /// use growable_bloom_filter::GrowableBloom;
    /// let mut bloom = GrowableBloom::new(0.05, 10);
    /// let item = 0;
    ///
    /// bloom.insert(&item);
    /// assert!(bloom.contains(&item));
    /// ```
    pub fn contains<T: Hash>(&self, item: T) -> bool {
        debug_assert!(!self.blooms.is_empty());
        self.blooms.iter().any(|bloom| bloom.contains(&item))
    }

    /// Insert `item` into the filter.
    ///
    /// This may resize the GrowableBloom.
    ///
    /// # Arguments
    ///
    /// * `item` - The item to insert
    ///
    /// # Example
    ///
    /// ```rust
    /// use growable_bloom_filter::GrowableBloom;
    /// let mut bloom = GrowableBloom::new(0.05, 10);
    /// let item = 0;
    ///
    /// bloom.insert(&item);
    /// bloom.insert(&-1);
    /// bloom.insert(&vec![1, 2, 3]);
    /// bloom.insert("hello");
    /// ```
    pub fn insert<T: Hash>(&mut self, item: T) {
        // Step 1: Ask if we already have it
        debug_assert!(!self.blooms.is_empty());
        if self.contains(&item) {
            return;
        }
        // Step 2: Insert it into the last
        let curr_bloom = self.blooms.last_mut().unwrap();
        curr_bloom.insert(&item);
        // Step 3: Grow if necessary
        if curr_bloom.fill_ratio_gte(0.8) {
            self.grow();
        }
    }

    /// Grow the GrowableBloom
    fn grow(&mut self) {
        self.curr_num_slice += 1;
        self.slice_size *= GrowableBloom::GROWTH_FACTOR;
        self.curr_seed ^= hash_u64(self.curr_seed);
        let new_bloom = Bloom::new(self.curr_num_slice, self.slice_size, self.curr_seed);
        self.blooms.push(new_bloom)
    }
}

#[cfg(test)]
mod growable_bloom_tests {
    mod test_bloom {
        use crate::Bloom;
        #[test]
        fn can_insert_bloom() {
            let mut b = Bloom::new(2, 1024, 10);
            let item = 20;
            b.insert(&item);
            assert!(b.contains(&item))
        }

        #[test]
        fn can_insert_string_bloom() {
            let mut b = Bloom::new(2, 1024, 10);
            let item: String = "hello world".to_owned();
            b.insert(&item);
            assert!(b.contains(&item))
        }
        #[test]
        fn test_slice_bloom() {
            let mut b = Bloom::new(3, 5, 10);
            let item: String = "hello world".to_owned();
            b.insert(&item);
            assert_eq!(b.field.count_ones(), 3);
        }
        #[test]
        fn does_not_contain() {
            let mut b = Bloom::new(2, 1024, 10);
            let upper = 100;
            for i in (0..upper).step_by(2) {
                b.insert(&i);
                assert_eq!(b.contains(&i), true);
            }
            for i in (1..upper).step_by(2) {
                assert_eq!(b.contains(&i), false);
            }
        }
        #[test]
        fn test_seeds() {
            let mut b1 = Bloom::new(2, 10, 0);
            let mut b2 = Bloom::new(2, 10, 1);
            b1.insert(&0);
            b2.insert(&0);
            assert_ne!(b1.field, b2.field);
        }
        #[test]
        fn can_insert_lots() {
            let mut b = Bloom::new(2, 1024, 10);
            for i in 0..1024 {
                b.insert(&i);
                assert!(b.contains(&i))
            }
        }
        #[test]
        fn test_fill_ratio() {
            let mut b = Bloom::new(2, 2, 0);
            let item: String = "hello world".to_owned();
            b.insert(&item);
            assert_eq!(b.fill_ratio_gte(0.5), true, "There's only two bits set!");
            assert_eq!(b.fill_ratio_gte(0.1), true);
            assert_eq!(b.fill_ratio_gte(0.50000000001), false);
        }
        #[test]
        fn slices_are_different() {
            let slice_len = 128;
            let mut b = Bloom::new(2, slice_len, 0);
            let item: String = "hello world".to_owned();
            b.insert(&item);
            assert_eq!(b.field[0..slice_len].len(), b.field[slice_len..].len());
            assert_ne!(b.field[0..slice_len], b.field[slice_len..]);
        }
    }

    mod test_growable {
        use crate::GrowableBloom;
        use serde_json;
        #[test]
        fn can_insert() {
            let mut b = GrowableBloom::new(0.05, 1000);
            let item = 20;
            b.insert(&item);
            assert!(b.contains(&item))
        }

        #[test]
        fn can_insert_string() {
            let mut b = GrowableBloom::new(0.05, 1000);
            let item: String = "hello world".to_owned();
            b.insert(&item);
            assert!(b.contains(&item))
        }

        #[test]
        fn does_not_contain() {
            let mut b = GrowableBloom::new(0.05, 1000);
            assert_eq!(b.contains(&"hello"), false);
            b.insert(&0);
            assert_eq!(b.contains(&"hello"), false);
            b.insert(&1);
            assert_eq!(b.contains(&"hello"), false);
            b.insert(&2);
            assert_eq!(b.contains(&"hello"), false);
        }
        #[test]
        fn can_insert_a_lot_of_elements() {
            let mut b = GrowableBloom::new(0.05, 1000);
            for i in 0..1000 {
                b.insert(&i);
                assert!(b.contains(&i));
            }
        }
        #[test]
        fn can_serialize_deserialize() {
            let mut b = GrowableBloom::new(0.05, 1000);
            b.insert(&0);
            let s = serde_json::to_string(&b).unwrap();
            let b_s: GrowableBloom = serde_json::from_str(&s).unwrap();
            assert!(b_s.contains(&0));
            assert_ne!(b_s.contains(&1), true);
            assert_ne!(b_s.contains(&1000), true);
        }
        #[test]
        fn verify_saturation() {
            let mut b = GrowableBloom::new(0.50, 100);
            for i in 0..1000 {
                b.insert(&i);
            }
            assert_eq!(b.contains(&10001), false)
        }
        #[test]
        fn test_types_saturation() {
            let mut b = GrowableBloom::new(0.50, 100);
            b.insert(&vec![1, 2, 3]);
            b.insert("hello");
            b.insert(&-1);
            b.insert(&0);
        }
    }

    mod bench {
        use crate::GrowableBloom;
        use test::Bencher;

        #[bench]
        fn bench_insert_normal_prob(b: &mut Bencher) {
            let mut gbloom = GrowableBloom::new(0.05, 1000);
            b.iter(|| gbloom.insert(10));
        }
        #[bench]
        fn bench_insert_small_prob(b: &mut Bencher) {
            let mut gbloom = GrowableBloom::new(0.0005, 1000);
            b.iter(|| gbloom.insert(10));
        }
        #[bench]
        fn bench_many(b: &mut Bencher) {
            let mut gbloom = GrowableBloom::new(0.05, 100000);
            b.iter(|| gbloom.insert(10));
        }
        #[bench]
        fn bench_insert_large(b: &mut Bencher) {
            let s: String = (0..10000).map(|_| 'X').collect();
            let mut gbloom = GrowableBloom::new(0.05, 100000);
            b.iter(|| gbloom.insert(&s))
        }
        #[bench]
        fn bench_insert_large_very_small_prob(b: &mut Bencher) {
            let s: String = (0..10000).map(|_| 'X').collect();
            let mut gbloom = GrowableBloom::new(0.000005, 100000);
            b.iter(|| gbloom.insert(&s))
        }
        #[bench]
        fn bench_grow(b: &mut Bencher) {
            let mut gbloom = GrowableBloom::new(0.90, 1);
            b.iter(|| {
                for i in 0..100 {
                    gbloom.insert(&i);
                    assert!(gbloom.contains(&i));
                }
            })
        }
    }
}
