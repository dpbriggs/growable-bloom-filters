#![deny(unsafe_code)]
#![cfg_attr(feature = "nightly", feature(test))]
///! Impl of Scalable Bloom Filters
///! http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.62.7953&rep=rep1&type=pdf

#[cfg(feature = "nightly")]
extern crate test;

use serde_derive::{Deserialize, Serialize};
use std::{
    hash::{Hash, Hasher},
    iter::Iterator,
};

/// Base Bloom Filter
#[derive(Deserialize, Serialize, PartialEq, Clone, Debug)]
struct Bloom {
    /// The actual bit field. Set to 0 with `Bloom::new`.
    #[serde(with = "serde_bytes")]
    buffer: Vec<u8>,
    /// The number of slices in the bloom filter.
    /// Equivalent to the hash function in the classic bloom filter.
    /// A single insertion will result in a single bit being set in each slice.
    num_slices: usize,
    /// The _bit_ length of each slice.
    slice_len: usize,
}

impl Bloom {
    /// Create a new Bloom filter
    ///
    /// # Arguments
    ///
    /// * `capacity` - target capacity.
    /// * `error_ratio` - false positive ratio [0..1.0].
    /// * `seed` - a seed to be used to initialize the hasher.
    fn new(capacity: usize, error_ratio: f64) -> Bloom {
        debug_assert!(capacity >= 1);
        debug_assert!(0.0 < error_ratio && error_ratio < 1.0);
        // directly from paper: k ~ log_2(1/desired_error_prob)
        let num_slices = ((1.0 / error_ratio).log2()).ceil() as usize;
        // re-arrange est_insertions ~ M(ln(2)^2 / ln(desired_error_prob))
        let opt_total_bits =
            (error_ratio.ln().abs() * capacity as f64 / 2f64.ln().powi(2)).ceil() as usize;
        // round up to the next byte
        let buffer_bytes = (opt_total_bits + 7) / 8;

        let mut buffer = Vec::with_capacity(buffer_bytes);
        buffer.resize(buffer_bytes, 0);
        let slice_len = buffer_bytes * 8 / num_slices;
        Bloom {
            buffer,
            num_slices,
            slice_len,
        }
    }

    /// Create an index iterator for a given item.
    ///
    /// This creates an iterator of `(byte idx, byte mask)` indices in the buffer.
    /// One per slice.
    ///
    /// # Arguments
    ///
    /// * `item` - The item to hash.
    fn index_iterator<T: Hash>(&self, item: T) -> impl Iterator<Item = (usize, u8)> {
        // Generate `self.num_slices` hashes from 2 hashes, using enhanced double hashing.
        let slice_len = self.slice_len;
        let (mut h1, mut h2) = double_hashing_hashes(item);
        (0..self.num_slices).map(move |i| {
            let hi = h1 % slice_len + i * slice_len;
            h1 = h1.wrapping_add(h2);
            h2 = h2.wrapping_add(i);
            (hi / 8, 1 << (hi % 8))
        })
    }

    /// Insert an *NEW* `item` into the Bloom.
    /// The caller is expected to check that the item is not contained
    /// in the filter before calling new.
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
        for (byte, mask) in self.index_iterator(item) {
            self.buffer[byte] |= mask;
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
            .all(|(byte, mask)| self.buffer[byte] & mask != 0)
    }
}

/// Returns 2 hashes for the given item
fn double_hashing_hashes<T: Hash>(item: T) -> (usize, usize) {
    let mut hs = ahash::AHasher::new_with_keys(
        0xe7b0c93ca8525013011d02b854ae8182,
        0x7bcc5cf9c39cec76fa336285d102d083,
    );
    item.hash(&mut hs);
    let h1 = hs.finish();

    hs = ahash::AHasher::new_with_keys(
        0x16f11fe89b0d677cb480a793d8e6c86c,
        0x6fe2e5aaf078ebc914f994a4c5259381,
    );
    item.hash(&mut hs);

    // h2 hash shouldn't be 0 for double hashing
    let h2 = hs.finish().max(1);

    (h1 as _, h2 as _)
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
///
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
    desired_error_prob: f64,
    est_insertions: usize,
    /// Number of items successfully inserted
    inserts: usize,
    /// Item capacity
    capacity: usize,
}

impl GrowableBloom {
    const GROWTH_FACTOR: usize = 2;
    const TIGHTENING_RATIO: f64 = 0.80;

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
        assert!(0.0 < desired_error_prob && desired_error_prob < 1.0);
        GrowableBloom {
            blooms: vec![],
            desired_error_prob,
            est_insertions,
            inserts: 0,
            capacity: 0,
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
    pub fn insert<T: Hash>(&mut self, item: T) -> bool {
        // Step 1: Ask if we already have it
        if self.contains(&item) {
            return false;
        }
        // Step 2: Grow if necessary
        if self.inserts >= self.capacity {
            self.grow();
        }
        // Step 3: Insert it into the last
        self.inserts += 1;
        let curr_bloom = self.blooms.last_mut().unwrap();
        curr_bloom.insert(&item);
        true
    }

    /// Clear the bloom filter.
    ///
    /// This does not resize the filter.
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
    /// bloom.clear();
    /// assert!(!bloom.contains(&item)); // No longer contains item
    /// ```
    pub fn clear(&mut self) {
        self.blooms.clear();
    }

    /// Whether this bloom filter contain any items.
    pub fn is_empty(&self) -> bool {
        self.blooms.is_empty()
    }

    /// Record if `item` already exists in the filter, and insert it if it doesn't already exist.
    ///
    /// Returns `true` if the item already existed in the filter.
    ///
    /// Note: This isn't faster than just inserting.
    ///
    /// # Example
    ///
    /// ```rust
    /// use growable_bloom_filter::GrowableBloom;
    /// let mut bloom = GrowableBloom::new(0.05, 10);
    /// let item = 0;
    ///
    /// let existed_before = bloom.check_and_set(&item);
    /// assert!(existed_before == false);
    /// let existed_before = bloom.check_and_set(&item);
    /// assert!(existed_before == true);
    /// ```
    pub fn check_and_set<T: Hash>(&mut self, item: T) -> bool {
        !self.insert(item)
    }

    /// Grow the GrowableBloom
    fn grow(&mut self) {
        let error_ratio =
            self.desired_error_prob * Self::TIGHTENING_RATIO.powi(self.blooms.len() as _);
        let capacity = self.est_insertions * Self::GROWTH_FACTOR.pow(self.blooms.len() as _);
        let new_bloom = Bloom::new(capacity, error_ratio);
        self.blooms.push(new_bloom);
        self.capacity += capacity;
    }
}

#[cfg(test)]
mod growable_bloom_tests {
    mod test_bloom {
        use crate::Bloom;

        #[test]
        fn can_insert_bloom() {
            let mut b = Bloom::new(100, 0.01);
            let item = 20;
            b.insert(&item);
            assert!(b.contains(&item))
        }

        #[test]
        fn can_insert_string_bloom() {
            let mut b = Bloom::new(100, 0.01);
            let item: String = "hello world".to_owned();
            b.insert(&item);
            assert!(b.contains(&item))
        }
        #[test]
        fn does_not_contain() {
            let mut b = Bloom::new(100, 0.01);
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
        fn can_insert_lots() {
            let mut b = Bloom::new(100, 0.01);
            for i in 0..1024 {
                b.insert(&i);
                assert!(b.contains(&i))
            }
        }
        #[test]
        fn test_refs() {
            let item = String::from("Hello World");
            let mut b = Bloom::new(100, 0.01);
            b.insert(&item);
            assert!(b.contains(&item));
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
            for &fp in &[0.01, 0.001] {
                // The paper gives an upper bound formula for the fp rate: fpUB <= fp0*/(1-r)
                // but for some reason the 0.001 fp bloom filter ends up with a bit higher fp than that
                // towards the end of the test, when it has 500+ times more items than the initial capacity.
                // I suspect that's due to our estimation for fill rate (by successful inserts) not being
                // fully accurate. The accurate way would be to check if 50+% of the bits of the filter are set,
                // but that's not practical performance wise.
                let fp_ub = fp / (1.0 - GrowableBloom::TIGHTENING_RATIO) * 1.33;

                let mut b = GrowableBloom::new(fp, 100);
                // insert 1000x more elements than initially allocated
                for i in 1..=100 * 1_000 {
                    b.insert(&i);

                    if i % 1_000 == 0 {
                        let est_fp_rate = (i + 1..).take(10_000).filter(|i| b.contains(i)).count()
                            as f64
                            / 10_000.0;

                        assert!(est_fp_rate <= fp_ub);
                    }
                }
                for i in 1..=100 * 1_000 {
                    assert!(b.contains(&i));
                }
            }
        }

        #[test]
        fn test_types_saturation() {
            let mut b = GrowableBloom::new(0.50, 100);
            b.insert(&vec![1, 2, 3]);
            b.insert("hello");
            b.insert(&-1);
            b.insert(&0);
        }

        #[test]
        fn can_check_and_set() {
            let mut b = GrowableBloom::new(0.05, 1000);
            let item = 20;
            assert!(!b.check_and_set(&item));
            assert!(b.check_and_set(&item));
        }
    }

    #[cfg(feature = "nightly")]
    mod bench {
        use crate::GrowableBloom;
        use test::Bencher;
        #[bench]
        fn bench_new(b: &mut Bencher) {
            b.iter(|| GrowableBloom::new(0.05, 1000));
        }
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
        fn bench_insert_string(b: &mut Bencher) {
            let s: String = (0..100).map(|_| 'X').collect();
            let mut gbloom = GrowableBloom::new(0.05, 100000);
            b.iter(|| gbloom.insert(&s))
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
