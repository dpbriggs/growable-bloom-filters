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
    num::NonZeroU64,
};

/// Base Bloom Filter
#[derive(Deserialize, Serialize, PartialEq, Clone, Debug)]
struct Bloom {
    /// The actual bit field. Set to 0 with `Bloom::new`.
    #[serde(rename = "b", with = "serde_bytes")]
    buffer: Box<[u8]>,
    /// The number of slices in the partitioned bloom filter.
    /// Equivalent to the number of hash function in the classic bloom filter.
    /// An insertion will result in a bit being set in each slice.
    #[serde(rename = "k")]
    num_slices: NonZeroU64,
}

impl Bloom {
    /// Create a new Bloom filter (specifically, a Partitioned Bloom filter)
    ///
    /// # Arguments
    ///
    /// * `capacity` - target capacity.
    /// * `error_ratio` - false positive ratio (0..1.0).
    /// * `seed` - a seed to be used to initialize the hasher.
    fn new(capacity: usize, error_ratio: f64) -> Bloom {
        // Directly from paper:
        // k = log2(1/P)   (num_slices)
        // n ≈ −m ln(1−p)  (slice_len_bits)
        // M = k * m       (total_bits)
        // for optimal filter p = 0.5, which gives:
        // n ≈ −m ln(0.5), rearranging: m = -n / ln(0.5), rearranging: m = n / log 2
        debug_assert!(capacity >= 1);
        debug_assert!(0.0 < error_ratio && error_ratio < 1.0);
        // We're using ceil instead of round in order to get an error rate <= the desired.
        // Using round can result in significantly higher error rates.
        let num_slices = ((1.0 / error_ratio).log2()).ceil() as u64;
        let slice_len_bits = (capacity as f64 / 2f64.ln()).ceil() as u64;
        let total_bits = num_slices * slice_len_bits;
        // round up to the next byte
        let buffer_bytes = ((total_bits + 7) / 8) as usize;

        let mut buffer = Vec::with_capacity(buffer_bytes);
        buffer.resize(buffer_bytes, 0);
        Bloom {
            buffer: buffer.into_boxed_slice(),
            num_slices: NonZeroU64::new(num_slices).unwrap(),
        }
    }

    /// Create an index iterator for a given item.
    ///
    /// This creates an iterator of pairs `(byte, mask)` indices in the buffer.
    /// The iterator will return one pair of indexes for each slice in the bloom filter.
    ///
    /// The pairs `(byte idx, byte mask)` are:
    ///     byte idx: byte idx in `self.buffer` to be extract for usage with the mask
    ///     byte mask: bit mask with a single bit set, can be ANDed (`&`) with
    ///                self.buffer[idx] to yield a number != 0 if the specified bit was set.
    ///                The mask can also be ORed (`|`) with the self.buffer[idx]
    ///                to set the corresponding bit.
    ///
    /// # Arguments
    ///
    /// * `item` - The item to hash.
    fn index_iterator<T: Hash>(&self, item: T) -> impl Iterator<Item = (usize, u8)> {
        // The _bit_ length (thus buffer.len() multiplied by 8) of each slice within buffer.
        // We'll use a NonZero type so that the compiler can avoid checking for
        // division/modulus by 0 inside the iterator.
        let slice_len = NonZeroU64::new(self.buffer.len() as u64 * 8 / self.num_slices).unwrap();

        // Generate `self.num_slices` hashes from 2 hashes, using enhanced double hashing.
        // See https://en.wikipedia.org/wiki/Double_hashing#Enhanced_double_hashing for details.
        // We choose to use 2x64 bit hashes instead of 2x32 ones as it gives significant better false positive ratios.
        let (mut h1, mut h2) = double_hashing_hashes(item);
        (0..self.num_slices.get()).map(move |i| {
            // Calculate hash(i)
            let hi = h1 % slice_len + i * slice_len.get();
            // Advance enhanced double hashing state
            h1 = h1.wrapping_add(h2);
            h2 = h2.wrapping_add(i);
            // Resulting index/mask based on hash(i)
            let idx = (hi / 8) as usize;
            let mask = 1u8 << (hi % 8);
            (idx, mask)
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
        // Set all bits (one per slice) corresponding to this item.
        //
        // Setting the bit:
        //    1000 0011 (self.buffer[idx])
        //    0001 0000 (mask)
        //    |---------
        //    1001 0011
        //
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
        // Check if all bits (one per slice) corresponding to this item are set.
        // See index_iterator comments for a detailed explanation.
        //
        // Potentially found case:
        //    0111 1111 (self.buffer[idx])
        //    0001 0000 (mask)
        //    &---------
        //    0001 0000 != 0
        //
        // Definitely not found case:
        //    1110 1111 (self.buffer[idx])
        //    0001 0000 (mask)
        //    &---------
        //    0000 0000 == 0
        //
        self.index_iterator(item)
            .all(|(byte, mask)| self.buffer[byte] & mask != 0)
    }
}

fn double_hashing_hashes<T: Hash>(item: T) -> (u64, u64) {
    // Using xxh3-64 with default seed/secret as a portable hasher.
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    item.hash(&mut hasher);
    let h1 = hasher.finish();

    // Write a nul byte to the existing state and get another hash.
    // This is appropriate when using a very high quality hasher,
    // which we know is the case.
    0u8.hash(&mut hasher);
    // h2 hash shouldn't be 0 for double hashing
    let h2 = hasher.finish().max(1);

    (h1, h2)
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
    #[serde(rename = "b")]
    blooms: Vec<Bloom>,
    #[serde(rename = "e")]
    desired_error_prob: f64,
    #[serde(rename = "t")]
    est_insertions: usize,
    /// Number of items successfully inserted
    #[serde(rename = "i")]
    inserts: usize,
    /// Item capacity
    #[serde(rename = "c")]
    capacity: usize,
}

impl GrowableBloom {
    // From the paper:
    // Considering the choice of s (GROWTH_FACTOR) = 2 for small expected growth and s = 4
    // for larger growth, one can see that r (TIGHTENING_RATIO) around 0.8 – 0.9 is a sensible choice.
    // Here we select good defaults for 10~1000x growth.
    const GROWTH_FACTOR: usize = 2;
    const TIGHTENING_RATIO: f64 = 0.8515625; // ~0.85 but has exact representation in f32/f64

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
        self.inserts = 0;
        self.capacity = 0;
    }

    /// Whether this bloom filter contain any items.
    pub fn is_empty(&self) -> bool {
        self.inserts == 0
    }

    /// The current estimated number of elements added to the filter.
    /// This is an estimation, so it may or may not increase after
    /// an insertion in the filter.
    ///
    /// # Example
    ///
    /// ```rust
    /// use growable_bloom_filter::GrowableBloom;
    /// let mut bloom = GrowableBloom::new(0.05, 10);
    ///
    /// bloom.insert(0);
    /// assert_eq!(bloom.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.inserts
    }

    /// The current estimated capacity of the filter.
    /// A filter starts with a capacity of 0 but will expand to accommodate more items.
    /// The actual ratio of increase depends on the values used to construct the bloom filter.
    ///
    /// # Example
    ///
    /// ```rust
    /// use growable_bloom_filter::GrowableBloom;
    /// let mut bloom = GrowableBloom::new(0.05, 10);
    ///
    /// assert_eq!(bloom.capacity(), 0);
    ///
    /// bloom.insert(0);
    /// assert_ne!(bloom.capacity(), 0);
    /// ```
    pub fn capacity(&self) -> usize {
        self.capacity
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
        // The paper gives an upper bound formula for the fp rate: fpUB <= fp0 * / (1-r)
        // This is because each sub bloom filter is created with an ever smaller
        // false-positive ratio, forming a geometric progression.
        // let r = TIGHTENING_RATIO
        // fpUB ~= fp0 * fp0*r * fp0*r*r * fp0*r*r*r ...
        // fp(x) = fp0 * (r**x)
        let error_ratio =
            self.desired_error_prob * Self::TIGHTENING_RATIO.powi(self.blooms.len() as _);
        // In order to have relatively small space overhead compared to a single appropriately sized bloom filter
        // the sub filters should be created with increasingly bigger sizes.
        // let s = GROWTH_FACTOR
        // cap(x) = cap0 * (s**x)
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
        fn len_capacity_clear() {
            let mut b = GrowableBloom::new(0.05, 1000);
            assert_eq!(b.len(), 0);
            assert_eq!(b.capacity(), 0);

            let item = 20;
            b.insert(&item);
            assert_ne!(b.len(), 0);
            assert_ne!(b.capacity(), 0);

            b.clear();
            assert_eq!(b.len(), 0);
            assert_eq!(b.capacity(), 0);
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
                let fp_ub = fp / (1.0 - GrowableBloom::TIGHTENING_RATIO);
                let initial_cap = 100u64;
                let growth = 1000u64;
                let mut b = GrowableBloom::new(fp, initial_cap as usize);
                // insert 1000x more elements than initially allocated
                for i in 1u64..=initial_cap * growth {
                    b.insert(&i);

                    if i % (initial_cap * growth / 10) == 0
                        || [1, 2, 5, 10, 25].iter().any(|&g| i == initial_cap * g)
                    {
                        // A lot of tests are required to get a good estimate
                        let est_fp_rate = (i + 1..).take(50_000).filter(|i| b.contains(i)).count()
                            as f64
                            / 50_000.0;

                        // Uncomment the following to get good output for experiments
                        // println!(
                        //     "{}x cap: {}fp ({}x)",
                        //     i / initial_cap,
                        //     est_fp_rate,
                        //     est_fp_rate / fp
                        // );
                        assert!(est_fp_rate <= fp_ub);
                    }
                }
                for i in 1u64..=initial_cap * growth {
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
        fn bench_insert_medium(b: &mut Bencher) {
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
