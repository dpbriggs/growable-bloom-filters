
# Table of Contents

1.  [Growable Bloom Filters](#org83681a5)
    1.  [Overview](#orgba82373)
    2.  [Applications](#org5789605)

[![img](https://github.com/dpbriggs/growable-bloom-filters/workflows/Growable%20Bloom%20Filters/badge.svg)](https://github.com/dpbriggs/growable-bloom-filters)


<a id="org83681a5"></a>

# Growable Bloom Filters

[CRATES.IO](https://crates.io/crates/growable-bloom-filter) | [DOCUMENTATION](https://docs.rs/growable-bloom-filter/latest/growable_bloom_filter/struct.GrowableBloom.html)


<a id="orgba82373"></a>

## Overview

Implementation of [Scalable Bloom Filters](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.62.7953&rep=rep1&type=pdf) which also provides serde serialization and deserialize.

A bloom filter lets you `insert` items, and then test association with `contains`.
It's space and time efficient, at the cost of false positives.
In particular, if `contains` returns `true`, it may be in filter.
But if `contains` returns false, it's definitely not in the bloom filter.

You can control the failure rate by setting `desired_error_prob` and `est_insertions` appropriately.

    use serde_json;
    use growable_bloom_filter::GrowableBloom;
    
    let mut gbloom = GrowableBloom::new(0.05, 1000);
    gbloom.insert(&0);
    assert!(gbloom.contains(&0));
    
    let s = serde_json::to_string(&gbloom).unwrap();
    let des_gbloom: GrowableBloom = serde_json::from_str(&s).unwrap();
    assert!(des_gbloom.contains(&0));


<a id="org5789605"></a>

## Applications

Bloom filters are typically used as a pre-cache to avoid expensive operations.
For example, if you need to ask ten thousand servers if they have data XYZ,
you could use GrowableBloom to figure out which ones do NOT have XYZ.

