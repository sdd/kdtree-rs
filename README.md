# kiddo

> A kd tree library that has high performance and const-generic dimensions. A fork of kdtree, with significant performance improvements and extra features.
> Thanks and kudos to mrhooray for the original kdtree library on which kiddo is based.

* [Crate](https://crates.io/crates/kiddo)
* [Documentation](https://docs.rs/kiddo)
* [Repository](https://github.com/sdd/kiddo_v1)
* [Kiddo v2](https://github.com/sdd/kiddo)
* [Usage](#usage)
* [Benchmarks](#benchmarks)
* [License](#license)

## Update: Version 2 is now in beta!

You're looking at the documentation for Kiddo v1. Kiddo v2, currently in beta, is a complete rewrite from the ground up and provides very significant improvements in performance. The original v0.x and v1.x repository has been moved to https://github.com/sdd/kiddo_v1 and will receive only bug fixes and sporadic updates. Primary development has been focussed on v2 since August 2022 and that's where the focus will remain. If you are in a position to try out Kiddo v2 during the beta period, please do! Feedback is very much appreciated! The API of v2 is very similar to v1 and should not require many changes to existing code. Head over to https://github.com/sdd/kiddo to take a look.


## Differences vs kdtree@0.6.0

* The most significant structural difference is that kiddo has been written with the number of dimensions as a const generic parameter. This has a few benefits: many runtime errors (such as `WrongDimension` errors) are now compile time errors as the dimensionality is known at compile time, requiring all methods that have a point as a parameter to be an array or slice of length K. Operations that previously required the use of `Vec`s (such as the `distance_to_space()` function) now operate on arrays/slices, eliminating costly heap allocations.

* kiddo provides a specialised `nearest_one()` method, for queries that need the nearest one element only. This method avoids any heap allocations, performing much faster than a call to `nearest()` for a single point as a consequence.

* kiddo extends kdtree's query API by adding two new query methods: `best_n_within()` and `best_n_within_into_iter()`. These are useful for performing queries such as "what are the tallest 10 mountains within 10 degrees of London", "which are the largest 100 settlements within 5 degrees of New York", or "find the brightest 100 stars within a 2 degree radius of this point on the sky". This requires your stored element type to implement `PartialOrd` or `Ord`, and for smaller values to be "better". Bringing this functionality inside of kiddo's implementation, rather than requiring an initial `within()` query followed by a filter of the results, can be over 10x faster, as can be seen in the benchmarks below.

* kdtree's within() function uses a `BinaryHeap` to ensure that the results are ordered by distance from the query point. This sorting can be expensive, especially with a large number of elements. Kiddo's `within_unsorted()` method returns items in arbitrary order. For use cases that don't need the response to be sorted, this is much faster.

* Some small performance gains arise from using a technique used by some Python BinaryHeap libraries. Rather than `pop()`ing and then immediately `push()`ing to a `BinaryHeap`, it is quicker in this scenario to swap the element at the of the top of the heap and then bubble the new element down.

* The node structure has been refactored to use an `Enum` for aspects of the nodes that differ between stem and leaf nodes, rather than every node having all of these parameters present as `Option`s. This has two benefits. Firstly, stronger correctness guarantees. A type system as strong as Rust's allows us to eliminate the possibility of inconsistent state by design. Secondly, slightly better memory usage (also helped by using arrays rather than `Vec`s for things such as node min/max bounds, possible because of the const generic dimensionality).


## Usage
Add `kiddo` to `Cargo.toml`
```toml
[dependencies]
kiddo = "0.2.4"
```

Add points to kdtree and query nearest n points with distance function
```rust
use kiddo::KdTree;
use kiddo::ErrorKind;
use kiddo::distance::squared_euclidean;

let a: ([f64; 2], usize) = ([0f64, 0f64], 0);
let b: ([f64; 2], usize) = ([1f64, 1f64], 1);
let c: ([f64; 2], usize) = ([2f64, 2f64], 2);
let d: ([f64; 2], usize) = ([3f64, 3f64], 3);

let mut kdtree = KdTree::new()?;

kdtree.add(&a.0, a.1)?;
kdtree.add(&b.0, b.1)?;
kdtree.add(&c.0, c.1)?;
kdtree.add(&d.0, d.1)?;

assert_eq!(kdtree.size(), 4);


assert_eq!(
    kdtree.nearest(&a.0, 0, &squared_euclidean).unwrap(),
    vec![]
);
assert_eq!(
    kdtree.nearest(&a.0, 1, &squared_euclidean).unwrap(),
    vec![(0f64, &0)]
);
assert_eq!(
    kdtree.nearest(&a.0, 2, &squared_euclidean).unwrap(),
    vec![(0f64, &0), (2f64, &1)]
);
assert_eq!(
    kdtree.nearest(&a.0, 3, &squared_euclidean).unwrap(),
    vec![(0f64, &0), (2f64, &1), (8f64, &2)]
);
assert_eq!(
    kdtree.nearest(&a.0, 4, &squared_euclidean).unwrap(),
    vec![(0f64, &0), (2f64, &1), (8f64, &2), (18f64, &3)]
);
assert_eq!(
    kdtree.nearest(&a.0, 5, &squared_euclidean).unwrap(),
    vec![(0f64, &0), (2f64, &1), (8f64, &2), (18f64, &3)]
);
assert_eq!(
    kdtree.nearest(&b.0, 4, &squared_euclidean).unwrap(),
    vec![(0f64, &1), (2f64, &0), (2f64, &2), (8f64, &3)]
);
```

## Benchmarks

### Comparison with kdtree@0.6.0

Criterion is used to perform a series of benchmarks. Each action is benchmarked against trees that contain 100, 1,000, 10,000, 100,000 and 1,000,000 nodes, and charted below.

The `Adding Items` benchmarks are repeated against 2d, 3d and 4d trees. The 3d benchmarks are ran with points that are both of type `f32` and of type `f64`.

All of the remaining tests are only performed against 3d trees, for expediency. The trees are populated with random source data whose points are all on a unit sphere. This use case is representative of common kd-tree usages in geospatial and astronomical contexts.

The `Nearest n Items` tests query the tree for the nearest 1, 100 and 1,000 points at each tree size. The test for the common case of the nearest one point uses kiddo's `nearest_one()` method, which is an optimised method for this specific common use case.




#### Methodology

The results and charts below were created via the following process:

* check out the original-kdtree-criterion branch. This branch is the same code as kdtree@0.6.0, with criterion benchmarks added that perform the same operations as the criterion tests in kiddo. For functions that are present in kiddo but not in kdtree, the criterion tests for kdtree contain extra code to post-process the results from kdtree calls to perform the same actions as the new methods in kiddo.

* use the following command to run the criterion benchmarks for kdtree and generate NDJSON encoded test results:

```bash
cargo criterion --message-format json > criterion-kdtree.ndjson
```

* check out the master branch.

* use the following command to run the criterion benchmarks for kiddo and generate NDJSON encoded test results:

```bash
cargo criterion --message-format json --all-features > criterion-kiddo.ndjson
```

* the graphs are generated in python using matplotlib. Ensure you have python installed, as well as the matplotlib and ndjdon python lbraries. Then run the following:

```bash
python ./generate_benchmark_charts.py
```

#### Results

The following results were obtained with the above methodology on a machine with these specs:

* AMD Ryzen 5 2500X @ 3600MHz
* 32Gb DDR4 @ 3200MHz

The results are stored inside this repo as `criterion-kiddo.ndjson` and `criterion-kdtree.ndjson`, should you wish
to perform your own analysis.

##### Adding items to the tree
Kiddo generally has a very small performance lead over kdtree@0.6.0 at larger tree sizes, with their performance being similar on smaller trees.

![Charts showing benchmark results for adding items](https://raw.githubusercontent.com/sdd/kiddo_v1/master/benchmark_adding.png)


##### Retrieving the nearest n items

Kiddo's optimised `nearest_one()` method gives a huge performance advantage for single item queries, with up to 9x faster performance.
Kiddo's standard `nearest()` method also outperforms kdtree@0.6.0.

![Charts showing benchmark results for retrieving the nearest n items](https://raw.githubusercontent.com/sdd/kiddo_v1/master/benchmark_nearest_n.png)

##### Retrieving all items within a distance, sorted
Things look closer here at first glance but the logarithmic nature of the charted data may obscure the fact that Kiddo is often up to twice as fast as kdtree@0.6.0 here.

![Charts showing benchmark results for retrieving all items within a specified distance](https://raw.githubusercontent.com/sdd/kiddo_v1/master/benchmark_within.png)

##### Retrieving all items within a distance, unsorted
kdtree@0.6.0 does not have a `within_unsorted()` method, so we are comparing kiddo's `within_unsorted()` to kdtree@0.6.0's `within()` here, with kiddo up to 5x faster on the million-item tree.

![Charts showing benchmark results for retrieving all items within a specified distance](https://raw.githubusercontent.com/sdd/kiddo_v1/master/benchmark_within_unsorted.png)

##### Retrieving the best n items within a specified distance
Kiddo's performance advantage here ranges from twice as fast for hundred-item trees up to as much as 20x faster for trees with a million items.

![Charts showing benchmark results for retrieving the best n items within a specified distance](https://raw.githubusercontent.com/sdd/kiddo_v1/master/benchmark_best_n_within.png)

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
