#[cfg(feature = "serialize")]
use serde_json;

extern crate kiddo;

use kiddo::distance::squared_euclidean;
use kiddo::ErrorKind;
use kiddo::KdTree;

static POINT_A: ([f64; 2], usize) = ([0f64, 0f64], 0);
static POINT_B: ([f64; 2], usize) = ([1f64, 1f64], 1);
static POINT_C: ([f64; 2], usize) = ([2f64, 2f64], 2);
static POINT_D: ([f64; 2], usize) = ([3f64, 3f64], 3);

#[cfg(feature = "serialize")]
#[test]
fn it_serializes_and_deserializes_properly() {
    let capacity_per_node = 2;
    let mut kdtree = KdTree::with_capacity(capacity_per_node).unwrap();

    kdtree.add(&POINT_A.0, POINT_A.1).unwrap();
    kdtree.add(&POINT_B.0, POINT_B.1).unwrap();
    kdtree.add(&POINT_C.0, POINT_C.1).unwrap();
    kdtree.add(&POINT_D.0, POINT_D.1).unwrap();

    let serialized = serde_json::to_string(&kdtree).unwrap();
    println!("serialized: {:?}", &kdtree);

    let deserialized: KdTree<f64, usize, 2> = serde_json::from_str(&serialized).unwrap();
    println!("deserialized: {:?}", &deserialized);

    assert_eq!(deserialized.size(), 4);
    assert_eq!(
        deserialized
            .nearest(&POINT_A.0, 0, &squared_euclidean)
            .unwrap(),
        vec![]
    );
}
