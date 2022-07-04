extern crate kiddo;

use kiddo::distance::squared_euclidean;
use kiddo::ErrorKind;
use kiddo::KdTree;

static POINT_A: ([f64; 2], usize) = ([0f64, 0f64], 0);
static POINT_B: ([f64; 2], usize) = ([1f64, 1f64], 1);
static POINT_C: ([f64; 2], usize) = ([2f64, 2f64], 2);
static POINT_D: ([f64; 2], usize) = ([3f64, 3f64], 3);

#[test]
fn it_works() {
    let capacity_per_node = 2;
    let mut kdtree = KdTree::with_per_node_capacity(capacity_per_node).unwrap();

    kdtree.add(&POINT_A.0, POINT_A.1).unwrap();
    kdtree.add(&POINT_B.0, POINT_B.1).unwrap();
    kdtree.add(&POINT_C.0, POINT_C.1).unwrap();
    kdtree.add(&POINT_D.0, POINT_D.1).unwrap();

    assert_eq!(kdtree.size(), 4);
    assert_eq!(
        kdtree.nearest(&POINT_A.0, 0, &squared_euclidean).unwrap(),
        vec![]
    );
    assert_eq!(
        kdtree.nearest(&POINT_A.0, 1, &squared_euclidean).unwrap(),
        vec![(0f64, &0)]
    );
    assert_eq!(
        kdtree.nearest(&POINT_A.0, 2, &squared_euclidean).unwrap(),
        vec![(0f64, &0), (2f64, &1)]
    );
    assert_eq!(
        kdtree.nearest(&POINT_A.0, 3, &squared_euclidean).unwrap(),
        vec![(0f64, &0), (2f64, &1), (8f64, &2)]
    );
    assert_eq!(
        kdtree.nearest(&POINT_A.0, 4, &squared_euclidean).unwrap(),
        vec![(0f64, &0), (2f64, &1), (8f64, &2), (18f64, &3)]
    );
    assert_eq!(
        kdtree.nearest(&POINT_A.0, 5, &squared_euclidean).unwrap(),
        vec![(0f64, &0), (2f64, &1), (8f64, &2), (18f64, &3)]
    );
    assert_eq!(
        kdtree.nearest(&POINT_B.0, 4, &squared_euclidean).unwrap(),
        vec![(0f64, &1), (2f64, &0), (2f64, &2), (8f64, &3)]
    );

    assert_eq!(
        kdtree.nearest_one(&POINT_A.0, &squared_euclidean).unwrap(),
        (0f64, &0)
    );
    assert_eq!(
        kdtree.nearest_one(&POINT_B.0, &squared_euclidean).unwrap(),
        (0f64, &1)
    );

    assert_eq!(
        kdtree.within(&POINT_A.0, 0.0, &squared_euclidean).unwrap(),
        vec![(0.0, &0)]
    );
    assert_eq!(
        kdtree.within(&POINT_B.0, 1.0, &squared_euclidean).unwrap(),
        vec![(0.0, &1)]
    );
    assert_eq!(
        kdtree.within(&POINT_B.0, 2.0, &squared_euclidean).unwrap(),
        vec![(0.0, &1), (2.0, &2), (2.0, &0)]
    );

    assert_eq!(
        kdtree
            .iter_nearest(&POINT_A.0, &squared_euclidean)
            .unwrap()
            .collect::<Vec<_>>(),
        vec![(0f64, &0), (2f64, &1), (8f64, &2), (18f64, &3)]
    );
}

#[test]
fn nearest_works_with_larger_example() {
    let vertices = vec![
        [68.91105387931276, 44.91668458105576],
        [-34.58731474945385, 71.74034814015292],
        [49.81627485893122, 13.441889056681546],
        [-50.870681576138466, 52.66452496629904],
        [-5.203156861316344, -20.99593863690137],
        [43.60999031840865, -77.39431889881345],
        [-2.587194808403382, 65.67594363600605],
        [17.32701517519223, -77.53994704470668],
        [-86.56502962282084, -88.9478905680815],
        [45.673095545318176, 33.796801087673686],
        [96.76125939245716, -16.121359071950366],
        [-29.246599975853428, 24.744789767875034],
        [-48.44195735538501, -16.431856700425442],
        [-50.88742960065202, 24.931374210927416],
        [96.27494950336964, 65.86741658775654],
        [-24.966035194707842, 95.0731027560162],
        [-76.39421787520853, 10.652314286279122],
    ];

    // test point is ~9 distance away from the last vertex inserted
    let test_point = [-75.06202391475783, 1.9077480803729827];

    let mut tree: KdTree<f64, (), 2> = KdTree::new();

    for vertex in vertices {
        tree.add(&vertex, ()).unwrap();
    }
    let nearest_distance = tree.nearest(&test_point, 1, &kiddo::distance::squared_euclidean).unwrap()[0].0.sqrt();

    assert_eq!(nearest_distance, 8.845460919462422);
}

#[test]
fn nearest_one_works_with_larger_example() {
    let vertices = vec![
        [68.91105387931276, 44.91668458105576],
        [-34.58731474945385, 71.74034814015292],
        [49.81627485893122, 13.441889056681546],
        [-50.870681576138466, 52.66452496629904],
        [-5.203156861316344, -20.99593863690137],
        [43.60999031840865, -77.39431889881345],
        [-2.587194808403382, 65.67594363600605],
        [17.32701517519223, -77.53994704470668],
        [-86.56502962282084, -88.9478905680815],
        [45.673095545318176, 33.796801087673686],
        [96.76125939245716, -16.121359071950366],
        [-29.246599975853428, 24.744789767875034],
        [-48.44195735538501, -16.431856700425442],
        [-50.88742960065202, 24.931374210927416],
        [96.27494950336964, 65.86741658775654],
        [-24.966035194707842, 95.0731027560162],
        [-76.39421787520853, 10.652314286279122],
    ];

    // test point is ~9 distance away from the last vertex inserted
    let test_point = [-75.06202391475783, 1.9077480803729827];

    let mut tree: KdTree<f64, (), 2> = KdTree::new();

    for vertex in vertices {
        tree.add(&vertex, ()).unwrap();
    }
    let nearest_distance = tree.nearest_one(&test_point, &kiddo::distance::squared_euclidean).unwrap().0.sqrt();

    assert_eq!(nearest_distance, 8.845460919462422);
}

#[test]
fn handles_non_finite_coordinate() {
    let point_a = ([std::f64::NAN, std::f64::NAN], 0f64);
    let point_b = ([std::f64::INFINITY, std::f64::INFINITY], 0f64);
    let mut kdtree = KdTree::with_per_node_capacity(1).unwrap();

    assert_eq!(
        kdtree.add(&point_a.0, point_a.1),
        Err(ErrorKind::NonFiniteCoordinate)
    );
    assert_eq!(
        kdtree.add(&point_b.0, point_b.1),
        Err(ErrorKind::NonFiniteCoordinate)
    );
    assert_eq!(
        kdtree.nearest(&point_b.0, 1, &squared_euclidean),
        Err(ErrorKind::NonFiniteCoordinate)
    );
    assert_eq!(
        kdtree.nearest(&point_a.0, 1, &squared_euclidean),
        Err(ErrorKind::NonFiniteCoordinate)
    );
}

#[test]
fn handles_singularity() {
    let mut kdtree = KdTree::with_per_node_capacity(1).unwrap();
    kdtree.add(&POINT_A.0, POINT_A.1).unwrap();
    kdtree.add(&POINT_A.0, POINT_A.1).unwrap();
    kdtree.add(&POINT_A.0, POINT_A.1).unwrap();
    kdtree.add(&POINT_B.0, POINT_B.1).unwrap();
    kdtree.add(&POINT_B.0, POINT_B.1).unwrap();
    kdtree.add(&POINT_B.0, POINT_B.1).unwrap();
    kdtree.add(&POINT_C.0, POINT_C.1).unwrap();
    kdtree.add(&POINT_C.0, POINT_C.1).unwrap();
    kdtree.add(&POINT_C.0, POINT_C.1).unwrap();
    assert_eq!(kdtree.size(), 9);
}

#[test]
fn handles_pending_order() {
    let item1 = ([0f64], 1);
    let item2 = ([100f64], 2);
    let item3 = ([45f64], 3);
    let item4 = ([55f64], 4);

    // Build a kd tree
    let capacity_per_node = 2;
    let mut kdtree = KdTree::with_per_node_capacity(capacity_per_node).unwrap();

    kdtree.add(&item1.0, item1.1).unwrap();
    kdtree.add(&item2.0, item2.1).unwrap();
    kdtree.add(&item3.0, item3.1).unwrap();
    kdtree.add(&item4.0, item4.1).unwrap();
    assert_eq!(
        kdtree.nearest(&[51f64], 2, &squared_euclidean).unwrap(),
        vec![(16.0, &4), (36.0, &3)]
    );
    assert_eq!(
        kdtree.nearest(&[51f64], 4, &squared_euclidean).unwrap(),
        vec![(16.0, &4), (36.0, &3), (2401.0, &2), (2601.0, &1)]
    );
    assert_eq!(
        kdtree.nearest(&[49f64], 2, &squared_euclidean).unwrap(),
        vec![(16.0, &3), (36.0, &4)]
    );
    assert_eq!(
        kdtree.nearest(&[49f64], 4, &squared_euclidean).unwrap(),
        vec![(16.0, &3), (36.0, &4), (2401.0, &1), (2601.0, &2)]
    );

    assert_eq!(
        kdtree
            .iter_nearest(&[49f64], &squared_euclidean)
            .unwrap()
            .collect::<Vec<_>>(),
        vec![(16.0, &3), (36.0, &4), (2401.0, &1), (2601.0, &2)]
    );
    assert_eq!(
        kdtree
            .iter_nearest(&[51f64], &squared_euclidean)
            .unwrap()
            .collect::<Vec<_>>(),
        vec![(16.0, &4), (36.0, &3), (2401.0, &2), (2601.0, &1)]
    );

    assert_eq!(
        kdtree.within(&[50f64], 1.0, &squared_euclidean).unwrap(),
        vec![]
    );
    assert_eq!(
        kdtree.within(&[50f64], 25.0, &squared_euclidean).unwrap(),
        vec![(25.0, &3), (25.0, &4)]
    );
    assert_eq!(
        kdtree.within(&[50f64], 30.0, &squared_euclidean).unwrap(),
        vec![(25.0, &3), (25.0, &4)]
    );
    assert_eq!(
        kdtree.within(&[55f64], 5.0, &squared_euclidean).unwrap(),
        vec![(0.0, &4)]
    );
    assert_eq!(
        kdtree.within(&[56f64], 5.0, &squared_euclidean).unwrap(),
        vec![(1.0, &4)]
    );
}

#[test]
fn handles_drops_correctly() {
    use std::ops::Drop;
    use std::sync::{Arc, Mutex};

    // Mock up a structure to keep track of Drops
    struct Test(Arc<Mutex<i32>>);

    impl PartialEq<Test> for Test {
        fn eq(&self, other: &Test) -> bool {
            *self.0.lock().unwrap() == *other.0.lock().unwrap()
        }
    }

    impl Drop for Test {
        fn drop(&mut self) {
            let mut drop_counter = self.0.lock().unwrap();
            *drop_counter += 1;
        }
    }

    let drop_counter = Arc::new(Mutex::new(0));

    let item1 = ([0f64, 0f64], Test(drop_counter.clone()));
    let item2 = ([1f64, 1f64], Test(drop_counter.clone()));
    let item3 = ([2f64, 2f64], Test(drop_counter.clone()));
    let item4 = ([3f64, 3f64], Test(drop_counter.clone()));

    {
        // Build a kd tree
        let capacity_per_node = 1;
        let mut kdtree = KdTree::with_per_node_capacity(capacity_per_node).unwrap();

        kdtree.add(&item1.0, item1.1).unwrap();
        kdtree.add(&item2.0, item2.1).unwrap();
        kdtree.add(&item3.0, item3.1).unwrap();
        kdtree.add(&item4.0, item4.1).unwrap();

        // Pre-drop check
        assert_eq!(*drop_counter.lock().unwrap(), 0);
    }

    // Post-drop check
    assert_eq!(*drop_counter.lock().unwrap(), 4);
}

#[test]
fn handles_remove_correctly() {
    let item1 = ([0f64], 1);
    let item2 = ([100f64], 2);
    let item3 = ([45f64], 3);
    let item4 = ([55f64], 4);
    let item5 = ([45f64], 5);

    // Build a kd tree
    let capacity_per_node = 2;
    let mut kdtree = KdTree::with_per_node_capacity(capacity_per_node).unwrap();

    kdtree.add(&item1.0, item1.1).unwrap();
    kdtree.add(&item2.0, item2.1).unwrap();
    kdtree.add(&item3.0, item3.1).unwrap();
    kdtree.add(&item4.0, item4.1).unwrap();
    kdtree.add(&item5.0, item5.1).unwrap();

    let num_removed = kdtree.remove(&&item3.0, &item3.1).unwrap();
    assert_eq!(kdtree.size(), 4);
    assert_eq!(num_removed, 1);
    assert_eq!(
        kdtree.nearest(&[51f64], 3, &squared_euclidean).unwrap(),
        vec![(16.0, &4), (36.0, &5), (2401.0, &2)]
    );
}

#[test]
fn handles_remove_multiple_match() {
    let item1 = ([0f64], 1);
    let item2 = ([0f64], 1);
    let item3 = ([100f64], 2);
    let item4 = ([45f64], 3);

    // Build a kd tree
    let capacity_per_node = 2;
    let mut kdtree = KdTree::with_per_node_capacity(capacity_per_node).unwrap();

    kdtree.add(&item1.0, item1.1).unwrap();
    kdtree.add(&item2.0, item2.1).unwrap();
    kdtree.add(&item3.0, item3.1).unwrap();
    kdtree.add(&item4.0, item4.1).unwrap();

    assert_eq!(kdtree.size(), 4);
    let num_removed = kdtree.remove(&&[0f64], &1).unwrap();
    assert_eq!(kdtree.size(), 2);
    assert_eq!(num_removed, 2);
    assert_eq!(
        kdtree.nearest(&[45f64], 1, &squared_euclidean).unwrap(),
        vec![(0.0, &3)]
    );
}

#[test]
fn handles_remove_no_match() {
    let item1 = ([0f64], 1);
    let item2 = ([100f64], 2);
    let item3 = ([45f64], 3);
    let item4 = ([55f64], 4);

    // Build a kd tree
    let capacity_per_node = 2;
    let mut kdtree = KdTree::with_per_node_capacity(capacity_per_node).unwrap();

    kdtree.add(&item1.0, item1.1).unwrap();
    kdtree.add(&item2.0, item2.1).unwrap();
    kdtree.add(&item3.0, item3.1).unwrap();
    kdtree.add(&item4.0, item4.1).unwrap();

    let num_removed = kdtree.remove(&&[1f64], &2).unwrap();
    assert_eq!(kdtree.size(), 4);
    assert_eq!(num_removed, 0);
    assert_eq!(
        kdtree.nearest(&[51f64], 2, &squared_euclidean).unwrap(),
        vec![(16.0, &4), (36.0, &3)]
    );
}

#[test]
fn error_messages_do_not_overflow_stack() {
    format!("{}", ErrorKind::NonFiniteCoordinate);
    format!("{}", ErrorKind::ZeroCapacity);
    format!("{}", ErrorKind::Empty);
}

#[test]
fn test_periodic_1d_nearest() {

    // data, query, knns are stored on the heap here because test threads have small stack sizes
    // in a real application, these could be stored in arrays.

    use rand::distributions::{Distribution, Uniform};
    use rayon::prelude::*;

    // Initialize dimensionality and data specs
    const K: usize = 1;
    const NQUERY: usize = 10_000;
    const NDATA: usize = 1_000;
    const PERIODIC: [f64; K] = [1.0; K];

    // Initialize rng
    let uniform = Uniform::new(0.0, 1.0);
    let mut rng = rand::thread_rng();

    // Initialize KdTree
    let mut tree = KdTree::periodic_with_per_node_capacity(32, &PERIODIC).unwrap();

    // Initialize data points (aka point cloud)
    let mut data = vec![[0.0; K]; NDATA];
    for i in 0..NDATA {
        for j in 0..K {

            // Save to array to compare with brute force
            data[i][j] = uniform.sample(&mut rng);
        }

        // Add to tree
        tree.add(&data[i], i).expect("Couldn't add to tree");
    }

    // Initialize query points
    let mut query = vec![[0.0; K]; NQUERY];
    for i in 0..NQUERY {
        for j in 0..K {
            // Save to array to compare with brute force
            query[i][j] = uniform.sample(&mut rng);
        }
    }

    // Query points
    let knns: Vec<(f64, &usize)> = (&query)
        .into_par_iter()
        .map_with(&tree, |t, q| {
            t.nearest_one(&q, &squared_euclidean).unwrap()
        }).collect::<Vec<_>>();

    // Check vs brute force
    for (query_index, q) in query.iter().enumerate() {
        
        // Initialize neighbor
        let mut neighbor = (std::f64::MAX, std::usize::MAX);

        for (data_index, d) in data.iter().enumerate() {

            // Initialize min
            let mut min: f64 = std::f64::MAX;

            // Calculate distance for every image lazily (i.e. 3^K instead of 2^K)
            for image_idx in 0..3_i32.pow(K as u32) {

                // Initialize current_image template
                let mut current_image: [i32; K] = [0; K];

                // Calculate current image
                for idx in 0..K {
                    current_image[idx] = (( image_idx / 3_i32.pow(idx as u32)) % 3) - 1;
                }

                // Construct current image position
                let mut image: [f64; K] = q.clone();
                for idx in 0..K {
                    image[idx] += (current_image[idx] as f64)*PERIODIC[idx];
                }

                // Calculate distance for this image
                let image_distance = squared_euclidean(&image, &d);

                // Compare with current min
                min = min.min(image_distance);
            }
            
            // If this is the new closest neighbor, replace 
            if min < neighbor.0 {
                neighbor = (min, data_index)
            }
        }

        // Check that brute force result agrees with KdTree result
        assert_eq!(&neighbor.1, knns[query_index].1, "{} {}", neighbor.0, knns[query_index].0);
        assert!((neighbor.0 - knns[query_index].0).abs() < std::f64::EPSILON, "{} {}", neighbor.0, knns[query_index].0);
    }
}


#[test]
fn test_periodic_2d_nearest() {

    // data, query, knns are stored on the heap here because test threads have small stack sizes
    // in a real application, these could be stored in arrays.

    use rand::distributions::{Distribution, Uniform};
    use rayon::prelude::*;

    // Initialize dimensionality and data specs
    const K: usize = 2;
    const NQUERY: usize = 10_000;
    const NDATA: usize = 1_000;
    const PERIODIC: [f64; K] = [1.0; K];

    // Initialize rng
    let uniform = Uniform::new(0.0, 1.0);
    let mut rng = rand::thread_rng();

    // Initialize KdTree
    let mut tree = KdTree::periodic_with_per_node_capacity(32, &PERIODIC).unwrap();

    // Initialize data points (aka point cloud)
    let mut data = vec![[0.0; K]; NDATA];
    for i in 0..NDATA {
        for j in 0..K {

            // Save to array to compare with brute force
            data[i][j] = uniform.sample(&mut rng);
        }

        // Add to tree
        tree.add(&data[i], i).expect("Couldn't add to tree");
    }

    // Initialize query points
    let mut query = vec![[0.0; K]; NQUERY];
    for i in 0..NQUERY {
        for j in 0..K {
            // Save to array to compare with brute force
            query[i][j] = uniform.sample(&mut rng);
        }
    }

    // Query points
    let knns: Vec<(f64, &usize)> = (&query)
        .into_par_iter()
        .map_with(&tree, |t, q| {
            t.nearest_one(&q, &squared_euclidean).unwrap()
        }).collect::<Vec<_>>();

    // Check vs brute force
    for (query_index, q) in query.iter().enumerate() {
        
        // Initialize neighbor
        let mut neighbor = (std::f64::MAX, std::usize::MAX);

        for (data_index, d) in data.iter().enumerate() {

            // Initialize min
            let mut min: f64 = std::f64::MAX;

            // Calculate distance for every image lazily (i.e. 3^K instead of 2^K)
            for image_idx in 0..3_i32.pow(K as u32) {

                // Initialize current_image template
                let mut current_image: [i32; K] = [0; K];

                // Calculate current image
                for idx in 0..K {
                    current_image[idx] = (( image_idx / 3_i32.pow(idx as u32)) % 3) - 1;
                }

                // Construct current image position
                let mut image: [f64; K] = q.clone();
                for idx in 0..K {
                    image[idx] += (current_image[idx] as f64)*PERIODIC[idx];
                }

                // Calculate distance for this image
                let image_distance = squared_euclidean(&image, &d);

                // Compare with current min
                min = min.min(image_distance);
            }
            
            // If this is the new closest neighbor, replace 
            if min < neighbor.0 {
                neighbor = (min, data_index)
            }
        }

        // Check that brute force result agrees with KdTree result
        assert_eq!(&neighbor.1, knns[query_index].1, "{} {}", neighbor.0, knns[query_index].0);
        assert!((neighbor.0 - knns[query_index].0).abs() < std::f64::EPSILON, "{} {}", neighbor.0, knns[query_index].0);
    }
}


#[test]
fn test_periodic_3d_nearest() {

    // data, query, knns are stored on the heap here because test threads have small stack sizes
    // in a real application, these could be stored in arrays.

    use rand::distributions::{Distribution, Uniform};
    use rayon::prelude::*;

    // Initialize dimensionality and data specs
    const K: usize = 3;
    const NQUERY: usize = 10_000;
    const NDATA: usize = 1_000;
    const PERIODIC: [f64; K] = [1.0; K];

    // Initialize rng
    let uniform = Uniform::new(0.0, 1.0);
    let mut rng = rand::thread_rng();

    // Initialize KdTree
    let mut tree = KdTree::periodic_with_per_node_capacity(32, &PERIODIC).unwrap();

    // Initialize data points (aka point cloud)
    let mut data = vec![[0.0; K]; NDATA];
    for i in 0..NDATA {
        for j in 0..K {

            // Save to array to compare with brute force
            data[i][j] = uniform.sample(&mut rng);
        }

        // Add to tree
        tree.add(&data[i], i).expect("Couldn't add to tree");
    }

    // Initialize query points
    let mut query = vec![[0.0; K]; NQUERY];
    for i in 0..NQUERY {
        for j in 0..K {
            // Save to array to compare with brute force
            query[i][j] = uniform.sample(&mut rng);
        }
    }

    // Query points
    let knns: Vec<(f64, &usize)> = (&query)
        .into_par_iter()
        .map_with(&tree, |t, q| {
            t.nearest_one(&q, &squared_euclidean).unwrap()
        }).collect::<Vec<_>>();

    // Check vs brute force
    for (query_index, q) in query.iter().enumerate() {
        
        // Initialize neighbor
        let mut neighbor = (std::f64::MAX, std::usize::MAX);

        for (data_index, d) in data.iter().enumerate() {

            // Initialize min
            let mut min: f64 = std::f64::MAX;

            // Calculate distance for every image lazily (i.e. 3^K instead of 2^K)
            for image_idx in 0..3_i32.pow(K as u32) {

                // Initialize current_image template
                let mut current_image: [i32; K] = [0; K];

                // Calculate current image
                for idx in 0..K {
                    current_image[idx] = (( image_idx / 3_i32.pow(idx as u32)) % 3) - 1;
                }

                // Construct current image position
                let mut image: [f64; K] = q.clone();
                for idx in 0..K {
                    image[idx] += (current_image[idx] as f64)*PERIODIC[idx];
                }

                // Calculate distance for this image
                let image_distance = squared_euclidean(&image, &d);

                // Compare with current min
                min = min.min(image_distance);
            }
            
            // If this is the new closest neighbor, replace 
            if min < neighbor.0 {
                neighbor = (min, data_index)
            }
        }

        // Check that brute force result agrees with KdTree result
        assert_eq!(&neighbor.1, knns[query_index].1, "{} {}", neighbor.0, knns[query_index].0);
        assert!((neighbor.0 - knns[query_index].0).abs() < std::f64::EPSILON, "{} {}", neighbor.0, knns[query_index].0);
    }
}




#[test]
fn test_periodic_1d_nearest_n() {

    // data, query, knns are stored on the heap here because test threads have small stack sizes
    // in a real application, these could be stored in arrays.

    use rand::distributions::{Distribution, Uniform};
    use rayon::prelude::*;

    // Number of neighbors
    const N: usize = 5;

    // Initialize dimensionality and data specs
    const K: usize = 1;
    const NQUERY: usize = 10_000;
    const NDATA: usize = 1_000;
    const PERIODIC: [f64; K] = [1.0; K];

    // Initialize rng
    let uniform = Uniform::new(0.0, 1.0);
    let mut rng = rand::thread_rng();

    // Initialize KdTree
    let mut tree = KdTree::periodic_with_per_node_capacity(32, &PERIODIC).unwrap();

    // Initialize data points (aka point cloud)
    let mut data = vec![[0.0; K]; NDATA];
    for i in 0..NDATA {
        for j in 0..K {

            // Save to array to compare with brute force
            data[i][j] = uniform.sample(&mut rng);
        }

        // Add to tree
        tree.add(&data[i], i).expect("Couldn't add to tree");
    }

    // Initialize query points
    let mut query = vec![[0.0; K]; NQUERY];
    for i in 0..NQUERY {
        for j in 0..K {
            // Save to array to compare with brute force
            query[i][j] = uniform.sample(&mut rng);
        }
    }

    // Query points
    let knns: Vec<Vec<(f64, &usize)>> = (&query)
        .into_par_iter()
        .map_with(&tree, |t, q| {
            t.nearest(&q, N, &squared_euclidean).unwrap()
        }).collect::<Vec<_>>();

    // Check vs brute force
    for (query_index, q) in query.iter().enumerate() {
        
        // Initialize neighbor
        let mut neighbors = [(std::f64::MAX, std::usize::MAX); N];

        for (data_index, d) in data.iter().enumerate() {

            // Initialize min
            let mut min: f64 = std::f64::MAX;

            // Calculate distance for every image lazily (i.e. 3^K instead of 2^K)
            for image_idx in 0..3_i32.pow(K as u32) {

                // Initialize current_image template
                let mut current_image: [i32; K] = [0; K];

                // Calculate current image
                for idx in 0..K {
                    current_image[idx] = (( image_idx / 3_i32.pow(idx as u32)) % 3) - 1;
                }

                // Construct current image position
                let mut image: [f64; K] = q.clone();
                for idx in 0..K {
                    image[idx] += (current_image[idx] as f64)*PERIODIC[idx];
                }

                // Calculate distance for this image
                let image_distance = squared_euclidean(&image, &d);

                // Compare with current min
                min = min.min(image_distance);
            }
            
            // If this is closer than the current Nth neighbor, replace and sort
            if min < neighbors[N-1].0 {
                neighbors[N-1] = (min, data_index);
                neighbors = sort_array(neighbors);
            }
        }

        // Check that brute force result agrees with KdTree result
        for (k, neighbor) in neighbors.iter().enumerate() {
            assert_eq!(&neighbor.1, knns[query_index][k].1, "{} {}", neighbor.0, knns[query_index][k].0);
            assert!((neighbor.0 - knns[query_index][k].0).abs() < std::f64::EPSILON, "{} {}", neighbor.0, knns[query_index][k].0);
        }
    }
}


#[test]
fn test_periodic_2d_nearest_n() {

    // data, query, knns are stored on the heap here because test threads have small stack sizes
    // in a real application, these could be stored in arrays.

    use rand::distributions::{Distribution, Uniform};
    use rayon::prelude::*;

    // Number of neighbors
    const N: usize = 5;

    // Initialize dimensionality and data specs
    const K: usize = 2;
    const NQUERY: usize = 10_000;
    const NDATA: usize = 1_000;
    const PERIODIC: [f64; K] = [1.0; K];

    // Initialize rng
    let uniform = Uniform::new(0.0, 1.0);
    let mut rng = rand::thread_rng();

    // Initialize KdTree
    let mut tree = KdTree::periodic_with_per_node_capacity(32, &PERIODIC).unwrap();

    // Initialize data points (aka point cloud)
    let mut data = vec![[0.0; K]; NDATA];
    for i in 0..NDATA {
        for j in 0..K {

            // Save to array to compare with brute force
            data[i][j] = uniform.sample(&mut rng);
        }

        // Add to tree
        tree.add(&data[i], i).expect("Couldn't add to tree");
    }

    // Initialize query points
    let mut query = vec![[0.0; K]; NQUERY];
    for i in 0..NQUERY {
        for j in 0..K {
            // Save to array to compare with brute force
            query[i][j] = uniform.sample(&mut rng);
        }
    }

    // Query points
    let knns: Vec<Vec<(f64, &usize)>> = (&query)
        .into_par_iter()
        .map_with(&tree, |t, q| {
            t.nearest(&q, N, &squared_euclidean).unwrap()
        }).collect::<Vec<_>>();

    // Check vs brute force
    for (query_index, q) in query.iter().enumerate() {
        
        // Initialize neighbor
        let mut neighbors = [(std::f64::MAX, std::usize::MAX); N];

        for (data_index, d) in data.iter().enumerate() {

            // Initialize min
            let mut min: f64 = std::f64::MAX;

            // Calculate distance for every image lazily (i.e. 3^K instead of 2^K)
            for image_idx in 0..3_i32.pow(K as u32) {

                // Initialize current_image template
                let mut current_image: [i32; K] = [0; K];

                // Calculate current image
                for idx in 0..K {
                    current_image[idx] = (( image_idx / 3_i32.pow(idx as u32)) % 3) - 1;
                }

                // Construct current image position
                let mut image: [f64; K] = q.clone();
                for idx in 0..K {
                    image[idx] += (current_image[idx] as f64)*PERIODIC[idx];
                }

                // Calculate distance for this image
                let image_distance = squared_euclidean(&image, &d);

                // Compare with current min
                min = min.min(image_distance);
            }
            
            // If this is closer than the current Nth neighbor, replace and sort
            if min < neighbors[N-1].0 {
                neighbors[N-1] = (min, data_index);
                neighbors = sort_array(neighbors);
            }
        }

        // Check that brute force result agrees with KdTree result
        for (k, neighbor) in neighbors.iter().enumerate() {
            assert_eq!(&neighbor.1, knns[query_index][k].1, "{} {}", neighbor.0, knns[query_index][k].0);
            assert!((neighbor.0 - knns[query_index][k].0).abs() < std::f64::EPSILON, "{} {}", neighbor.0, knns[query_index][k].0);
        }
    }
}


#[test]
fn test_periodic_3d_nearest_n() {

    // data, query, knns are stored on the heap here because test threads have small stack sizes
    // in a real application, these could be stored in arrays.

    use rand::distributions::{Distribution, Uniform};
    use rayon::prelude::*;

    // Number of neighbors
    const N: usize = 5;

    // Initialize dimensionality and data specs
    const K: usize = 3;
    const NQUERY: usize = 10_000;
    const NDATA: usize = 1_000;
    const PERIODIC: [f64; K] = [1.0; K];

    // Initialize rng
    let uniform = Uniform::new(0.0, 1.0);
    let mut rng = rand::thread_rng();

    // Initialize KdTree
    let mut tree = KdTree::periodic_with_per_node_capacity(32, &PERIODIC).unwrap();

    // Initialize data points (aka point cloud)
    let mut data = vec![[0.0; K]; NDATA];
    for i in 0..NDATA {
        for j in 0..K {

            // Save to array to compare with brute force
            data[i][j] = uniform.sample(&mut rng);
        }

        // Add to tree
        tree.add(&data[i], i).expect("Couldn't add to tree");
    }

    // Initialize query points
    let mut query = vec![[0.0; K]; NQUERY];
    for i in 0..NQUERY {
        for j in 0..K {
            // Save to array to compare with brute force
            query[i][j] = uniform.sample(&mut rng);
        }
    }

    // Query points
    let knns: Vec<Vec<(f64, &usize)>> = (&query)
        .into_par_iter()
        .map_with(&tree, |t, q| {
            t.nearest(&q, N, &squared_euclidean).unwrap()
        }).collect::<Vec<_>>();

    // Check vs brute force
    for (query_index, q) in query.iter().enumerate() {
        
        // Initialize neighbor
        let mut neighbors = [(std::f64::MAX, std::usize::MAX); N];

        for (data_index, d) in data.iter().enumerate() {

            // Initialize min
            let mut min: f64 = std::f64::MAX;

            // Calculate distance for every image lazily (i.e. 3^K instead of 2^K)
            for image_idx in 0..3_i32.pow(K as u32) {

                // Initialize current_image template
                let mut current_image: [i32; K] = [0; K];

                // Calculate current image
                for idx in 0..K {
                    current_image[idx] = (( image_idx / 3_i32.pow(idx as u32)) % 3) - 1;
                }

                // Construct current image position
                let mut image: [f64; K] = q.clone();
                for idx in 0..K {
                    image[idx] += (current_image[idx] as f64)*PERIODIC[idx];
                }

                // Calculate distance for this image
                let image_distance = squared_euclidean(&image, &d);

                // Compare with current min
                min = min.min(image_distance);
            }
            
            // If this is closer than the current Nth neighbor, replace and sort
            if min < neighbors[N-1].0 {
                neighbors[N-1] = (min, data_index);
                neighbors = sort_array(neighbors);
            }
        }

        // Check that brute force result agrees with KdTree result
        for (k, neighbor) in neighbors.iter().enumerate() {
            assert_eq!(&neighbor.1, knns[query_index][k].1, "{} {}", neighbor.0, knns[query_index][k].0);
            assert!((neighbor.0 - knns[query_index][k].0).abs() < std::f64::EPSILON, "{} {}", neighbor.0, knns[query_index][k].0);
        }

    }
}



#[test]
fn test_periodic_1d_within() {

    // data, query, knns are stored on the heap here because test threads have small stack sizes
    // in a real application, these could be stored in arrays.

    use rand::distributions::{Distribution, Uniform};
    use rayon::prelude::*;

    // Radius to test
    const RADIUS: f64 = 0.05;

    // Initialize dimensionality and data specs
    const K: usize = 1;
    const NQUERY: usize = 10_000;
    const NDATA: usize = 1_000;
    const PERIODIC: [f64; K] = [1.0; K];

    // Initialize rng
    let uniform = Uniform::new(0.0, 1.0);
    let mut rng = rand::thread_rng();

    // Initialize KdTree
    let mut tree = KdTree::periodic_with_per_node_capacity(32, &PERIODIC).unwrap();

    // Initialize data points (aka point cloud)
    let mut data = vec![[0.0; K]; NDATA];
    for i in 0..NDATA {
        for j in 0..K {

            // Save to array to compare with brute force
            data[i][j] = uniform.sample(&mut rng);
        }

        // Add to tree
        tree.add(&data[i], i).expect("Couldn't add to tree");
    }

    // Initialize query points
    let mut query = vec![[0.0; K]; NQUERY];
    for i in 0..NQUERY {
        for j in 0..K {
            // Save to array to compare with brute force
            query[i][j] = uniform.sample(&mut rng);
        }
    }

    // Query points
    let knns: Vec<Vec<(f64, &usize)>> = (&query)
        .into_par_iter()
        .map_with(&tree, |t, q| {
            t.within(&q, RADIUS, &squared_euclidean).unwrap()
        }).collect::<Vec<_>>();

    // Check vs brute force
    for (query_index, q) in query.iter().enumerate() {
        
        // Initialize neighbor
        let mut neighbors = vec![];

        for (data_index, d) in data.iter().enumerate() {

            // Initialize min
            let mut min: f64 = std::f64::MAX;

            // Calculate distance for every image lazily (i.e. 3^K instead of 2^K)
            for image_idx in 0..3_i32.pow(K as u32) {

                // Initialize current_image template
                let mut current_image: [i32; K] = [0; K];

                // Calculate current image
                for idx in 0..K {
                    current_image[idx] = (( image_idx / 3_i32.pow(idx as u32)) % 3) - 1;
                }

                // Construct current image position
                let mut image: [f64; K] = q.clone();
                for idx in 0..K {
                    image[idx] += (current_image[idx] as f64)*PERIODIC[idx];
                }

                // Calculate distance for this image
                let image_distance = squared_euclidean(&image, &d);

                // Compare with current min
                min = min.min(image_distance);
            }
            
            // If this within the target radius, append
            if min < RADIUS {
                neighbors.push((min, data_index));
            }
        }

        // Sort neighbors
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Check that brute force result agrees with KdTree result
        for (k, neighbor) in neighbors.iter().enumerate() {
            assert_eq!(&neighbor.1, knns[query_index][k].1, "{} {}", neighbor.0, knns[query_index][k].0);
            assert!((neighbor.0 - knns[query_index][k].0).abs() < std::f64::EPSILON, "{} {}", neighbor.0, knns[query_index][k].0);
        }
    }
}


#[test]
fn test_periodic_2d_within() {

    // data, query, knns are stored on the heap here because test threads have small stack sizes
    // in a real application, these could be stored in arrays.

    use rand::distributions::{Distribution, Uniform};
    use rayon::prelude::*;

    // Radius to test
    const RADIUS: f64 = 0.05;

    // Initialize dimensionality and data specs
    const K: usize = 2;
    const NQUERY: usize = 10_000;
    const NDATA: usize = 1_000;
    const PERIODIC: [f64; K] = [1.0; K];

    // Initialize rng
    let uniform = Uniform::new(0.0, 1.0);
    let mut rng = rand::thread_rng();

    // Initialize KdTree
    let mut tree = KdTree::periodic_with_per_node_capacity(32, &PERIODIC).unwrap();

    // Initialize data points (aka point cloud)
    let mut data = vec![[0.0; K]; NDATA];
    for i in 0..NDATA {
        for j in 0..K {

            // Save to array to compare with brute force
            data[i][j] = uniform.sample(&mut rng);
        }

        // Add to tree
        tree.add(&data[i], i).expect("Couldn't add to tree");
    }

    // Initialize query points
    let mut query = vec![[0.0; K]; NQUERY];
    for i in 0..NQUERY {
        for j in 0..K {
            // Save to array to compare with brute force
            query[i][j] = uniform.sample(&mut rng);
        }
    }

    // Query points
    let knns: Vec<Vec<(f64, &usize)>> = (&query)
        .into_par_iter()
        .map_with(&tree, |t, q| {
            t.within(&q, RADIUS, &squared_euclidean).unwrap()
        }).collect::<Vec<_>>();

    // Check vs brute force
    for (query_index, q) in query.iter().enumerate() {
        
        // Initialize neighbor
        let mut neighbors = vec![];

        for (data_index, d) in data.iter().enumerate() {

            // Initialize min
            let mut min: f64 = std::f64::MAX;

            // Calculate distance for every image lazily (i.e. 3^K instead of 2^K)
            for image_idx in 0..3_i32.pow(K as u32) {

                // Initialize current_image template
                let mut current_image: [i32; K] = [0; K];

                // Calculate current image
                for idx in 0..K {
                    current_image[idx] = (( image_idx / 3_i32.pow(idx as u32)) % 3) - 1;
                }

                // Construct current image position
                let mut image: [f64; K] = q.clone();
                for idx in 0..K {
                    image[idx] += (current_image[idx] as f64)*PERIODIC[idx];
                }

                // Calculate distance for this image
                let image_distance = squared_euclidean(&image, &d);

                // Compare with current min
                min = min.min(image_distance);
            }
            
            // If this within the target radius, append
            if min < RADIUS {
                neighbors.push((min, data_index));
            }
        }

        // Sort neighbors
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Check that brute force result agrees with KdTree result
        for (k, neighbor) in neighbors.iter().enumerate() {
            assert_eq!(&neighbor.1, knns[query_index][k].1, "{} {}", neighbor.0, knns[query_index][k].0);
            assert!((neighbor.0 - knns[query_index][k].0).abs() < std::f64::EPSILON, "{} {}", neighbor.0, knns[query_index][k].0);
        }
    }
}


#[test]
fn test_periodic_3d_within() {

    // data, query, knns are stored on the heap here because test threads have small stack sizes
    // in a real application, these could be stored in arrays.

    use rand::distributions::{Distribution, Uniform};
    use rayon::prelude::*;

    // Radius to test
    const RADIUS: f64 = 0.05;

    // Initialize dimensionality and data specs
    const K: usize = 3;
    const NQUERY: usize = 10_000;
    const NDATA: usize = 1_000;
    const PERIODIC: [f64; K] = [1.0; K];

    // Initialize rng
    let uniform = Uniform::new(0.0, 1.0);
    let mut rng = rand::thread_rng();

    // Initialize KdTree
    println!("initializing tree");
    let mut tree = KdTree::periodic_with_per_node_capacity(32, &PERIODIC).unwrap();

    // Initialize data points (aka point cloud)
    let mut data = vec![[0.0; K]; NDATA];
    for i in 0..NDATA {
        for j in 0..K {

            // Save to array to compare with brute force
            data[i][j] = uniform.sample(&mut rng);
        }

        // Add to tree
        tree.add(&data[i], i).expect("Couldn't add to tree");
    }

    // Initialize query points
    let mut query = vec![[0.0; K]; NQUERY];
    for i in 0..NQUERY {
        for j in 0..K {
            // Save to array to compare with brute force
            query[i][j] = uniform.sample(&mut rng);
        }
    }

    // Query points
    let knns: Vec<Vec<(f64, &usize)>> = (&query)
        .into_par_iter()
        .map_with(&tree, |t, q| {
            t.within(&q, RADIUS, &squared_euclidean).unwrap()
        }).collect::<Vec<_>>();

    // Check vs brute force
    for (query_index, q) in query.iter().enumerate() {
        
        // Initialize neighbor
        let mut neighbors = vec![];

        for (data_index, d) in data.iter().enumerate() {

            // Initialize min
            let mut min: f64 = std::f64::MAX;

            // Calculate distance for every image lazily (i.e. 3^K instead of 2^K)
            for image_idx in 0..3_i32.pow(K as u32) {

                // Initialize current_image template
                let mut current_image: [i32; K] = [0; K];

                // Calculate current image
                for idx in 0..K {
                    current_image[idx] = (( image_idx / 3_i32.pow(idx as u32)) % 3) - 1;
                }

                // Construct current image position
                let mut image: [f64; K] = q.clone();
                for idx in 0..K {
                    image[idx] += (current_image[idx] as f64)*PERIODIC[idx];
                }

                // Calculate distance for this image
                let image_distance = squared_euclidean(&image, &d);

                // Compare with current min
                min = min.min(image_distance);
            }
            
            // If this within the target radius, append
            if min < RADIUS {
                neighbors.push((min, data_index));
            }
        }

        // Sort neighbors
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Check that brute force result agrees with KdTree result
        for (k, neighbor) in neighbors.iter().enumerate() {
            assert_eq!(&neighbor.1, knns[query_index][k].1, "{} {}", neighbor.0, knns[query_index][k].0);
            assert!((neighbor.0 - knns[query_index][k].0).abs() < std::f64::EPSILON, "{} {}", neighbor.0, knns[query_index][k].0);
        }

    }
}




fn sort_array<A, T>(mut array: A) -> A
where
    A: AsMut<[(T, usize)]>,
    T: PartialOrd,
{
    let slice = array.as_mut();
    slice.sort_by(|a, b| a.partial_cmp(b).unwrap());

    array
}