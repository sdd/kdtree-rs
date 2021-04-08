#![feature(test)]
#[macro_use]
extern crate lazy_static;

extern crate kiddo;
extern crate rand;
extern crate test;
extern crate num_traits;
extern crate aligned;

use rand::distributions::{UnitSphereSurface, Distribution};
use kiddo::distance::{squared_euclidean, dot_product, dot_product_sse, dot_product_sse_aligned};
use kiddo::KdTree;
use test::Bencher;

use num_traits::{FromPrimitive};

use std::arch::x86_64::*;
use aligned::{Aligned, A16};
use std::hint::black_box;

union SimdToArray {
    array: [f32; 4],
    simd: __m128
}

lazy_static! {
    static ref SPHERE: UnitSphereSurface = UnitSphereSurface::new();
}

//fn rand_data() -> ([f64; 3], f64) {
    //rand::random()
//}

fn rand_unit_sphere_point_f64() -> [f64; 3] {
    SPHERE.sample(&mut rand::thread_rng())
}

fn rand_unit_sphere_point_f32() -> [f32; 3] {
    let sph64: [f64; 3] = SPHERE.sample(&mut rand::thread_rng());
    let res: Aligned<A16, _> = Aligned([
        f32::from_f64(sph64[0]).unwrap(),
        f32::from_f64(sph64[1]).unwrap(),
        f32::from_f64(sph64[2]).unwrap(),
    ]);
    *res
}

fn rand_unit_sphere_point_f32_qwalign() -> [f32; 4] {
    let sph64: [f64; 3] = SPHERE.sample(&mut rand::thread_rng());
    let res: Aligned<A16, _> = Aligned([
        f32::from_f64(sph64[0]).unwrap(),
        f32::from_f64(sph64[1]).unwrap(),
        f32::from_f64(sph64[2]).unwrap(),
        0f32
    ]);
    *res
}

fn rand_data() -> ([f64; 3], f64) {
    rand::random()
}

fn rand_sphere_data() -> ([f64; 3], usize) {
    (rand_unit_sphere_point_f64(), rand::random())
}

fn rand_sphere_data_f32() -> ([f32; 3], usize) {
    (rand_unit_sphere_point_f32(), rand::random())
}

fn rand_sphere_data_f32_qw() -> ([f32; 4], usize) {
    (rand_unit_sphere_point_f32_qwalign(), rand::random())
}

#[bench]
fn bench_add_to_kdtree_with_1k_3d_points(b: &mut Bencher) {
    let len = 1000usize;
    let point = rand_data();
    let mut points = vec![];
    let mut kdtree = KdTree::with_capacity(16).unwrap();
    for _ in 0..len {
        points.push(rand_data());
    }
    for i in 0..points.len() {
        kdtree.add(&points[i].0, points[i].1).unwrap();
    }
    b.iter(|| kdtree.add(&point.0, point.1).unwrap());
}

#[bench]
fn bench_nearest_from_kdtree_with_1k_3d_points(b: &mut Bencher) {
    let len = 100000usize;
    let point = rand_data();
    let mut points = vec![];
    let mut kdtree = KdTree::with_capacity(16).unwrap();
    for _ in 0..len {
        points.push(rand_data());
    }
    for i in 0..points.len() {
        kdtree.add(&points[i].0, points[i].1).unwrap();
    }
    b.iter(|| kdtree.nearest(&point.0, 1000, &squared_euclidean).unwrap());
}

#[bench]
fn bench_nearest_from_kdtree_with_150k_3d_points_squared_euclidean(b: &mut Bencher) {
    let len = 150000usize;
    let point = rand_sphere_data();
    let mut points = vec![];
    let mut kdtree = KdTree::with_capacity(16).unwrap();

    for _ in 0..len {
        points.push(rand_sphere_data());
    }

    for i in 0..points.len() {
        kdtree.add(&points[i].0, points[i].1).unwrap();
    }

    b.iter(|| kdtree.nearest(&point.0, 50000, &squared_euclidean).unwrap());
}

#[bench]
fn bench_nearest_with_1_from_kdtree_with_150k_3d_points_squared_euclidean(b: &mut Bencher) {
    let len = 150000usize;
    let point = rand_sphere_data();
    let mut points = vec![];
    let mut kdtree = KdTree::with_capacity(16).unwrap();

    for _ in 0..len {
        points.push(rand_sphere_data());
    }

    for i in 0..points.len() {
        kdtree.add(&points[i].0, points[i].1).unwrap();
    }

    b.iter(|| kdtree.nearest(&point.0, 1, &squared_euclidean).unwrap());
}

#[bench]
fn bench_nearest_one_from_kdtree_with_150k_3d_points_squared_euclidean(b: &mut Bencher) {
    let len = 150000usize;
    let point = rand_sphere_data();
    let mut points = vec![];
    let mut kdtree = KdTree::with_capacity(16).unwrap();

    for _ in 0..len {
        points.push(rand_sphere_data());
    }

    for i in 0..points.len() {
        kdtree.add(&points[i].0, points[i].1).unwrap();
    }

    b.iter(|| kdtree.nearest_one(&point.0, &squared_euclidean).unwrap());
}

#[bench]
fn bench_best_n_within_150k_3d_squared_euclidean(b: &mut Bencher) {
    let len = 150000usize;
    let point = rand_sphere_data();
    let mut points = vec![];
    let mut kdtree = KdTree::with_capacity(16).unwrap();

    for _ in 0..len {
        points.push(rand_sphere_data());
    }

    for i in 0..points.len() {
        kdtree.add(&points[i].0, points[i].1).unwrap();
    }

    b.iter(|| kdtree.best_n_within(&point.0, 0.1, 500, &squared_euclidean).unwrap());
}

#[bench]
fn bench_nearest_from_kdtree_with_150k_3d_points_dot_product(b: &mut Bencher) {
    let len = 150000usize;
    let point = rand_sphere_data_f32();
    let mut points = vec![];
    let mut kdtree = KdTree::with_capacity(16).unwrap();

    for _ in 0..len {
        points.push(rand_sphere_data_f32());
    }

    for i in 0..points.len() {
        kdtree.add(&points[i].0, points[i].1).unwrap();
    }

    b.iter(|| kdtree.nearest(&point.0, 50000, &dot_product).unwrap());
}

#[bench]
fn bench_nearest_from_kdtree_with_150k_3d_points_dot_product_sse(b: &mut Bencher) {
    println!("starting");
    let len = 150000usize;
    let point = rand_sphere_data_f32();
    let mut points = vec![];
    let mut kdtree = KdTree::with_capacity(16).unwrap();

    for _ in 0..len {
        points.push(rand_sphere_data_f32());
    }

    for i in 0..points.len() {
        kdtree.add(&points[i].0, points[i].1).unwrap();
    }

    println!("calling nearest");
    b.iter(|| kdtree.nearest(&point.0, 50000, &dot_product_sse).unwrap());
}

#[bench]
fn bench_nearest_from_kdtree_with_150k_3d_points_dot_product_sse_aligned(b: &mut Bencher) {
    println!("starting");
    let len = 150000usize;
    let point = rand_sphere_data_f32_qw();
    let mut points = vec![];
    let mut kdtree = KdTree::with_capacity(16).unwrap();

    for _ in 0..len {
        points.push(rand_sphere_data_f32_qw());
    }

    for i in 0..points.len() {
        kdtree.add(&points[i].0, points[i].1).unwrap();
    }

    println!("calling nearest");
    b.iter(|| kdtree.nearest(&point.0, 50000, &dot_product_sse).unwrap());
}
