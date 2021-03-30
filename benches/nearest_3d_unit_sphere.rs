#[macro_use]
extern crate lazy_static;
extern crate criterion;
extern crate kdtree;
extern crate rand;
extern crate num_traits;
extern crate aligned;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use rand::distributions::{UnitSphereSurface, Distribution};
use kdtree::distance::{squared_euclidean};
use kdtree::KdTree;
use num_traits::{FromPrimitive};
use aligned::{Aligned, A16};

use std::arch::x86_64::*;

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


pub fn nearest_1_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest(1)");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        //group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {

            let point = rand_sphere_data();

            let mut points = vec![];
            let mut kdtree = KdTree::with_capacity(3, 16);
            for _ in 0..size {
                points.push(rand_sphere_data());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1).unwrap();
            }

            b.iter(|| {
                black_box(kdtree.nearest(&point.0, 1, &squared_euclidean)).unwrap()
            });
        });
    }
}

pub fn nearest_100_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest(100)");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        //group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {

            let point = rand_sphere_data();

            let mut points = vec![];
            let mut kdtree = KdTree::with_capacity(3, 16);
            for _ in 0..size {
                points.push(rand_sphere_data());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1).unwrap();
            }

            b.iter(|| {
                black_box(kdtree.nearest(&point.0, 100, &squared_euclidean)).unwrap()
            });
        });
    }
}


pub fn nearest_1000_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest(1000)");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        //group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {

            let point = rand_sphere_data();

            let mut points = vec![];
            let mut kdtree = KdTree::with_capacity(3, 16);
            for _ in 0..size {
                points.push(rand_sphere_data());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1).unwrap();
            }

            b.iter(|| {
                black_box(kdtree.nearest(&point.0, 1000, &squared_euclidean)).unwrap()
            });
        });
    }
}

criterion_group!(benches, nearest_1_euclidean2, nearest_100_euclidean2, nearest_1000_euclidean2);
criterion_main!(benches);