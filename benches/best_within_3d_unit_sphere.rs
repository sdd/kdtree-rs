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
use num_traits::{FromPrimitive, Float};
use aligned::{Aligned, A16};

use std::arch::x86_64::*;
use std::collections::BinaryHeap;

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

pub fn best_1_within_small_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("best 1: within(0.01)");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        //group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {

            let point = rand_sphere_data();

            let mut points = vec![];
            let mut kdtree = KdTree::with_capacity(16).unwrap();
            for _ in 0..size {
                points.push(rand_sphere_data());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1).unwrap();
            }

            b.iter(|| {
                black_box(kdtree.within(&point.0, 0.01, &squared_euclidean).unwrap().iter().map(|(_, x)|*x).min())
            });
        });
    }
}

pub fn best_1_within_medium_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("best 1: within(0.05)");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        //group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {

            let point = rand_sphere_data();

            let mut points = vec![];
            let mut kdtree = KdTree::with_capacity(16).unwrap();
            for _ in 0..size {
                points.push(rand_sphere_data());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1).unwrap();
            }

            b.iter(|| {
                black_box(kdtree.within(&point.0, 0.05, &squared_euclidean).unwrap().iter().map(|(_, x)|*x).min())
            });
        });
    }
}

pub fn best_1_within_large_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("best 1: within(0.25)");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        //group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {

            let point = rand_sphere_data();

            let mut points = vec![];
            let mut kdtree = KdTree::with_capacity(16).unwrap();
            for _ in 0..size {
                points.push(rand_sphere_data());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1).unwrap();
            }

            b.iter(|| {
                black_box(kdtree.within(&point.0, 0.25, &squared_euclidean).unwrap().iter().map(|(_, x)|*x).min())
            });
        });
    }
}

pub fn best_100_within_small_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("best 100: within(0.01)");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        //group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {

            let point = rand_sphere_data();

            let mut points = vec![];
            let mut kdtree = KdTree::with_capacity(16).unwrap();
            for _ in 0..size {
                points.push(rand_sphere_data());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1).unwrap();
            }

            b.iter(|| {
                let max_dist = usize::MAX;
                let within = kdtree.within(&point.0, 0.01, &squared_euclidean).unwrap();
                let mut evaluated: BinaryHeap<usize> = BinaryHeap::new();

                for (_, &element) in within.iter() {
                    if element <= max_dist {
                        if evaluated.len() < 100 {
                            evaluated.push(element);
                        } else {
                            let mut top = evaluated.peek_mut().unwrap();
                            if element < *top {
                                *top = element;
                            }
                        }
                    }
                }

                let res = black_box(evaluated);
            });
        });
    }
}

pub fn best_100_within_medium_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("best 100: within(0.05)");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        //group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {

            let point = rand_sphere_data();

            let mut points = vec![];
            let mut kdtree = KdTree::with_capacity(16).unwrap();
            for _ in 0..size {
                points.push(rand_sphere_data());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1).unwrap();
            }

            b.iter(|| {
                let max_dist = usize::MAX;
                let within = kdtree.within(&point.0, 0.05, &squared_euclidean).unwrap();
                let mut evaluated: BinaryHeap<usize> = BinaryHeap::new();

                for (_, &element) in within.iter() {
                    if element <= max_dist {
                        if evaluated.len() < 100 {
                            evaluated.push(element);
                        } else {
                            let mut top = evaluated.peek_mut().unwrap();
                            if element < *top {
                                *top = element;
                            }
                        }
                    }
                }

                let res = black_box(evaluated);
            });
        });
    }
}

pub fn best_100_within_large_euclidean2(c: &mut Criterion) {
    let mut group = c.benchmark_group("best 100: within(0.25)");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        //group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {

            let point = rand_sphere_data();

            let mut points = vec![];
            let mut kdtree = KdTree::with_capacity(16).unwrap();
            for _ in 0..size {
                points.push(rand_sphere_data());
            }
            for i in 0..points.len() {
                kdtree.add(&points[i].0, points[i].1).unwrap();
            }

            b.iter(|| {
                let max_dist = usize::MAX;
                let within = kdtree.within(&point.0, 0.25, &squared_euclidean).unwrap();
                let mut evaluated: BinaryHeap<usize> = BinaryHeap::new();

                for (_, &element) in within.iter() {
                    if element <= max_dist {
                        if evaluated.len() < 100 {
                            evaluated.push(element);
                        } else {
                            let mut top = evaluated.peek_mut().unwrap();
                            if element < *top {
                                *top = element;
                            }
                        }
                    }
                }

                let res = black_box(evaluated);
            });
        });
    }
}


criterion_group!(benches, best_1_within_small_euclidean2, best_1_within_medium_euclidean2, best_1_within_large_euclidean2, best_100_within_small_euclidean2, best_100_within_medium_euclidean2, best_100_within_large_euclidean2);
criterion_main!(benches);