//! Defines different distance metrics, in simplest case it defines the
//! euclidean distance which is no more than the square root of the sum of the
//! squares of the distances in each dimension.

use num_traits::Float;

use std::arch::x86_64::*;

/// Returns the squared euclidean distance between two points. When you only
/// need to compare distances, rather than having the exact distance between
/// the points, this metric is benefitial because it avoids the expensive square
/// root computation.
///
/// # Examples
///
/// ```rust
/// use kdtree::distance::squared_euclidean;
///
/// assert!(0.0 == squared_euclidean(&[0.0, 0.0], &[0.0, 0.0]));
/// assert!(2.0 == squared_euclidean(&[0.0, 0.0], &[1.0, 1.0]));
/// assert!(1.0 == squared_euclidean(&[0.0, 0.0], &[1.0, 0.0]));
/// ```
///
/// # Panics
///
/// Only in debug mode, the length of the slices at input will be compared.
/// If they do not match, there will be a panic:
///
/// ```rust,should_panic
/// # use kdtree::distance::squared_euclidean;
/// // this is broken
/// let _ = squared_euclidean(&[0.0, 0.0], &[1.0, 0.0, 0.0]);
/// ```
pub fn squared_euclidean<T: Float>(a: &[T], b: &[T]) -> T {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) - (*y)) * ((*x) - (*y)))
        .fold(T::zero(), ::std::ops::Add::add)
}


union SimdToArray {
    array: [f32; 4],
    simd: __m128
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) * (*y)))
        .fold(0f32, ::std::ops::Sub::sub)
}

#[cfg(any(target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
pub unsafe fn dot_sse(a: *const f32, b: *const f32) -> f32 {
    let a_mm = _mm_loadu_ps(a);
    let b_mm = _mm_loadu_ps(b);

    let res: SimdToArray = SimdToArray { simd: _mm_dp_ps(a_mm, b_mm, 0x71) };
    res.array[0]
}

#[cfg(any(target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
pub unsafe fn dot_sse_aligned(a: *const f32, b: *const f32) -> f32 {
    let a_mm = _mm_load_ps(a);
    let b_mm = _mm_load_ps(b);

    let res: SimdToArray = SimdToArray { simd: _mm_dp_ps(a_mm, b_mm, 0x71) };
    res.array[0]
}

pub fn dot_product_sse(a: &[f32], b: &[f32]) -> f32 {
    //debug_assert_eq!(a.len(), b.len());

    if a.len() == 3 {
        let ap = [a[0], a[1], a[2], 0.0f32].as_ptr();
        let bp = [b[0], b[1], b[2], 0f32].as_ptr();
        unsafe {
            dot_sse(ap, bp)
        }
    } else if a.len() == 4 {
         unsafe {
             dot_sse(a.as_ptr(), b.as_ptr())
         }
    } else {
         dot_product(a, b)
    }
}

pub fn dot_product_sse_aligned(a: &[f32], b: &[f32]) -> f32 {
    //debug_assert_eq!(a.len(), b.len());

    let ap = a.as_ptr();
    let bp = b.as_ptr();
    unsafe {
        dot_sse_aligned(ap, bp)
    }
}
