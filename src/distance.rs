//! Defines different distance metrics, in simplest case it defines the
//! euclidean distance which is no more than the square root of the sum of the
//! squares of the distances in each dimension.

use num_traits::Float;
use itertools::{iproduct, Itertools, MinMaxResult};

#[cfg(any(target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86_64"))]
union SimdToArray {
    array: [f32; 4],
    simd: __m128,
}


/// Returns the squared euclidean distance between two points. When you only
/// need to compare distances, rather than having the exact distance between
/// the points, this metric is beneficial because it avoids the expensive square
/// root computation.
///
/// # Examples
///
/// ```rust
/// use kiddo::distance::squared_euclidean;
///
/// assert!(0.0 == squared_euclidean(&[0.0, 0.0], &[0.0, 0.0]));
/// assert!(2.0 == squared_euclidean(&[0.0, 0.0], &[1.0, 1.0]));
/// assert!(1.0 == squared_euclidean(&[0.0, 0.0], &[1.0, 0.0]));
/// ```
pub fn squared_euclidean<T: Float, const K: usize>(a: &[T; K], b: &[T; K]) -> T {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) - (*y)) * ((*x) - (*y)))
        .fold(T::zero(), ::std::ops::Add::add)
}

pub fn periodic_distance<T: Float, const K: usize>(
    a: &[T; K],
    b: &[T; K],
    boxsize: Option<[T; K]>,
    distance: &dyn Fn(&[T; K], &[T; K]) -> T,
) -> T {

    
    // Gather 3^K distances and choose minimum
    let mut ds : Vec<T> = Vec::with_capacity(3_usize.pow(K as u32));

    // 3^K copies (incl original)
    use std::convert::TryInto;
    let copies : Vec<Vec<T>> = n_product::<T, K>().expect("error making copies");
    for copy in copies {
        let new_p1 = a.to_vec().iter().enumerate().map(|(i, &x)| x + copy[i]*boxsize.unwrap()[i]).collect::<Vec<T>>();
        ds.push(distance(&new_p1.try_into().unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of length {} but it was {}", K, v.len())), &b))
    }

    debug_assert_eq!(ds.len(), 3_usize.pow(K as u32));
   // return min, discard max
    *match ds.iter().minmax(){
        MinMaxResult::MinMax(min, _max) => Ok(min),
        _ => Err("No Min"), 
    }.unwrap()

}

// Used for periodic distance calculation
fn n_product<T: Float, const K: usize> () -> Result<Vec<Vec<T>>, String> {

    let m : [T; 3] = [-T::one(), T::zero(), T::one()];

    match K {
        1 => Ok(iproduct!(m).map(|a| vec![a]).collect::<Vec<Vec<T>>>()),
        2 => Ok(iproduct!(m, m).map(|(a, b)| vec![a, b]).collect::<Vec<Vec<T>>>()),
        3 => Ok(iproduct!(m, m, m).map(|(a, b, c)| vec![a, b, c]).collect::<Vec<Vec<T>>>()),
        4 => Ok(iproduct!(m, m, m, m).map(|(a, b, c, d)| vec![a, b, c, d]).collect::<Vec<Vec<T>>>()),
        5 => Ok(iproduct!(m, m, m, m, m).map(|(a, b, c, d, e)| vec![a, b, c, d, e]).collect::<Vec<Vec<T>>>()),
        6 => Ok(iproduct!(m, m, m, m, m, m).map(|(a, b, c, d, e, f)| vec![a, b, c, d, e, f]).collect::<Vec<Vec<T>>>()),
        _ => Err("Dimensions not in 1..=6 not implemented".to_string())
    }
}

pub fn dot_product<const K: usize>(a: &[f32; K], b: &[f32; K]) -> f32 {
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

    let res: SimdToArray = SimdToArray {
        simd: _mm_dp_ps(a_mm, b_mm, 0x71),
    };
    res.array[0]
}

#[cfg(any(target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
pub unsafe fn dot_sse_aligned(a: *const f32, b: *const f32) -> f32 {
    let a_mm = _mm_load_ps(a);
    let b_mm = _mm_load_ps(b);

    let res: SimdToArray = SimdToArray {
        simd: _mm_dp_ps(a_mm, b_mm, 0x71),
    };
    res.array[0]
}

#[cfg(any(target_arch = "x86_64"))]
pub fn dot_product_sse<const K: usize>(a: &[f32; K], b: &[f32; K]) -> f32 {
    if K == 3 {
        dot_product_sse_3(&a[0..3], &a[0..3])
    } else if K == 4 {
        dot_product_sse_4(&a[0..4], &a[0..4])
    } else {
        dot_product(a, b)
    }
}

#[cfg(any(target_arch = "x86_64"))]
pub fn dot_product_sse_3(a: &[f32], b: &[f32]) -> f32 {
    let ap = [a[0], a[1], a[2], 0f32].as_ptr();
    let bp = [b[0], b[1], b[2], 0f32].as_ptr();
    unsafe { dot_sse(ap, bp) }
}

#[cfg(any(target_arch = "x86_64"))]
pub fn dot_product_sse_4(a: &[f32], b: &[f32]) -> f32 {
    unsafe { dot_sse(a.as_ptr(), b.as_ptr()) }
}

#[cfg(any(target_arch = "x86_64"))]
pub fn dot_product_sse_aligned(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    unsafe { dot_sse_aligned(ap, bp) }
}
