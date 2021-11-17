#![doc(html_root_url = "https://docs.rs/kiddo/0.2.1")]
#![doc(issue_tracker_base_url = "https://github.com/sdd/kiddo/issues/")]

//! # kiddo
//!
//! K-dimensional tree library (bucket point-region implementation).
//! A fork of kdtree. Refactored to use const generics, with some performance improvements and extra features.
//! Thanks and kudos to mrhooray for the original kdtree library on which kiddo is based.
//!
//! Ideal for neareast-neighbour stype queries on astronomical and geospatial datasets.
//!
//! ## Installation
//!
//! Add `kiddo` to `Cargo.toml`
//! ```toml
//! [dependencies]
//! kiddo = "0.2.1"
//! ```
//!
//! ## Usage
//! ```rust
//! use kiddo::KdTree;
//! use kiddo::ErrorKind;
//! use kiddo::distance::squared_euclidean;
//!
//! let a: ([f64; 2], usize) = ([0f64, 0f64], 0);
//! let b: ([f64; 2], usize) = ([1f64, 1f64], 1);
//! let c: ([f64; 2], usize) = ([2f64, 2f64], 2);
//! let d: ([f64; 2], usize) = ([3f64, 3f64], 3);
//!
//! let mut kdtree = KdTree::new();
//!
//! kdtree.add(&a.0, a.1)?;
//! kdtree.add(&b.0, b.1)?;
//! kdtree.add(&c.0, c.1)?;
//! kdtree.add(&d.0, d.1)?;
//!
//! assert_eq!(kdtree.size(), 4);
//! assert_eq!(
//!     kdtree.nearest(&a.0, 0, &squared_euclidean)?,
//!     vec![]
//! );
//! assert_eq!(
//!     kdtree.nearest(&a.0, 1, &squared_euclidean)?,
//!     vec![(0f64, &0)]
//! );
//! assert_eq!(
//!     kdtree.nearest(&a.0, 2, &squared_euclidean)?,
//!     vec![(0f64, &0), (2f64, &1)]
//! );
//! assert_eq!(
//!     kdtree.nearest(&a.0, 3, &squared_euclidean)?,
//!     vec![(0f64, &0), (2f64, &1), (8f64, &2)]
//! );
//! assert_eq!(
//!     kdtree.nearest(&a.0, 4, &squared_euclidean)?,
//!     vec![(0f64, &0), (2f64, &1), (8f64, &2), (18f64, &3)]
//! );
//! assert_eq!(
//!     kdtree.nearest(&a.0, 5, &squared_euclidean)?,
//!     vec![(0f64, &0), (2f64, &1), (8f64, &2), (18f64, &3)]
//! );
//! assert_eq!(
//!     kdtree.nearest(&b.0, 4, &squared_euclidean)?,
//!     vec![(0f64, &1), (2f64, &0), (2f64, &2), (8f64, &3)]
//! );
//! # Ok::<(), kiddo::ErrorKind>(())
//! ```

#[cfg(feature = "serialize")]
extern crate serde;
#[cfg(feature = "serialize")]
#[cfg_attr(feature = "serialize", macro_use)]
extern crate serde_derive;

mod custom_serde;
pub mod distance;
mod heap_element;
pub mod kiddo;
mod util;

pub use crate::kiddo::ErrorKind;
pub use crate::kiddo::KdTree;
