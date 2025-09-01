//! # RapidGeo Similarity
//!
//! Fast trajectory similarity measures for geographic polylines.
//!
//! This crate provides efficient implementations of curve similarity algorithms
//! specifically designed for geographic data. It includes:
//!
//! - **Fréchet distance** - Measures similarity while considering point order
//! - **Hausdorff distance** - Measures maximum deviation between curves
//! - **Batch processing** - Parallel computation for multiple comparisons
//!
//! All algorithms use great-circle distance (haversine) for geographic accuracy.
//!
//! ## Quick Start
//!
//! ```rust
//! use rapidgeo_similarity::{discrete_frechet_distance, LngLat};
//!
//! let route1 = vec![
//!     LngLat::new_deg(-122.0, 37.0),
//!     LngLat::new_deg(-122.1, 37.1),
//! ];
//! let route2 = vec![
//!     LngLat::new_deg(-122.0, 37.0),
//!     LngLat::new_deg(-122.2, 37.2),
//! ];
//!
//! let distance = discrete_frechet_distance(&route1, &route2).unwrap();
//! println!("Fréchet distance: {} meters", distance);
//! ```
//!
//! ## Choosing an Algorithm
//!
//! - **Fréchet distance**: Use when point order matters (e.g., comparing GPS tracks)
//! - **Hausdorff distance**: Use when only shape matters (e.g., comparing building outlines)
//!
//! ## Features
//!
//! - `batch` - Enables parallel batch processing functions (requires rayon)

pub use rapidgeo_distance::LngLat;

pub mod frechet;
pub mod hausdorff;

#[cfg(feature = "batch")]
pub mod batch;

// Re-export main functions and types for convenience
pub use frechet::{
    discrete_frechet_distance, discrete_frechet_distance_with_threshold, DiscreteFrechet,
};
pub use hausdorff::{hausdorff_distance, hausdorff_distance_with_threshold, Hausdorff};

#[cfg(feature = "batch")]
pub use batch::{
    batch_frechet_distance, batch_frechet_distance_threshold, pairwise_frechet_matrix,
};

/// A trait for measuring similarity between two polylines.
///
/// This trait provides a common interface for different similarity algorithms
/// like Fréchet distance and Hausdorff distance. All implementations use
/// great-circle distance for geographic coordinates.
pub trait SimilarityMeasure {
    /// Calculate the distance between two polylines.
    ///
    /// # Arguments
    ///
    /// * `a` - First polyline as a slice of longitude/latitude points
    /// * `b` - Second polyline as a slice of longitude/latitude points
    ///
    /// # Returns
    ///
    /// The similarity distance in meters, or an error if the input is invalid.
    ///
    /// # Errors
    ///
    /// Returns `SimilarityError::EmptyInput` if either polyline is empty.
    fn distance(&self, a: &[LngLat], b: &[LngLat]) -> Result<f64, SimilarityError>;
}

/// Errors that can occur when computing similarity measures.
#[derive(Debug, Clone, PartialEq)]
pub enum SimilarityError {
    /// One or both input polylines are empty.
    EmptyInput,
    /// Input contains invalid data.
    InvalidInput(String),
}

impl std::fmt::Display for SimilarityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimilarityError::EmptyInput => write!(f, "input polylines cannot be empty"),
            SimilarityError::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
        }
    }
}

impl std::error::Error for SimilarityError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_error_display() {
        let empty_error = SimilarityError::EmptyInput;
        assert_eq!(empty_error.to_string(), "input polylines cannot be empty");

        let invalid_error = SimilarityError::InvalidInput("test message".to_string());
        assert_eq!(invalid_error.to_string(), "invalid input: test message");
    }
}
