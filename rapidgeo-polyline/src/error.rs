//! Error types for polyline operations.

use std::fmt;

/// Errors that can occur during polyline encoding or decoding operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PolylineError {
    /// Invalid characters found in the polyline string
    InvalidCharacter { character: char, position: usize },
    /// Truncated or malformed polyline data
    TruncatedData,
    /// Coordinate overflow during encoding or decoding
    CoordinateOverflow,
    /// Invalid precision value (must be between 1 and 11)
    InvalidPrecision(u8),
    /// Empty input where coordinates were expected
    EmptyInput,
    /// Invalid coordinate value (NaN, infinity, or out of bounds)
    InvalidCoordinate(String),
}

impl fmt::Display for PolylineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolylineError::InvalidCharacter {
                character,
                position,
            } => {
                write!(
                    f,
                    "Invalid character '{}' at position {}",
                    character, position
                )
            }
            PolylineError::TruncatedData => {
                write!(f, "Polyline data is truncated or malformed")
            }
            PolylineError::CoordinateOverflow => {
                write!(f, "Coordinate value overflow during encoding or decoding")
            }
            PolylineError::InvalidPrecision(precision) => {
                write!(
                    f,
                    "Invalid precision {}, must be between 1 and 11",
                    precision
                )
            }
            PolylineError::EmptyInput => {
                write!(f, "Empty input provided")
            }
            PolylineError::InvalidCoordinate(message) => {
                write!(f, "Invalid coordinate: {}", message)
            }
        }
    }
}

impl std::error::Error for PolylineError {}

/// Result type for polyline operations.
pub type PolylineResult<T> = Result<T, PolylineError>;
