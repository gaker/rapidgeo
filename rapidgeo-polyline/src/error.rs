//! Error types for polyline operations.

use std::fmt;

/// Errors that can occur during polyline encoding or decoding operations.
///
/// These errors provide detailed information about what went wrong during
/// polyline processing, including specific locations for parsing errors
/// and descriptive messages for coordinate validation failures.
#[derive(Debug, Clone, PartialEq)]
pub enum PolylineError {
    /// Invalid character found in polyline string.
    ///
    /// Polyline strings must contain only ASCII characters in the range 63-126 ('?' to '~').
    /// This error includes the invalid character and its position in the string.
    InvalidCharacter { character: char, position: usize },

    /// Truncated or malformed polyline data.
    ///
    /// Occurs when the polyline string ends unexpectedly, such as having a latitude
    /// delta without a corresponding longitude delta, or when the variable-length
    /// encoding is incomplete.
    TruncatedData,

    /// Coordinate value overflow during encoding or decoding.
    ///
    /// Happens when coordinate calculations exceed the range of 64-bit integers,
    /// typically with very high precision values or extreme coordinate differences.
    CoordinateOverflow,

    /// Invalid precision value.
    ///
    /// Precision must be between 1 and 11 inclusive. Higher precision values
    /// provide more accuracy but may cause overflow with large coordinate values.
    InvalidPrecision(u8),

    /// Empty input where coordinates were expected.
    ///
    /// Currently unused - empty inputs are handled gracefully by returning
    /// empty results rather than errors.
    EmptyInput,

    /// Invalid coordinate value.
    ///
    /// Occurs when coordinates are NaN, infinite, or outside reasonable
    /// geographic bounds (±180° longitude, ±90° latitude).
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
///
/// This is a convenience type alias for `Result<T, PolylineError>` used
/// throughout the crate. All public functions return this type to provide
/// consistent error handling.
///
/// # Example
///
/// ```rust
/// use rapidgeo_polyline::{encode, PolylineResult};
/// use rapidgeo_distance::LngLat;
///
/// fn encode_route(coords: &[LngLat]) -> PolylineResult<String> {
///     encode(coords, 5)
/// }
/// ```
pub type PolylineResult<T> = Result<T, PolylineError>;
