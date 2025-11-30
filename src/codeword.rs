use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Represents a single codeword in the codebook model.
///
/// # Attributes
/// * `rgb_min` - Minimum RGB values observed
/// * `rgb_max` - Maximum RGB values observed
/// * `i_min` - Minimum brightness (I) value
/// * `i_max` - Maximum brightness (I) value
/// * `frequency` - Number of times this codeword has been matched
/// * `lambda` - Maximum negative run-length (MNRL)
/// * `first_access` - Time when codeword was first created
/// * `last_access` - Time when codeword was last updated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codeword {
    pub rgb_min: Array1<f32>,
    pub rgb_max: Array1<f32>,
    pub i_min: f32,
    pub i_max: f32,
    pub frequency: u32,
    pub lambda: u32,
    pub first_access: u32,
    pub last_access: u32,
}

impl Codeword {
    /// Create a new codeword from a pixel observation
    pub fn new(pixel: &Array1<f32>, current_time: u32) -> Self {
        let pixel_i = pixel.sum();

        Codeword {
            rgb_min: pixel.clone(),
            rgb_max: pixel.clone(),
            i_min: pixel_i,
            i_max: pixel_i,
            frequency: 1,
            lambda: 0,
            first_access: current_time,
            last_access: current_time,
        }
    }
    /// Update codeword statistics with new pixel observation
    pub fn update(&mut self, pixel: &Array1<f32>, current_time: u32) {
        let pixel_i = pixel.sum();

        // Update RGB bounds
        for i in 0..3 {
            self.rgb_min[i] = self.rgb_min[i].min(pixel[i]);
            self.rgb_max[i] = self.rgb_max[i].max(pixel[i]);
        }

        // Update brightness bounds
        self.i_min = self.i_min.min(pixel_i);
        self.i_max = self.i_max.max(pixel_i);

        // Update temporal information
        self.frequency += 1;
        let time_since_access = current_time - self.last_access;
        self.lambda = self.lambda.max(time_since_access);
        self.last_access = current_time;
    }

    /// Get the average RGB values of this codeword
    pub fn avg_rgb(&self) -> Array1<f32> {
        (&self.rgb_min + &self.rgb_max) / 2.0
    }
}
