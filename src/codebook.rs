use crate::codeword::Codeword;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

// alpha: Lower bound factor for brightness matching
// beta: Upper bound factor for brightness matching
// lambda_param: Maximum time a codeword can go without being accessed
// epsilon: Color distortion threshold

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebookModel {
    pub alpha: f32,
    pub beta: f32,
    pub lambda: f32,
    pub epsilon: f32,
    pub width: usize,
    pub height: usize,
    pub codebooks: Vec<Vec<Codeword>>,
    pub current_time: u32,
}
impl CodebookModel {
    pub fn new(
        alpha: f32,
        beta: f32,
        lambda: f32,
        epsilon: f32,
        width: usize,
        height: usize,
    ) -> Self {
        let codewords = vec![Vec::new(); width * height];
        CodebookModel {
            alpha,
            beta,
            lambda,
            epsilon,
            width,
            height,
            codebooks: codewords,
            current_time: 0,
        }
    }

    // Create codebook from learning phase
    // frame: Input frame (H x W x 3) in BGR format
    pub fn learning_phase(&mut self, frame: &Vec<Array1<f32>>) {
        self.current_time += 1;

        let frame_rgb = Self::bgr_to_rgb(frame);
        for (idx, pixel) in frame_rgb.iter().enumerate() {
            let codebook = &mut self.codebooks[idx];

            let match_result = {
                let mut best_match = None;
                let mut min_distortion = f32::INFINITY;

                for (i, codeword) in codebook.iter().enumerate() {
                    // Brightness check
                    let pixel_i = pixel.sum();
                    if !(codeword.i_min as f32 * self.alpha <= pixel_i
                        && pixel_i <= codeword.i_max as f32 * self.beta)
                    {
                        continue;
                    }

                    // Color distortion check
                    let mut dist = 0.0;
                    for j in 0..3 {
                        let pw = pixel[j];
                        let cw_min = codeword.rgb_min[j];
                        let cw_max = codeword.rgb_max[j];
                        if pw < cw_min {
                            dist += (cw_min - pw).powi(2);
                        } else if pw > cw_max {
                            dist += (pw - cw_max).powi(2);
                        }
                    }
                    let distortion = dist.sqrt();

                    if distortion <= self.epsilon && distortion <= min_distortion {
                        min_distortion = distortion;
                        best_match = Some(i);
                    }
                }
                best_match
            };

            if let Some(cw_index) = match_result {
                // Update existing codeword
                let codeword = &mut codebook[cw_index];
                codeword.update(pixel, self.current_time);
            } else {
                // Create new codeword
                let new_codeword = Codeword::new(pixel, self.current_time);
                codebook.push(new_codeword);
            }
        }
    }
    // Detection Phase
    pub fn foreground_detect(&mut self, frame: &Vec<Array1<f32>>) -> Vec<bool> {
        let mut foreground_mask = vec![false; self.width * self.height];
        self.current_time += 1;

        let frame_rgb = Self::bgr_to_rgb(frame);
        for (idx, pixel) in frame_rgb.iter().enumerate() {
            let codebook = &mut self.codebooks[idx];
            let mut is_background = false;

            for codeword in codebook.iter_mut() {
                let pixel_i = pixel.sum();
                // Brightness check
                if !(codeword.i_min as f32 * self.alpha <= pixel_i
                    && pixel_i <= codeword.i_max as f32 * self.beta)
                {
                    continue;
                }

                // Color distortion check
                let mut dist = 0.0;
                for j in 0..3 {
                    let pw = pixel[j];
                    let cw_min = codeword.rgb_min[j];
                    let cw_max = codeword.rgb_max[j];
                    if pw < cw_min {
                        dist += (cw_min - pw).powi(2);
                    } else if pw > cw_max {
                        dist += (pw - cw_max).powi(2);
                    }
                }
                let distortion = dist.sqrt();

                if distortion <= self.epsilon {
                    is_background = true;
                    codeword.last_access = self.current_time;
                    break;
                }
            }

            foreground_mask[idx] = !is_background;
        }

        // Remove stale codewords
        for codebook in self.codebooks.iter_mut() {
            codebook.retain(|cw| (self.current_time - cw.last_access) as f32 <= self.lambda);
        }

        foreground_mask
    }

    // Cleanup stale codewords based on lambda parameter
    pub fn cleanup_codewords(&mut self) {
        for codebook in self.codebooks.iter_mut() {
            codebook.retain(|codeword| {
                let time_since_access = self.current_time - codeword.last_access;
                time_since_access <= self.lambda as u32
            });
        }
    }

    fn bgr_to_rgb(bgr_pixels: &[Array1<f32>]) -> Vec<Array1<f32>> {
        bgr_pixels
            .iter()
            .map(|pixel| {
                Array1::from_vec(vec![pixel[2], pixel[1], pixel[0]]) // Swap B and R channels
            })
            .collect()
    }

    /// Save the codebook model to a binary file
    ///
    /// # Arguments
    /// * `path` - Path to save the codebook to
    ///
    /// # Returns
    /// * `Result<(), Box<dyn std::error::Error>>` - Success or error
    ///
    /// # Example
    /// ```no_run
    /// # use codebook_model::codebook::CodebookModel;
    /// let model = CodebookModel::new(0.4, 1.1, 100.0, 10.0, 640, 480);
    /// model.save_to_file("my_codebook.cbm").unwrap();
    /// ```
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Save the codebook model to a JSON file (human-readable but larger)
    ///
    /// # Arguments
    /// * `path` - Path to save the codebook to
    ///
    /// # Returns
    /// * `Result<(), Box<dyn std::error::Error>>` - Success or error
    pub fn save_to_json<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Load a codebook model from a binary file
    ///
    /// # Arguments
    /// * `path` - Path to load the codebook from
    ///
    /// # Returns
    /// * `Result<CodebookModel, Box<dyn std::error::Error>>` - Loaded model or error
    ///
    /// # Example
    /// ```no_run
    /// # use codebook_model::codebook::CodebookModel;
    /// let model = CodebookModel::load_from_file("my_codebook.cbm").unwrap();
    /// ```
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model = bincode::deserialize_from(reader)?;
        Ok(model)
    }

    /// Load a codebook model from a JSON file
    ///
    /// # Arguments
    /// * `path` - Path to load the codebook from
    ///
    /// # Returns
    /// * `Result<CodebookModel, Box<dyn std::error::Error>>` - Loaded model or error
    pub fn load_from_json<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model = serde_json::from_reader(reader)?;
        Ok(model)
    }

    /// Clone the codebook structure but with new parameters
    /// This allows testing different detection parameters without re-learning
    ///
    /// # Arguments
    /// * `alpha` - New alpha parameter
    /// * `beta` - New beta parameter
    /// * `epsilon` - New epsilon parameter
    /// * `lambda` - New lambda parameter
    ///
    /// # Returns
    /// * `CodebookModel` - New model with same learned data but different parameters
    pub fn with_params(&self, alpha: f32, beta: f32, epsilon: f32, lambda: f32) -> Self {
        CodebookModel {
            alpha,
            beta,
            epsilon,
            lambda,
            width: self.width,
            height: self.height,
            codebooks: self.codebooks.clone(),
            current_time: self.current_time,
        }
    }
}

impl fmt::Display for CodebookModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CodebookModel(alpha: {}, beta: {}, lambda: {}, epsilon: {}, width: {}, height: {}, current_time: {})",
            self.alpha,
            self.beta,
            self.lambda,
            self.epsilon,
            self.width,
            self.height,
            self.current_time
        )
    }
}

