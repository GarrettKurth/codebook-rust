use crate::codebook::CodebookModel;
use ndarray::Array1;
use opencv::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoInfo {
    pub path: String,
    pub learning_frames: u32,
    pub total_processing_time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    pub lambda: u32,
    pub alpha: f32,
    pub beta: f32,
    pub epsilon: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub accuracy: f32,
    pub specificity: f32,
    pub true_positive_rate: f32,
    pub false_positive_rate: f32,
    // Could also be an enum for different metric types
    pub temporal_consistency: Option<f32>,
    pub temporal_stability: Option<f32>,
    pub avg_foreground_ratio: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationQuality {
    pub avg_connected_components: f32,
    pub avg_component_size: f32,
    pub fragmentation_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResults {
    pub video_info: VideoInfo,
    pub model_parameters: ModelParameters,
    pub accuracy_metrics: AccuracyMetrics,
    pub segmentation_quality: SegmentationQuality,
}

#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    precision: f32,
    recall: f32,
    f1_score: f32,
    accuracy: f32,
    specificity: f32,
    false_positive_rate: f32,
    true_positive_rate: f32,
    area_under_curve: f32,
    processing_time: f32,
    results: Vec<EvaluationResults>,
    frames_col: Option<i32>,
    frames_row: Option<i32>,
}
impl EvaluationMetrics {
    pub fn new() -> Self {
        EvaluationMetrics {
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            accuracy: 0.0,
            specificity: 0.0,
            false_positive_rate: 0.0,
            true_positive_rate: 0.0,
            area_under_curve: 0.0,
            processing_time: 0.0,
            results: Vec::new(),
            frames_row: None,
            frames_col: None,
        }
    }
    pub fn evaluate_video(&mut self, model: CodebookModel, video_path: &str, learning_frames: u32) {
        let mut processor = crate::video_processor::VideoProcessor::new(model.clone());
        let processing_stats = processor.process_video(
            video_path,
            None,
            learning_frames as usize,
            100,
            false,
            false,
            false,
        );
    }
    /// Computes segmentation quality for a video using a codebook.
    ///
    /// # Panics
    ///
    /// Panics if .
    fn compute_segmentation(
        &mut self,
        mut model: CodebookModel,
        video_path: &str,
        _learning_frames: u32,
    ) -> (SegmentationQuality, Vec<Vec<bool>>) {
        let mut cap =
            opencv::videoio::VideoCapture::from_file(video_path, opencv::videoio::CAP_ANY).unwrap();

        let mut segmentation_results = Vec::new();
        let mut _frame_count = 0;

        loop {
            let mut frame = opencv::core::Mat::default();
            let read_success = cap.read(&mut frame).unwrap_or(false);

            if !read_success || frame.empty() {
                break; // end of video
            }
            if self.frames_col.is_none() {
                self.frames_col = Some(frame.cols());
            }

            if self.frames_row.is_none() {
                self.frames_row = Some(frame.rows());
            }

            _frame_count += 1;

            let processed_frame = Self::mat_to_array1_vec(&frame);
            let foreground_mask = model.foreground_detect(&processed_frame);
            segmentation_results.push(foreground_mask);
        }
        if segmentation_results.is_empty() {
            return (
                SegmentationQuality {
                    avg_connected_components: 0.0,
                    avg_component_size: 0.0,
                    fragmentation_score: 0.0,
                },
                segmentation_results,
            );
        }

        let mut average_components = vec![];
        let mut average_component_sizes = vec![];

        for mask in segmentation_results.iter() {
            let mask_data: Vec<u8> = mask.iter().map(|&v| v as u8).collect();
            let rows = self.frames_row.unwrap_or(0);
            let cols = self.frames_col.unwrap_or(0);
            let temp_mat = opencv::core::Mat::from_slice(&mask_data).unwrap();
            let mask_mat = temp_mat.reshape(1, rows).unwrap();

            //Find connected components
            let mut labels = opencv::core::Mat::default();
            let num_labels = opencv::imgproc::connected_components(
                &mask_mat,
                &mut labels,
                8, // 8 connectivity
                opencv::core::CV_32S,
            )
            .unwrap();

            // push num of components, subtract 1 for the background
            average_components.push((num_labels - 1) as f32);

            // Calc comp sizes
            if num_labels > 1 {
                for label in 1..num_labels {
                    let mut count = 0;
                    for row in 0..rows {
                        for col in 0..cols {
                            let pixel_label = labels.at_2d::<i32>(row, col).unwrap();
                            if *pixel_label == label {
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        average_component_sizes.push(count as f32);
                    }
                }
            }
        }
        // Final Calculation
        let avg_components = if average_components.is_empty() {
            0.0
        } else {
            average_components.iter().sum::<f32>() / average_components.len() as f32
        };

        let avg_component_size = if average_component_sizes.is_empty() {
            0.0
        } else {
            average_component_sizes.iter().sum::<f32>() / average_component_sizes.len() as f32
        };

        //Fragmentation Score, higher = more fragmented
        let fragmentation_score = if avg_component_size > 0.0 {
            avg_components / avg_component_size
        } else {
            0.0
        };

        (
            SegmentationQuality {
                avg_connected_components: avg_components,
                avg_component_size,
                fragmentation_score,
            },
            segmentation_results,
        )
    }

    fn commpute_synthetic(
        &mut self,
        segmentation_quality: SegmentationQuality,
        segmentation_results: Vec<Vec<bool>>,
        truth_masks: &[Vec<bool>],
    ) -> AccuracyMetrics {
        if segmentation_results.is_empty() || truth_masks.is_empty() {
            return AccuracyMetrics {
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                accuracy: 0.0,
                specificity: 0.0,
                true_positive_rate: 0.0,
                false_positive_rate: 0.0,
                temporal_consistency: None,
                temporal_stability: None,
                avg_foreground_ratio: None,
            };
        }

        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;

        let frame_count = segmentation_results.len().min(truth_masks.len());
        for i in 0..frame_count {
            let pred = &segmentation_results[i];
            let truth = &truth_masks[i];
            for (p, t) in pred.iter().zip(truth.iter()) {
                match (*p, *t) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, true) => fn_ += 1,
                    (false, false) => tn += 1,
                }
            }
        }

        let precision = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f32 / (tp + fn_) as f32
        } else {
            0.0
        };
        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + self.recall)
        } else {
            0.0
        };
        let accuracy = if tp + tn + fp + fn_ > 0 {
            (tp + tn) as f32 / (tp + tn + fp + fn_) as f32
        } else {
            0.0
        };
        let specificity = if tn + fp > 0 {
            tn as f32 / (tn + fp) as f32
        } else {
            0.0
        };
        let true_positive_rate = recall;
        let false_positive_rate = if fp + tn > 0 {
            fp as f32 / (fp + tn) as f32
        } else {
            0.0
        };

        // Calculate temporal consistancy
        let mut consistency_scores = vec![];
        for i in 1..segmentation_results.len() {
            let prev_mask = &segmentation_results[i - 1];
            let curr_mask = &segmentation_results[i];

            let union = prev_mask
                .iter()
                .zip(curr_mask.iter())
                .filter(|(a, b)| **a || **b)
                .count();
            let intersection = prev_mask
                .iter()
                .zip(curr_mask.iter())
                .filter(|(a, b)| **a && **b)
                .count();
            if union > 0 {
                let jaccard_index = intersection as f32 / union as f32;
                consistency_scores.push(jaccard_index);
            } else {
                consistency_scores.push(1.0)
            }
        }
        let temporal_consistency = if consistency_scores.is_empty() {
            0.0
        } else {
            consistency_scores.iter().sum::<f32>() / consistency_scores.len() as f32
        };

        // Calculate average foreground ratio
        let total_pixels: usize = segmentation_results.iter().map(|m| m.len()).sum();
        let foreground_pixels: usize = segmentation_results
            .iter()
            .map(|m| m.iter().filter(|&&b| b).count())
            .sum();
        let avg_foreground_ratio = if total_pixels > 0 {
            foreground_pixels as f32 / total_pixels as f32
        } else {
            0.0
        };
        let temporal_stability = if segmentation_results.len() < 2 {
            1.0
        } else {
            let pixel_count = match (self.frames_col, self.frames_row) {
                (Some(cols), Some(rows)) => (cols * rows) as usize,
                _ => 0,
            };
            let mut stable_pixels = 0;

            for pixel_idx in 0..pixel_count {
                let first_val = segmentation_results[0][pixel_idx];
                let is_stable = segmentation_results
                    .iter()
                    .all(|mask| mask[pixel_idx] == first_val);
                if is_stable {
                    stable_pixels += 1;
                }
            }
            stable_pixels as f32 / pixel_count as f32
        };
        AccuracyMetrics {
            precision,
            recall,
            f1_score,
            accuracy,
            specificity,
            true_positive_rate,
            false_positive_rate,
            temporal_consistency: Some(temporal_consistency),
            temporal_stability: Some(temporal_stability),
            avg_foreground_ratio: Some(avg_foreground_ratio),
        }
    }
    fn mat_to_array1_vec(
        frame: &opencv::prelude::Mat,
    ) -> Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>>> {
        use opencv::prelude::*;

        let rows = frame.rows();
        let cols = frame.cols();
        let channels = frame.channels();

        let mut result = Vec::with_capacity((rows * cols) as usize);

        // Convert Mat to Vec<Array1<f32>>
        for row in 0..rows {
            for col in 0..cols {
                let mut pixel_data = vec![0.0f32; channels as usize];

                // Extract pixel values for all channels
                for ch in 0..channels {
                    let pixel_value = frame
                        .at_2d::<opencv::core::Vec3b>(row, col)
                        .map(|pixel| pixel[ch as usize] as f32)
                        .unwrap_or(0.0);
                    pixel_data[ch as usize] = pixel_value;
                }

                result.push(Array1::from_vec(pixel_data));
            }
        }

        result
    }
}
