use ndarray::Array1;
use ndarray::Array2;
use opencv::{
    core::{self, CV_8UC1, CV_8UC3, Mat, MatExprTraitConst},
    prelude::*,
};
use serde::{Deserialize, Serialize};

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
pub struct QualityEvaluation {
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
    pub spatial_coherence: f32,
    pub flicker_rate: f32,
    pub fragmentation_score: f32,
    pub superpixel_purity: f32,
    pub structural_quality: StructuralQuality,
    pub overall_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralQuality {
    /// Average component size
    pub avg_component_size: f32,

    /// Variance in component sizes (lower is better)
    pub size_variance: f32,

    /// Average compactness of components (1.0 = perfect circle)
    pub avg_compactness: f32,

    /// Number of foreground components
    pub num_components: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationQuality {
    pub avg_connected_components: f32,
    pub avg_component_size: f32,
    pub fragmentation_score: f32,
}

pub fn calculate_spatial_coherence(
    frame: &Array2<u8>,
    mask: &Vec<bool>,
    width: usize,
    height: usize,
) -> f32 {
    let gradients = calculate_gradients(frame, width, height);
    let mask_boundaries = find_mask_boundaries(mask, width, height);

    let mut alignment_sum = 0.0;
    let mut boundary_count = 0;

    for (idx, &is_boundary) in mask_boundaries.iter().enumerate() {
        if is_boundary {
            alignment_sum += gradients[idx];
            boundary_count += 1;
        }
    }
    if boundary_count == 0 {
        return 0.5; // Neutral score if no boundaries
    }

    // Normalize to [0, 1]
    let avg_gradient_at_boundary = alignment_sum / boundary_count as f32;
    let max_possible_gradient = 255.0 * 1.414; // Approx max gradient magnitude for 8-bit images
    (avg_gradient_at_boundary / max_possible_gradient).min(1.0)
}

/// Calculate image gradients using Sobel operator
fn calculate_gradients(frame: &Array2<u8>, width: usize, height: usize) -> Vec<f32> {
    let mut gradients = vec![0.0; width * height];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;

            // Sobel X
            let gx = -1.0 * frame[[y - 1, x - 1]] as f32
                + 1.0 * frame[[y - 1, x + 1]] as f32
                + -2.0 * frame[[y, x - 1]] as f32
                + 2.0 * frame[[y, x + 1]] as f32
                + -1.0 * frame[[y + 1, x - 1]] as f32
                + 1.0 * frame[[y + 1, x + 1]] as f32;

            // Sobel Y
            let gy = -1.0 * frame[[y - 1, x - 1]] as f32
                - 2.0 * frame[[y - 1, x]] as f32
                - 1.0 * frame[[y - 1, x + 1]] as f32
                + 1.0 * frame[[y + 1, x - 1]] as f32
                + 2.0 * frame[[y + 1, x]] as f32
                + 1.0 * frame[[y + 1, x + 1]] as f32;

            // Gradient magnitude
            gradients[idx] = (gx * gx + gy * gy).sqrt();
        }
    }

    gradients
}
/// Find boundaries in the binary mask
fn find_mask_boundaries(mask: &Vec<bool>, width: usize, height: usize) -> Vec<bool> {
    let mut boundaries = vec![false; width * height];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;

            // Check if any 8-connected neighbor has different classification
            if mask[idx] {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }

                        let ny = (y as i32 + dy) as usize;
                        let nx = (x as i32 + dx) as usize;
                        let nidx = ny * width + nx;

                        if !mask[nidx] {
                            boundaries[idx] = true;
                            break;
                        }
                    }
                    if boundaries[idx] {
                        break;
                    }
                }
            }
        }
    }
    boundaries
}
fn calculate_flicker_rate(masks: &[Vec<bool>], width: usize, height: usize) -> f32 {
    if masks.len() < 3 {
        return 0.0; // Can't detect flicker with <3 frames
    }

    let pixel_count = width * height;
    let num_frames = masks.len();
    let mut total_change_rate = 0.0;

    for pixel_idx in 0..pixel_count {
        let mut changes = 0;
        for frame_idx in 1..num_frames {
            if masks[frame_idx][pixel_idx] != masks[frame_idx - 1][pixel_idx] {
                changes += 1;
            }
        }

        // Calculate change rate: changes / possible_changes
        // For a pixel that flickers every frame: change_rate = 1.0
        // For a pixel that changes once (bg→fg or fg→bg): change_rate = 1/(num_frames-1)
        // For stable pixel: change_rate = 0.0
        let max_possible_changes = (num_frames - 1) as f32;
        let change_rate = changes as f32 / max_possible_changes;

        // Only count as flicker if change_rate > threshold
        // A pixel that changes once has rate = 1/199 ≈ 0.005 (not flicker)
        // A pixel that changes 10 times has rate = 10/199 ≈ 0.05 (mild flicker)
        // A pixel that changes 40 times has rate = 40/199 ≈ 0.20 (high flicker)
        // Threshold: consider flickering if changes > 2.5% of frames
        let flicker_threshold = 0.025;

        if change_rate > flicker_threshold {
            total_change_rate += change_rate;
        }
    }

    // Return average flicker intensity (not just binary count)
    // This gives more nuanced measure: how much pixels are flickering
    total_change_rate / pixel_count as f32
}

/// Calculate fragmentation score using connected components
fn calculate_fragmentation(
    mask: &Vec<bool>,
    width: usize,
    height: usize,
) -> (f32, StructuralQuality) {
    let components = find_connected_components(mask, width, height);

    if components.is_empty() {
        return (
            0.0,
            StructuralQuality {
                avg_component_size: 0.0,
                size_variance: 0.0,
                avg_compactness: 0.0,
                num_components: 0,
            },
        );
    }

    let total_foreground: usize = mask.iter().filter(|&&x| x).count();
    let num_components = components.len();

    // Calculate component statistics
    let sizes: Vec<usize> = components.iter().map(|c| c.len()).collect();
    let avg_size = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;

    let variance = sizes
        .iter()
        .map(|&size| {
            let diff = size as f32 - avg_size;
            diff * diff
        })
        .sum::<f32>()
        / sizes.len() as f32;

    // Calculate compactness for each component
    let compactness_scores: Vec<f32> = components
        .iter()
        .map(|component| calculate_compactness(component, width))
        .collect();

    let avg_compactness = compactness_scores.iter().sum::<f32>() / compactness_scores.len() as f32;

    // Fragmentation score: high when many small components
    // Formula: components / sqrt(total_foreground)
    // Normalize by expected maximum (empirically ~20 for typical scenes)
    let fragmentation = (num_components as f32) / (total_foreground as f32).sqrt();
    let normalized_fragmentation = (fragmentation / 20.0).min(1.0); // Changed from 10.0 to 20.0

    (
        normalized_fragmentation,
        StructuralQuality {
            avg_component_size: avg_size,
            size_variance: variance,
            avg_compactness,
            num_components,
        },
    )
}

/// Find connected components using flood fill
fn find_connected_components(mask: &Vec<bool>, width: usize, height: usize) -> Vec<Vec<usize>> {
    let mut visited = vec![false; width * height];
    let mut components = Vec::new();

    for idx in 0..mask.len() {
        if mask[idx] && !visited[idx] {
            let component = flood_fill(mask, &mut visited, idx, width, height);
            if !component.is_empty() {
                components.push(component);
            }
        }
    }

    components
}

/// Flood fill to find a connected component
fn flood_fill(
    mask: &Vec<bool>,
    visited: &mut Vec<bool>,
    start: usize,
    width: usize,
    height: usize,
) -> Vec<usize> {
    let mut component = Vec::new();
    let mut stack = vec![start];

    while let Some(idx) = stack.pop() {
        if visited[idx] {
            continue;
        }

        visited[idx] = true;
        component.push(idx);

        let y = idx / width;
        let x = idx % width;

        // Check 4-connected neighbors
        let neighbors = [
            (y.wrapping_sub(1), x), // Up
            (y + 1, x),             // Down
            (y, x.wrapping_sub(1)), // Left
            (y, x + 1),             // Right
        ];

        for (ny, nx) in neighbors {
            if ny < height && nx < width {
                let nidx = ny * width + nx;
                if mask[nidx] && !visited[nidx] {
                    stack.push(nidx);
                }
            }
        }
    }

    component
}

/// Calculate compactness of a component (4π*area / perimeter²)
fn calculate_compactness(component: &Vec<usize>, width: usize) -> f32 {
    let area = component.len() as f32;

    // Calculate perimeter by counting boundary pixels
    let mut perimeter = 0;
    let component_set: std::collections::HashSet<_> = component.iter().cloned().collect();

    for &idx in component {
        let y = idx / width;
        let x = idx % width;

        // Check 4-connected neighbors
        let neighbors = [
            (y.wrapping_sub(1), x),
            (y + 1, x),
            (y, x.wrapping_sub(1)),
            (y, x + 1),
        ];

        for (ny, nx) in neighbors {
            let nidx = ny * width + nx;
            if !component_set.contains(&nidx) {
                perimeter += 1;
            }
        }
    }

    if perimeter == 0 {
        return 0.0;
    }

    // Compactness = 4π*area / perimeter²
    (4.0 * std::f32::consts::PI * area) / (perimeter as f32 * perimeter as f32)
}

/// Calculate actual superpixel purity using SLIC algorithm
/// Returns purity score (0.0-1.0) where higher means better segmentation
fn calculate_superpixel_purity(
    frame: &Array2<u8>,
    mask: &Vec<bool>,
    width: usize,
    height: usize,
) -> f32 {
    // Convert grayscale Array2 to BGR Mat for SLIC
    let mut bgr_mat = match Mat::zeros(height as i32, width as i32, CV_8UC3) {
        Ok(m) => m.to_mat().unwrap(),
        Err(_) => return 0.5,
    };

    for row in 0..height {
        for col in 0..width {
            let gray_val = frame[[row, col]];
            // Set all BGR channels to same value (grayscale)
            if let Ok(pixel) = bgr_mat.at_2d_mut::<core::Vec3b>(row as i32, col as i32) {
                pixel.clone_from(&core::Vec3b::from([gray_val, gray_val, gray_val]));
            }
        }
    }

    // Convert mask to Mat
    let mut mask_mat = match Mat::zeros(height as i32, width as i32, CV_8UC1) {
        Ok(m) => m.to_mat().unwrap(),
        Err(_) => return 0.5,
    };

    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            let val = if mask[idx] { 255u8 } else { 0u8 };
            if let Ok(pixel) = mask_mat.at_2d_mut::<u8>(row as i32, col as i32) {
                *pixel = val;
            }
        }
    }

    // Generate superpixels using SLIC
    let region_size = 20; // Superpixel size
    let ruler = 10.0; // Compactness parameter

    let mut slic = match opencv::ximgproc::create_superpixel_slic(
        &bgr_mat,
        opencv::ximgproc::SLIC,
        region_size,
        ruler,
    ) {
        Ok(s) => s,
        Err(_) => return 0.5, // Fallback if SLIC fails
    };

    if slic.iterate(10).is_err() {
        return 0.5;
    }

    let mut labels = Mat::default();
    if slic.get_labels(&mut labels).is_err() {
        return 0.5;
    }

    let num_superpixels = match slic.get_number_of_superpixels() {
        Ok(n) => n,
        Err(_) => return 0.5,
    };

    // Calculate purity for each superpixel
    let mut pure_superpixels = 0;
    let mut total_superpixels = 0;

    for sp_id in 0..num_superpixels {
        let mut fg_count = 0;
        let mut bg_count = 0;

        // Count foreground/background pixels in this superpixel
        for row in 0..height {
            for col in 0..width {
                let label = match labels.at_2d::<i32>(row as i32, col as i32) {
                    Ok(&l) => l,
                    Err(_) => continue,
                };

                if label == sp_id {
                    let idx = row * width + col;
                    if mask[idx] {
                        fg_count += 1;
                    } else {
                        bg_count += 1;
                    }
                }
            }
        }

        if fg_count + bg_count > 0 {
            total_superpixels += 1;
            // Superpixel is pure if > 90% is one class
            let total = fg_count + bg_count;
            let purity = fg_count.max(bg_count) as f32 / total as f32;
            if purity > 0.9 {
                pure_superpixels += 1;
            }
        }
    }

    if total_superpixels == 0 {
        return 0.0;
    }

    pure_superpixels as f32 / total_superpixels as f32
}

/// Compute a full QualityEvaluation using segmentation results, ground truth, and all metrics.
/// This is a static function and expects frame size and a sample frame for spatial metrics.
/// `frames` is a vector of grayscale frames (Array2<u8>) corresponding to each mask.
pub fn compute_quality_evaluation(
    segmentation_results: &[Vec<bool>],
    truth_masks: &[Vec<bool>],
    frames: &[Array2<u8>],
    width: usize,
    height: usize,
) -> QualityEvaluation {
    // Compute accuracy metrics

    let frame_count = segmentation_results
        .len()
        .min(truth_masks.len())
        .min(frames.len());

    let (
        precision,
        recall,
        f1_score,
        accuracy,
        specificity,
        true_positive_rate,
        false_positive_rate,
    ) = if !truth_masks.is_empty() {
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;
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
            2.0 * precision * recall / (precision + recall)
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
        (
            precision,
            recall,
            f1_score,
            accuracy,
            specificity,
            true_positive_rate,
            false_positive_rate,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    };
    // Temporal consistency
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

    // Average foreground ratio
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

    // Temporal stability
    let temporal_stability = if segmentation_results.len() < 2 {
        1.0
    } else {
        let pixel_count = width * height;
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

    // Spatial coherence: average over all frames/masks
    let spatial_coherence = if !segmentation_results.is_empty()
        && frames.len() == segmentation_results.len()
    {
        let mut sum = 0.0;
        for i in 0..frame_count {
            sum += calculate_spatial_coherence(&frames[i], &segmentation_results[i], width, height);
        }
        sum / frame_count as f32
    } else {
        0.0
    };

    // Flicker rate
    let flicker_rate = calculate_flicker_rate(segmentation_results, width, height);

    // Fragmentation and structural quality: average over all masks
    let (fragmentation_score, structural_quality) = if !segmentation_results.is_empty() {
        let mut frag_sum = 0.0;
        let mut sq_sum = StructuralQuality {
            avg_component_size: 0.0,
            size_variance: 0.0,
            avg_compactness: 0.0,
            num_components: 0,
        };
        for mask in segmentation_results.iter().take(frame_count) {
            let (frag, sq) = calculate_fragmentation(mask, width, height);
            frag_sum += frag;
            sq_sum.avg_component_size += sq.avg_component_size;
            sq_sum.size_variance += sq.size_variance;
            sq_sum.avg_compactness += sq.avg_compactness;
            sq_sum.num_components += sq.num_components;
        }
        let n = frame_count as f32;
        (
            frag_sum / n,
            StructuralQuality {
                avg_component_size: sq_sum.avg_component_size / n,
                size_variance: sq_sum.size_variance / n,
                avg_compactness: sq_sum.avg_compactness / n,
                num_components: (sq_sum.num_components as f32 / n).round() as usize,
            },
        )
    } else {
        (
            0.0,
            StructuralQuality {
                avg_component_size: 0.0,
                size_variance: 0.0,
                avg_compactness: 0.0,
                num_components: 0,
            },
        )
    };

    // Superpixel purity: average over all frames/masks
    let superpixel_purity = if !segmentation_results.is_empty()
        && frames.len() == segmentation_results.len()
    {
        let mut sum = 0.0;
        for i in 0..frame_count {
            sum += calculate_superpixel_purity(&frames[i], &segmentation_results[i], width, height);
        }
        sum / frame_count as f32
    } else {
        0.0
    };

    // Overall score (simple average of main metrics, can be customized)
    let overall_score = (f1_score
        + temporal_consistency
        + spatial_coherence
        + (1.0 - fragmentation_score)
        + superpixel_purity)
        / 5.0;

    QualityEvaluation {
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
        spatial_coherence,
        flicker_rate,
        fragmentation_score,
        superpixel_purity,
        structural_quality,
        overall_score,
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
