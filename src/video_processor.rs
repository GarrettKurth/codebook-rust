use crate::codebook::CodebookModel;
use ndarray::Array1;
use opencv::{prelude::*, videoio};
use anyhow::{Result, Context};
use std::time::Instant;
use std::path::{Path, PathBuf};
use std::fs;

#[derive(Debug, Clone)]
pub struct VideoProcessor {
    pub model: CodebookModel,
}

#[derive(Debug, Clone)]
pub struct ProcessingStats {
    pub total_frames: usize,
    pub learning_frames: usize,
    pub processing_frames: usize,
    pub total_processing_time: f64,
    pub avg_processing_time: f64,
}
impl VideoProcessor {
    pub fn new(model: CodebookModel) -> Self {
        VideoProcessor { model }
    }
    fn create_result_visualization(
        &self,
        frame: &opencv::prelude::Mat,
        foreground_mask: &Vec<bool>,
    ) -> opencv::prelude::Mat {
        let mut result = frame.clone();
        let rows = frame.rows();
        let cols = frame.cols();

        for row in 0..rows {
            for col in 0..cols {
                let idx = (row * cols + col) as usize;
                if foreground_mask[idx] {
                    // Highlight foreground pixels in red
                    result
                        .at_2d_mut::<opencv::core::Vec3b>(row, col)
                        .map(|pixel| {
                            pixel[0] = 0; // Blue
                            pixel[1] = 0; // Green
                            pixel[2] = 255; // Red
                        })
                        .unwrap();
                }
            }
        }
        result
    }

    fn create_binary_mask(
        &self,
        frame: &opencv::prelude::Mat,
        foreground_mask: &Vec<bool>,
    ) -> opencv::prelude::Mat {
        let rows = frame.rows();
        let cols = frame.cols();
        let mut mask = opencv::core::Mat::zeros(rows, cols, opencv::core::CV_8UC1).unwrap().to_mat().unwrap();

        for row in 0..rows {
            for col in 0..cols {
                let idx = (row * cols + col) as usize;
                let pixel_value = if foreground_mask[idx] { 255u8 } else { 0u8 };
                mask.at_2d_mut::<u8>(row, col).map(|pixel| *pixel = pixel_value).unwrap();
            }
        }
        mask
    }

    /// Helper function to list image files in a directory and sort them
    fn get_sorted_image_files(folder_path: &str) -> Result<Vec<PathBuf>> {
        let path = Path::new(folder_path);
        
        if !path.exists() {
            anyhow::bail!("Folder does not exist: {}", folder_path);
        }
        
        if !path.is_dir() {
            anyhow::bail!("Path is not a directory: {}", folder_path);
        }

        let mut image_files: Vec<PathBuf> = fs::read_dir(path)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file() && 
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| {
                        let ext_lower = ext.to_lowercase();
                        ext_lower == "png" || ext_lower == "jpg" || 
                        ext_lower == "jpeg" || ext_lower == "bmp"
                    })
                    .unwrap_or(false)
            })
            .collect();

        // Sort by filename for consistent ordering
        image_files.sort();

        if image_files.is_empty() {
            anyhow::bail!("No image files (PNG, JPG, BMP) found in: {}", folder_path);
        }

        Ok(image_files)
    }

    /// Process a folder of individual frame images (PNG, JPG, BMP)
    /// folder_path: Path to folder containing sequential image frames
    /// output_folder: Optional path for output frames
    /// learning_frames: Number of frames to use for background learning
    /// cleanup_interval: Interval for cleaning stale codewords
    /// save_frames: Whether to save individual result frames
    /// save_binary_masks: Whether to save raw binary masks (white=fg, black=bg)
    pub fn process_frame_folder(
        &mut self,
        folder_path: &str,
        output_folder: Option<&str>,
        learning_frames: usize,
        cleanup_interval: usize,
        save_frames: bool,
        save_binary_masks: bool,
    ) -> Result<ProcessingStats> {
        let start_time = Instant::now();
        
        let image_files = Self::get_sorted_image_files(folder_path)?;
        let total_frames = image_files.len();

        if let Some(out_folder) = output_folder {
            fs::create_dir_all(out_folder)
                .context(format!("Failed to create output folder: {}", out_folder))?;
        }

        let mut frame_count = 0;
        let mut processing_frames = 0;

        for image_path in &image_files {
            // Read image frame
            let frame = opencv::imgcodecs::imread(
                image_path.to_str().context("Invalid image path")?,
                opencv::imgcodecs::IMREAD_COLOR,
            ).context(format!("Failed to read image: {:?}", image_path))?;

            if frame.empty() {
                eprintln!("Skipping empty frame: {:?}", image_path);
                continue;
            }

            let processed_frame = Self::mat_to_array1_vec(&frame);

            if frame_count < learning_frames {
                self.model.learning_phase(&processed_frame);
                
                // Progress update for learning phase
                if frame_count % 10 == 0 || frame_count == learning_frames - 1 {
                    let progress = (frame_count + 1) as f32 / learning_frames as f32 * 100.0;
                    println!("Learning phase: {:.1}% ({}/{})", progress, frame_count + 1, learning_frames);
                }
            } else {
                // Switch to processing phase
                if frame_count == learning_frames {
                    println!("Starting foreground detection phase...");
                }
                
                let foreground_mask = self.model.foreground_detect(&processed_frame);
                let result_frame = self.create_result_visualization(&frame, &foreground_mask);
                let binary_mask = self.create_binary_mask(&frame, &foreground_mask);

                if save_frames {
                    let output_path = if let Some(out_folder) = output_folder {
                        let filename = image_path.file_name()
                            .context("No filename in path")?
                            .to_str()
                            .context("Invalid filename")?;
                        format!("{}/{}", out_folder, filename)
                    } else {
                        // Save with _result suffix in same directory
                        let stem = image_path.file_stem()
                            .context("No file stem")?
                            .to_str()
                            .context("Invalid file stem")?;
                        let parent = image_path.parent()
                            .context("No parent directory")?
                            .to_str()
                            .context("Invalid parent path")?;
                        format!("{}/{}_result.png", parent, stem)
                    };
                    
                    opencv::imgcodecs::imwrite(
                        &output_path,
                        &result_frame,
                        &opencv::core::Vector::new(),
                    ).context("Failed to save result frame")?;
                }

                if save_binary_masks {
                    let mask_path = if let Some(out_folder) = output_folder {
                        let stem = image_path.file_stem()
                            .context("No file stem")?
                            .to_str()
                            .context("Invalid file stem")?;
                        format!("{}/{}_mask.png", out_folder, stem)
                    } else {
                        let stem = image_path.file_stem()
                            .context("No file stem")?
                            .to_str()
                            .context("Invalid file stem")?;
                        let parent = image_path.parent()
                            .context("No parent directory")?
                            .to_str()
                            .context("Invalid parent path")?;
                        format!("{}/{}_mask.png", parent, stem)
                    };
                    
                    opencv::imgcodecs::imwrite(
                        &mask_path,
                        &binary_mask,
                        &opencv::core::Vector::new(),
                    ).context("Failed to save binary mask")?;
                }
                
                processing_frames += 1;
                
                if processing_frames % 25 == 0 || processing_frames == 1 {
                    let total_progress = frame_count as f32 / total_frames as f32 * 100.0;
                    println!("Processing: {:.1}% ({}/{}) - {} frames processed", 
                            total_progress, frame_count + 1, total_frames, processing_frames);
                }
            }

            frame_count += 1;

            if frame_count % cleanup_interval == 0 {
                self.model.cleanup_codewords();
                println!("Cleaned up stale codewords at frame {}", frame_count);
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();
        let avg_time = if processing_frames > 0 {
            total_time / processing_frames as f64
        } else {
            0.0
        };

        Ok(ProcessingStats {
            total_frames: frame_count,
            learning_frames: learning_frames.min(frame_count),
            processing_frames,
            total_processing_time: total_time,
            avg_processing_time: avg_time,
        })
    }

    // input_path: Path to input video file
    // output_path: Path for output video (optional)
    // learning_frames: Number of frames to use for background learning
    // cleanup_interval: Interval for cleaning stale codewords
    // display_results: Whether to display results in real-time
    // save_frames: Whether to save individual result frames
    pub fn process_video(
        &mut self,
        input_path: &str,
        output_path: Option<&str>,
        learning_frames: usize,
        cleanup_interval: usize,
        _display_results: bool,
        save_frames: bool,
        save_binary_masks: bool,
    ) -> Result<ProcessingStats> {
        let start_time = Instant::now();
        
        let mut cap = videoio::VideoCapture::from_file(input_path, videoio::CAP_ANY)
            .context("Failed to create VideoCapture")?;
        
        if !cap.is_opened()? {
            anyhow::bail!("Failed to open video file: {}", input_path);
        }

        let total_frames = cap.get(videoio::CAP_PROP_FRAME_COUNT)? as usize;
        println!("Total frames in video: {}", total_frames);

        let mut out_writer: Option<videoio::VideoWriter> = None;
        if let Some(output_path) = output_path {
            let fourcc = videoio::VideoWriter::fourcc('M', 'P', '4', 'V')?;
            let fps = cap.get(videoio::CAP_PROP_FPS)?;
            let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
            let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
            
            out_writer = Some(videoio::VideoWriter::new(
                output_path,
                fourcc,
                fps,
                opencv::core::Size::new(width, height),
                true,
            )?);
        }

        let mut frame_count = 0;
        let mut processing_frames = 0;

        loop {
            let mut frame = opencv::core::Mat::default();
            let read_success = cap.read(&mut frame)?;
            
            if !read_success || frame.empty() {
                break; // End of video
            }

            // Convert frame to Vec<Array1<f32>> for processing
            let processed_frame = Self::mat_to_array1_vec(&frame);

            if frame_count < learning_frames {
                self.model.learning_phase(&processed_frame);
                
                // Progress update for learning phase
                if frame_count % 10 == 0 || frame_count == learning_frames - 1 {
                    let progress = (frame_count + 1) as f32 / learning_frames as f32 * 100.0;
                    println!("Learning phase: {:.1}% ({}/{})", progress, frame_count + 1, learning_frames);
                }
            } else {
                if frame_count == learning_frames {
                    println!("Starting foreground detection phase...");
                }
                
                let foreground_mask = self.model.foreground_detect(&processed_frame);
                let result_frame = self.create_result_visualization(&frame, &foreground_mask);
                let binary_mask = self.create_binary_mask(&frame, &foreground_mask);

                if let Some(ref mut writer) = out_writer {
                    writer.write(&result_frame)
                        .context("Failed to write frame to output video")?;
                }
                
                if save_frames {
                    let frame_filename = format!("frame_{:05}.png", frame_count);
                    opencv::imgcodecs::imwrite(
                        &frame_filename,
                        &result_frame,
                        &opencv::core::Vector::new(),
                    ).context("Failed to save frame")?;
                }

                if save_binary_masks {
                    let mask_filename = format!("mask_{:05}.png", frame_count);
                    opencv::imgcodecs::imwrite(
                        &mask_filename,
                        &binary_mask,
                        &opencv::core::Vector::new(),
                    ).context("Failed to save binary mask")?;
                }
                
                processing_frames += 1;
                
                if processing_frames % 25 == 0 {
                    let total_progress = frame_count as f32 / total_frames as f32 * 100.0;
                    println!("Processing: {:.1}% ({}/{}) - {} frames processed", 
                            total_progress, frame_count + 1, total_frames, processing_frames);
                }
            }

            frame_count += 1;

            if frame_count % cleanup_interval == 0 {
                self.model.cleanup_codewords();
                println!("Cleaned up stale codewords");
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();
        let avg_time = if processing_frames > 0 {
            total_time / processing_frames as f64
        } else {
            0.0
        };

        Ok(ProcessingStats {
            total_frames: frame_count,
            learning_frames: learning_frames.min(frame_count),
            processing_frames,
            total_processing_time: total_time,
            avg_processing_time: avg_time,
        })
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
