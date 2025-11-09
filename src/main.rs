use clap::{Parser, Subcommand};
use std::path::PathBuf;
use anyhow::{Result, Context};
use codebook_model::{CodebookModel, VideoProcessor};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(about = "A CLI tool for background subtraction using the Codebook algorithm")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process video with codebook algorithm (learn and detect in one step)
    Process {
        /// Input video file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output video file path
        #[arg(short, long)]
        output: PathBuf,

        /// Number of frames to use for background learning
        #[arg(short, long, default_value_t = 100)]
        learning_frames: usize,

        /// Alpha parameter (lower bound factor for brightness matching)
        #[arg(long, default_value_t = 0.6)]
        alpha: f32,

        /// Beta parameter (upper bound factor for brightness matching)
        #[arg(long, default_value_t = 1.2)]
        beta: f32,

        /// Lambda parameter (maximum time a codeword can go without being accessed)
        #[arg(long, default_value_t = 100.0)]
        lambda: f32,

        /// Epsilon parameter (color distortion threshold)
        #[arg(long, default_value_t = 10.0)]
        epsilon: f32,
    },

    /// Process folder of individual frame images (PNG, JPG, BMP)
    ProcessFolder {
        /// Input folder containing sequential image frames
        #[arg(short, long)]
        input: PathBuf,

        /// Output folder for result frames (optional, defaults to input folder with _result suffix)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Number of frames to use for background learning
        #[arg(short, long, default_value_t = 100)]
        learning_frames: usize,

        /// Alpha parameter (lower bound factor for brightness matching)
        #[arg(long, default_value_t = 0.6)]
        alpha: f32,

        /// Beta parameter (upper bound factor for brightness matching)
        #[arg(long, default_value_t = 1.2)]
        beta: f32,

        /// Lambda parameter (maximum time a codeword can go without being accessed)
        #[arg(long, default_value_t = 100.0)]
        lambda: f32,

        /// Epsilon parameter (color distortion threshold)
        #[arg(long, default_value_t = 10.0)]
        epsilon: f32,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Process {
            input,
            output,
            learning_frames,
            alpha,
            beta,
            lambda,
            epsilon,
        } => {
            process_video(
                input,
                output,
                learning_frames,
                alpha,
                beta,
                lambda,
                epsilon,
            )
        }
        Commands::ProcessFolder {
            input,
            output,
            learning_frames,
            alpha,
            beta,
            lambda,
            epsilon,
        } => process_folder(
            input,
            output,
            learning_frames,
            alpha,
            beta,
            lambda,
            epsilon,
        ),
    }
}

fn process_video(
    input: PathBuf,
    output: PathBuf,
    learning_frames: usize,
    alpha: f32,
    beta: f32,
    lambda: f32,
    epsilon: f32,
) -> Result<()> {
    // Validate input file exists
    if !input.exists() {
        anyhow::bail!("Input file does not exist: {}", input.display());
    }

    let input_str = input.to_str().context("Input path contains invalid UTF-8")?;
    let output_str = output.to_str().context("Output path contains invalid UTF-8")?;

    println!("Codebook Background Subtraction - Process Mode");
    println!("Input:  {}", input_str);
    println!("Output: {}", output_str);
    println!("Parameters:");
    println!("  - Learning frames: {}", learning_frames);
    println!("  - Alpha: {}", alpha);
    println!("  - Beta: {}", beta);
    println!("  - Lambda: {}", lambda);
    println!("  - Epsilon: {}", epsilon);
    println!();

    // Get video dimensions
    let (vid_width, vid_height) = get_video_dimensions(&input, None, None)?;
    println!("Video dimensions: {}x{}", vid_width, vid_height);
    println!();

    // Create codebook model
    let model = CodebookModel::new(alpha, beta, lambda, epsilon, vid_width, vid_height);

    // Create video processor
    let mut processor = VideoProcessor::new(model);

    // Process the video
    println!("Processing video...");
    let stats = processor.process_video(
        input_str,
        Some(output_str),
        learning_frames,
        50, // default cleanup interval
        false,
        false, // save_frames
        false, // export_binary_masks
    )?;

    println!("Processing complete!");
    println!("Statistics:");
    println!("  - Total frames processed: {}", stats.total_frames);
    println!("  - Processing time: {:.2}s", stats.total_processing_time);
    println!("  - Average time per frame: {:.2}ms", stats.avg_processing_time * 1000.0);
    println!("  - Output saved to: {}", output_str);

    Ok(())
}

fn get_video_dimensions(input: &PathBuf, width: Option<usize>, height: Option<usize>) -> Result<(usize, usize)> {
    // If dimensions are provided, use them
    if let (Some(width), Some(height)) = (width, height) {
        return Ok((width, height));
    }

    // Otherwise, auto-detect from video file
    use opencv::{prelude::*, videoio};
    
    let input_str = input.to_str()
        .context("Input path contains invalid UTF-8")?;
    
    let cap = videoio::VideoCapture::from_file(input_str, videoio::CAP_ANY)
        .context("Failed to open video file for dimension detection")?;
    
    if !cap.is_opened()? {
        anyhow::bail!("Could not open video file: {}", input_str);
    }

    let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as usize;
    let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as usize;

    if width == 0 || height == 0 {
        anyhow::bail!("Could not determine video dimensions");
    }

    Ok((width, height))
}

fn process_folder(
    input: PathBuf,
    output: Option<PathBuf>,
    learning_frames: usize,
    alpha: f32,
    beta: f32,
    lambda: f32,
    epsilon: f32,
) -> Result<()> {
    // Validate input folder exists
    if !input.exists() {
        anyhow::bail!("Input folder does not exist: {}", input.display());
    }
    if !input.is_dir() {
        anyhow::bail!("Input path is not a directory: {}", input.display());
    }

    let input_str = input.to_str().context("Input path contains invalid UTF-8")?;
    let output_str = output.as_ref().and_then(|p| p.to_str());

    println!("Codebook Background Subtraction - Process Folder Mode");
    println!("Input folder:  {}", input_str);
    if let Some(out) = output_str {
        println!("Output folder: {}", out);
    } else {
        println!("Output: Results saved with _result suffix in input folder");
    }
    println!("Parameters:");
    println!("  - Learning frames: {}", learning_frames);
    println!("  - Alpha: {}", alpha);
    println!("  - Beta: {}", beta);
    println!("  - Lambda: {}", lambda);
    println!("  - Epsilon: {}", epsilon);
    println!();

    // Get image dimensions from first image
    let (img_width, img_height) = get_image_dimensions(&input)?;
    println!("Image dimensions: {}x{}", img_width, img_height);
    println!();

    // Create codebook model
    let model = CodebookModel::new(alpha, beta, lambda, epsilon, img_width, img_height);

    // Create video processor
    let mut processor = VideoProcessor::new(model);

    // Process the folder
    println!("Processing image folder...");
    let stats = processor.process_frame_folder(
        input_str,
        output_str,
        learning_frames,
        50, // default cleanup interval
        true, // save_frames
        false, // export_binary_masks
    )?;

    println!("Processing complete!");
    println!("Statistics:");
    println!("  - Total frames processed: {}", stats.total_frames);
    println!("  - Processing time: {:.2}s", stats.total_processing_time);
    println!("  - Average time per frame: {:.2}ms", stats.avg_processing_time * 1000.0);
    if let Some(out) = output_str {
        println!("  - Output saved to: {}", out);
    }

    Ok(())
}

fn get_image_dimensions(folder: &PathBuf) -> Result<(usize, usize)> {
    use opencv::{prelude::*, imgcodecs};
    use std::fs;

    // Find first image file
    let first_image = fs::read_dir(folder)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|path| {
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
        .context("No image files found in folder")?;

    // Read first image to get dimensions
    let img = imgcodecs::imread(
        first_image.to_str().context("Invalid image path")?,
        imgcodecs::IMREAD_COLOR,
    )?;

    if img.empty() {
        anyhow::bail!("Could not read first image for dimensions");
    }

    let width = img.cols() as usize;
    let height = img.rows() as usize;

    if width == 0 || height == 0 {
        anyhow::bail!("Invalid image dimensions");
    }

    Ok((width, height))
}
