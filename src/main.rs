use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::fs;
use anyhow::{Result, Context};
use codebook_model::{
    evaluation::{self, ParameterSweepConfig},
    CodebookModel,
    VideoProcessor,
};

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

        /// Optional path to persist the learned codebook (.cbm or .json)
        #[arg(long, value_name = "PATH")]
        save_codebook: Option<PathBuf>,
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

        /// Optional path to persist the learned codebook (.cbm or .json)
        #[arg(long, value_name = "PATH")]
        save_codebook: Option<PathBuf>,
    },

    /// Run a parameter sweep evaluation against a learned codebook
    Evaluate {
        /// Path to a serialized codebook (.cbm or .json)
        #[arg(long)]
        codebook: PathBuf,

        /// Input video path used for evaluation
        #[arg(long)]
        video: PathBuf,

        /// Output JSON file to store sweep results
        #[arg(long)]
        output: PathBuf,

        /// Optional folder containing ground-truth masks
        #[arg(long)]
        truth_masks: Option<PathBuf>,

        /// Override alpha sweep values (comma separated)
        #[arg(long)]
        alpha_values: Option<String>,

        /// Override beta sweep values (comma separated)
        #[arg(long)]
        beta_values: Option<String>,

        /// Override epsilon sweep values (comma separated)
        #[arg(long)]
        epsilon_values: Option<String>,

        /// Override lambda sweep values (comma separated)
        #[arg(long)]
        lambda_values: Option<String>,

        /// Sample only every Nth frame when evaluating
        #[arg(long, default_value_t = 1)]
        frame_stride: usize,

        /// Maximum number of frames to sample (after stride)
        #[arg(long)]
        max_frames: Option<usize>,
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
            save_codebook,
        } => {
            process_video(
                input,
                output,
                learning_frames,
                alpha,
                beta,
                lambda,
                epsilon,
                save_codebook,
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
            save_codebook,
        } => process_folder(
            input,
            output,
            learning_frames,
            alpha,
            beta,
            lambda,
            epsilon,
            save_codebook,
        ),
        Commands::Evaluate {
            codebook,
            video,
            output,
            truth_masks,
            alpha_values,
            beta_values,
            epsilon_values,
            lambda_values,
            frame_stride,
            max_frames,
        } => run_evaluation_command(
            codebook,
            video,
            output,
            truth_masks,
            alpha_values,
            beta_values,
            epsilon_values,
            lambda_values,
            frame_stride,
            max_frames,
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
    save_codebook: Option<PathBuf>,
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

    if let Some(path) = save_codebook {
        save_codebook_to_path(&processor.model, &path)?;
        println!("Saved codebook to {}", path.display());
    }

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
    save_codebook: Option<PathBuf>,
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

    if let Some(path) = save_codebook {
        save_codebook_to_path(&processor.model, &path)?;
        println!("Saved codebook to {}", path.display());
    }

    Ok(())
}

fn save_codebook_to_path(model: &CodebookModel, path: &PathBuf) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create directory {}", parent.display())
            })?;
        }
    }

    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
        .unwrap_or_else(|| "".to_string());

    if extension == "json" {
        model
            .save_to_json(path)
            .map_err(|e| anyhow::anyhow!("Failed to save JSON codebook: {}", e))?;
    } else {
        model
            .save_to_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to save binary codebook: {}", e))?;
    }

    Ok(())
}

fn run_evaluation_command(
    codebook_path: PathBuf,
    video_path: PathBuf,
    output_path: PathBuf,
    truth_masks: Option<PathBuf>,
    alpha_values: Option<String>,
    beta_values: Option<String>,
    epsilon_values: Option<String>,
    lambda_values: Option<String>,
    frame_stride: usize,
    max_frames: Option<usize>,
) -> Result<()> {
    let mut config = ParameterSweepConfig::default();

    if let Some(overrides) = parse_value_list(alpha_values, "alpha")? {
        config.alpha_values = overrides;
    }
    if let Some(overrides) = parse_value_list(beta_values, "beta")? {
        config.beta_values = overrides;
    }
    if let Some(overrides) = parse_value_list(epsilon_values, "epsilon")? {
        config.epsilon_values = overrides;
    }
    if let Some(overrides) = parse_value_list(lambda_values, "lambda")? {
        config.lambda_values = overrides;
    }

    config.frame_stride = frame_stride.max(1);
    config.max_frames = max_frames;

    let base_model = load_codebook_from_path(&codebook_path)?;

    let (width, height) = get_video_dimensions(&video_path, None, None)?;

    let truth_data = if let Some(mask_path) = truth_masks {
        Some(
            evaluation::load_truth_masks_from_folder(
                mask_path,
                width,
                height,
                config.frame_stride,
                config.max_frames,
            )
            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
        )
    } else {
        None
    };

    let video_str = video_path
        .to_str()
        .context("Video path contains invalid UTF-8")?;

    let sweep_results = evaluation::parameter_sweep(
        &base_model,
        video_str,
        &config,
        truth_data.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create output directory {}", parent.display())
            })?;
        }
    }

    evaluation::save_sweep_results(&sweep_results, &output_path)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    if let Some(best) = evaluation::find_best_parameters(&sweep_results, (0.5, 1.0, 0.3)) {
        println!(
            "Best candidate -> alpha {:.3}, beta {:.3}, epsilon {:.3}, lambda {:.1}, F1 {:.3}",
            best.parameters.alpha,
            best.parameters.beta,
            best.parameters.epsilon,
            best.parameters.lambda,
            best.quality_evaluation.f1_score
        );
    }

    println!(
        "Saved {} sweep entries to {}",
        sweep_results.len(),
        output_path.display()
    );

    Ok(())
}

fn parse_value_list(arg: Option<String>, label: &str) -> Result<Option<Vec<f32>>> {
    if let Some(raw) = arg {
        let mut values = Vec::new();
        for token in raw.split(',') {
            let trimmed = token.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: f32 = trimmed.parse().with_context(|| {
                format!(
                    "Invalid {} value '{}'. Expected comma-separated floats.",
                    label, trimmed
                )
            })?;
            values.push(value);
        }
        return Ok(if values.is_empty() { None } else { Some(values) });
    }
    Ok(None)
}

fn load_codebook_from_path(path: &PathBuf) -> Result<CodebookModel> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
        .unwrap_or_else(|| "".to_string());

    if extension == "json" {
        CodebookModel::load_from_json(path)
            .map_err(|e| anyhow::anyhow!("Failed to load JSON codebook: {}", e))
    } else {
        CodebookModel::load_from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load binary codebook: {}", e))
    }
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
