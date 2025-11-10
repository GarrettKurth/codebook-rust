# codebook-rust

Rust implmentation of the Codebook algorithm for foreground-background segmentation in images and videos as proposed by Kim et al. in [Real-Time Foreground–Background Segmentation Using Codebook Model](https://ieeexplore.ieee.org/document/1643681).

## Features
- Foreground-background segmentation using the Codebook algorithm
- Support for image and video input
- Configurable parameters for fine-tuning the segmentation process
- Exportable codebooks in JSON or binary formats

## Installation
This is a CLI tool, to build and install it, make sure you have Rust and Cargo installed, then run:

```bash
cargo build --release
```

