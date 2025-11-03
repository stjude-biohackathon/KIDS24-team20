# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a biomedical image analysis pipeline for cell segmentation, tracking, and quantitative analysis. It processes multi-channel microscopy time-lapse data using deep learning models (Cellpose) and graph-based tracking (Trackastra) to study cell behavior over time.

## Core Technology Stack

- **Python 3** - Primary language
- **Cellpose** - Deep learning cell segmentation using pre-trained models (cyto, cyto2, cyto3, nuclei)
- **Trackastra** - Graph-based cell tracking across time frames
- **PyTorch** - GPU acceleration backend (CUDA recommended)
- **scikit-image, pandas, numpy** - Core data processing libraries
- **napari** - Interactive microscopy image visualization

## Development Commands

### Installation
No requirements.txt exists. Install dependencies manually:
```bash
pip install cellpose ipywidgets matplotlib torch scikit-image pandas alive_progress tifffile trackastra napari
```

### Running the Pipeline
```bash
# Command-line execution (currently not implemented - see Architecture Notes)
python3 SegmentAndTrack.py --config-file=demo_config.yaml

# Interactive execution (current working method)
python3 SegmentAndTrack.py
# Then enter data path when prompted

# Jupyter notebook execution
jupyter notebook cellpose_2D_TD.ipynb
```

### No Testing Framework
Currently no automated tests exist. Manual validation through execution logs and visual inspection.

## Architecture

### Three-Stage Sequential Pipeline
1. **Segmentation Stage** (`runCellpose()`): Multi-channel TIFF → labeled cell masks
2. **Tracking Stage** (`runTrackastra()`): Cell masks → tracked cells over time
3. **Analysis Stage** (`runAnalysis()`): Tracked cells → quantitative measurements

### Key Files Structure
- `SegmentAndTrack.py` - Main interactive entry point
- `segmentationTrackFunctions.py` - Core function library (current version)
- `segmentionTrackFunctions.py` - Legacy version (typo in name, missing parameters)
- `cellpose_2D_TD_headless.py` - Converted notebook for batch processing
- `demo_config.yaml` - Analysis configuration (not currently integrated with Python code)

### Data Flow
```
Input: Multi-channel TIFF stacks (time × height × width × channels)
├── Channel 0: Cytoplasm (segmentation target)
├── Channel 1: GFP signal (for cell identification)
└── Channel 2: Nuclei (optional, aids segmentation)

Output:
├── Cell tracking challenge format (edges.csv, spots.csv, tracks.csv)
├── Labeled mask images (.tif)
├── Cell statistics (.csv/.xlsx)
└── Execution logs (cellpose_2D_log.txt)
```

### Output Directory Structure
Each processed image creates:
```
{imgName}_segmentationTrackingResults/
├── {imgName}_cellposeMaskResults.tif
├── {imgName}_tracked_masks.tif
├── {imgName}_gfpMask.tif
├── {imgName}_dataframe.csv
└── ch1slices/ (temporary processing files)
```

## Configuration System

Uses YAML configuration files following this structure:
```yaml
segmentation_settings:
  run_cellpose: True/False
  cellpose_model: "cyto3"  # Options: cyto, cyto2, cyto3, nuclei
  cell_diameter_est: 140   # Pixels
  flow_threshold: 2.19     # Segmentation confidence
  cell_body_channel: 0     # Channel indices
  cell_nucleus_channel: 2

tracking_settings:
  run_trackastra: True/False
  trackastra_model: "greedy_nodiv"  # Options: greedy, greedy_nodiv, ilp
  max_track_distance: 50            # Max pixel movement between frames
  use_gfp_filter: True
```

## Architecture Notes & Current Limitations

### Critical Issues to Address
1. **Config Integration Missing**: `demo_config.yaml` exists but Python code doesn't load it
2. **Duplicate Functions**: Two versions of tracking functions exist - use `segmentationTrackFunctions.py` (current)
3. **Hard-coded Paths**: iLastik paths are hard-coded and need parameterization
4. **Interactive Input**: Main script prompts for paths, limiting automation
5. **Command-line Args**: `--config-file` parameter mentioned in README but not implemented

### Code Organization Patterns
- **Functional decomposition**: Core logic as modular functions with specific responsibilities
- **Explicit parameter passing**: Functions take many explicit parameters rather than using classes/objects
- **File system isolation**: Each image generates separate results directory
- **Procedural style**: Limited use of classes, primarily function-based flow

### GPU Considerations
- Code checks for CUDA availability with `torch.cuda.is_available()`
- GPU strongly recommended for Cellpose segmentation performance
- Large memory requirements for processing 4D biological image stacks

### Cell Analysis Metrics
- GFP positive identification (threshold > 110 intensity)
- Cell area measurements (converted to micrometers: pixels × 0.183²)
- Tracking duration (frames cell is successfully tracked)
- Morphological properties (area, ellipse axes, circularity, solidity)
- Movement analysis (displacement, speed from tracking data)

## Development Priorities

When working on this codebase:
1. **Implement config file loading** - Parse YAML and eliminate hard-coded parameters
2. **Add command-line argument parsing** - Make `--config-file` parameter functional
3. **Consolidate duplicate files** - Remove legacy `segmentionTrackFunctions.py`
4. **Create requirements.txt** - Pin dependency versions for reproducibility
5. **Add comprehensive logging** - Centralize logging across all pipeline stages
6. **Remove interactive prompts** - Enable full automation and batch processing