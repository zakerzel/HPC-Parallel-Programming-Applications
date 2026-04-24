# Exercise 2 - Parallel Cell Image Processing and Morphological Characterization

This exercise implements a serial and multiprocessing pipeline for microscopy image processing. The code expects grayscale images and computes connected-component measurements after segmentation.

## Files

- `src/inspect_dataset.py`: inspects image sizes and file formats
- `src/process_images.py`: serial and parallel object extraction
- `src/visualize_measurements.py`: summary tables and plots

## Run

```bash
python src/inspect_dataset.py --input data/DIC-C2DH-HeLa/01
python src/process_images.py --input data/DIC-C2DH-HeLa/01 --mode serial
python src/process_images.py --input data/DIC-C2DH-HeLa/01 --mode parallel --workers 4
python src/visualize_measurements.py
```
