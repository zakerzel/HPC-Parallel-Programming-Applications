# Exercise 2 – Parallel Cell Image Processing and Morphological Characterization

## Goal
Build a serial and parallel image-processing pipeline over the DIC-C2DH-HeLa dataset.

## Minimum outputs per object
- Bounding box
- Area
- Major axis length
- Minor axis length

## Suggested experiments
- Compare serial and multiprocessing execution
- Test different numbers of workers
- Save one or more annotated example images
- Export a summary table per image

## Folder notes
- `serial/`: serial pipeline
- `parallel_mp/`: multiprocessing pipeline
- `data/`: dataset or download instructions
- `outputs/`: CSV summaries, timing tables, figures
