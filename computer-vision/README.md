# Computer vision

Classic image processing and deep learning for vision, applied end to end: from pixel operations to a document scanner that reads handwriting, and an object detector evaluated in depth.

| Notebook | Topics | Data |
|---|---|---|
| [01 · Image processing fundamentals](01_image_processing_fundamentals.ipynb) | global and local thresholding, morphology, connected components and region properties, watershed, denoising filters, Sobel and Canny edges | scikit-image sample images (coins, cameraman) |
| [02 · Document scanner](02_document_scanner.ipynb) | sheet segmentation, Hough transform, corner detection, homography rectification, digit segmentation, a CNN trained on MNIST, orientation from classifier confidence, failure analysis | 31 photos of handwritten sheets |
| [03 · License plate detection](03_license_plate_detection.ipynb) | YOLOv8 transfer learning, IoU matching, precision-recall curve and AP computed by hand, threshold choice, recall by object size, error analysis | Large License Plate Detection Dataset (Kaggle, CC0) |

## Highlights

- **24 coins out of 24** counted and measured with thresholds, morphology and connected components (notebook 01).
- **From a skewed photo to digits read correctly**: the scanner rectifies the sheet with a homography and reads every large digit on the successfully rectified sheets; it also detects and reports the photos it cannot handle (sheet outside the frame, glare) instead of returning a wrong page (notebook 02).
- **One CPU epoch of fine-tuning gives AP 0.86** on unseen photos, and the analysis by plate size shows exactly where the detector fails: plates only a few pixels wide (notebook 03).

## Files

- `scanner.py`: sheet segmentation, Hough border detection, corner selection and A4 rectification.
- `models/license_plate_yolov8s.pt`: YOLOv8-small fine-tuned on license plates (one epoch); `models/training_results.csv`: its training log.

MNIST is downloaded automatically. The license plate dataset (about 2.5 GB) is downloaded with `kagglehub` on first run of notebook 03.
