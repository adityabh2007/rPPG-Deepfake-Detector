# rPPG Deepfake Detector

This beginner-friendly project looks for a pulse in a face video. Each heartbeat causes tiny changes in skin color that a camera can record. The program measures average red, green, and blue values from the forehead and cheeks, then uses the POS method to emphasize pulse-like changes.

It summarizes those signals and trains a small machine-learning model to distinguish the real and fake videos in your own dataset. A prediction is an estimate, not proof: it depends on the quality and variety of the videos used for training.

## Install

Use Python 3.10 or newer. From this project folder, install the listed libraries:

```powershell
python -m pip install -r requirements.txt
```

For the notebook walkthrough, open `notebooks/explore.ipynb` in a Python notebook environment after installing these packages.

## Organize the videos

Put face videos in the matching folders. The program searches inside each folder, including subfolders, and supports MP4, AVI, MOV, MKV, WebM, MPEG, and MPG files.

```text
data/
├── real/
│   ├── real_clip_01.mp4
│   └── more_real_clips/
└── fake/
    ├── fake_clip_01.mp4
    └── more_fake_clips/
```

Use clear, mostly frontal faces that remain visible for at least 10 seconds. A small starting dataset might have 20–50 videos in each class. FaceForensics++ and Celeb-DF are examples of research datasets; obtain them from their official sources and follow each dataset's license, access rules, and terms of use. Do not redistribute videos unless the terms allow it.

## Run the program

Run these commands from `rppg-deepfake-detection`:

```powershell
python main.py train
python main.py predict path\to\video.mp4
python main.py plot path\to\video.mp4
```

Training extracts and caches features in `results/features.npz`, evaluates a scaled RBF-SVM with stratified 5-fold cross-validation, compares logistic regression and random forest, then saves the final model as `results/model.joblib`. It also saves the ROC curve and confusion matrix in `results/`. At least five usable videos per class are needed for 5-fold evaluation.

Prediction prints the estimated label and its model confidence. If the median SNR from the three face regions is below 3 dB, it prints `uncertain: signal too weak` instead of forcing a label. The `plot` command saves the filtered pulse and its power spectrum for each region to `results/`.

Run the signal-processing self-tests from the project folder:

```powershell
python src\filters.py
python src\features.py
```

Each test creates a noisy 1.2 Hz pulse at 30 frames per second and checks that the estimated heart rate is close to 72 BPM.

## Limitations

- Video compression can weaken or remove the tiny color changes used to estimate a pulse.
- A good deepfake may reproduce or copy a pulse signal.
- Lighting, camera motion, head movement, and skin tone can change signal quality.
- Small datasets can overfit; cross-validation does not replace a separate test set.
- The SNR warning is only a rough signal-quality check and cannot guarantee a reliable prediction.

## Run the program

Run these commands from `rppg-deepfake-detection`:

```powershell
python main.py train
python main.py predict path\to\video.mp4
python main.py plot path\to\video.mp4
```

Training extracts and caches features in `results/features.npz`, evaluates a scaled RBF-SVM with stratified 5-fold cross-validation, compares logistic regression and random forest, then saves the final model as `results/model.joblib`. It also saves the ROC curve and confusion matrix in `results/`. At least five usable videos per class are needed for 5-fold evaluation.

Prediction prints the estimated label and its model confidence. If the median SNR from the three face regions is below 3 dB, it prints `uncertain: signal too weak` instead of forcing a label. The `plot` command saves the filtered pulse and its power spectrum for each region to `results/`.

Run the signal-processing self-tests from the project folder:

```powershell
python src\filters.py
python src\features.py
```

Each test creates a noisy 1.2 Hz pulse at 30 frames per second and checks that the estimated heart rate is close to 72 BPM.

## Limitations

- Video compression can weaken or remove the tiny color changes used to estimate a pulse.
- A good deepfake may reproduce or copy a pulse signal.
- Lighting, camera motion, head movement, and skin tone can change signal quality.
- Small datasets can overfit; cross-validation does not replace a separate test set.
- The SNR warning is only a rough signal-quality check and cannot guarantee a reliable prediction.
