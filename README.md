## Directions to Run Synnet (Expert Lipsync Discriminator)

### Training

**Scenarios and How to Run:**

  1. No preprocessed data, no checkpoint:

  * `python run_syncnet.py --mode train --data-dir <PATH TO DATA DIRECTORY (containing .mp4 files)>`
  
  * Optionally add `--load-limit <data point limit (integer)>` to stop processing data after a certain minimum number of data points have already been collected by preprocessing.

  2. No processed data to train model with, pick up training from checkpoint:

  * `python run_syncnet.py --mode train --data-dir <PATH TO DATA DIRECTORY (containing .mp4 files)> --load-from <path to saved model (.h5 file)>`

  * Optionally add `--load-limit <data point limit (integer)>` to stop processing data after a certain minimum number of data points have already been collected by preprocessing.
  
  3. Data preprocessed (.npy files exist in data directory), no checkpoint:

  * `python run_syncnet.py --mode train`

  4. Data preprocessed (.npy files exist in data directory), pick up training from checkpoint:

  * `python run_syncnet.py --mode train  --load-from <path to saved model (.h5 file)>`

### Testing
* `python run_syncnet.py --mode test --load-from <path to saved model (.h5 file)>`
