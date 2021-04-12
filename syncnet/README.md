### Requirements

`brew install ffmpeg`

### Data

Unzip the current converted.zip file into `syncnet/data`.

If saved_data does not contain the tf.data.Datasets for the audio and
video, then preprocess will generate these datasets using the data in that converted/ file from
the zip. Otherwise, it will load the preprocessed datasets.