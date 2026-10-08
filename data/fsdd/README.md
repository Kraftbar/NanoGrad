# Local Free Spoken Digit Dataset

FSDD (Jakobovski et al.): 3,000 recordings of the spoken digits 0-9, 6
speakers x 10 digits x 50 recordings, 8 kHz mono 16-bit WAV, named
`{digit}_{speaker}_{index}.wav`. Recordings 0-4 of each speaker and digit
are the test set by the dataset's convention.

```bash
git clone --depth 1 https://github.com/Jakobovski/free-spoken-digit-dataset.git /tmp/fsdd
mkdir -p data/fsdd && cp -r /tmp/fsdd/recordings data/fsdd/
```
