# Local Super-Resolution Data

SRCNN's data (Dong et al., 2014), as BMP files:

- `train/`: the 91-image training set (Yang et al., 2010)
- `set5/`: Set5 (Bevilacqua et al., 2012), the 5 standard test images
- `set14/`: Set14 (Zeyde et al., 2010), 14 test images

```bash
git clone --depth 1 --filter=blob:none https://github.com/tegg89/SRCNN-Tensorflow.git /tmp/srcnn
mkdir -p data/sr/train data/sr/set5 data/sr/set14
cp /tmp/srcnn/Train/*.bmp data/sr/train/
cp /tmp/srcnn/Test/Set5/*.bmp data/sr/set5/
cp /tmp/srcnn/Test/Set14/*.bmp data/sr/set14/
```

Check: `python3 sr.py --bicubic-only` prints set5 30.39 dB and set14 27.54 dB.
