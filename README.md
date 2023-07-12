[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# jaroo

This project increases signal to noise ratio in astronomical photos.

## Convert raw data to fits

In order to convert raw image from camera to fits
image, we use `dcraw` and `convert` commands.
To install these libraries in `macOS` use:

```bash
brew install dcraw
brew install imagemagick@6
echo 'export PATH="$PATH:/opt/homebrew/opt/imagemagick@6/bin"' >> ~/.zshrc
```

To install `dcraw` on `Ubuntu`:

```bash
sudo apt-get update -y
sudo apt-get install -y dcraw
sudo apt-get install -y netbpm
```

In order to convert raw photos to FITS,
put raw photos in directory `data/raw` and run

```bash
bash process_raw.sh
```

FITS photos will be written in directory `data/fits`
