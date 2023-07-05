# vid3d
Generate stereoscopic 3d videos/images with the power of AI!
vid3d is a simple commandline tool for batch processing regular "flat" media into stereoscopic 3d media.

It uses ZoeDepth (https://github.com/isl-org/ZoeDepth) to generate a depthmap, which is then used
to create an approximate right frame from a real left frame. The real left frame and the faux right frame are then combined 
into a single left-right stereoscopic 3d image.

# Usage
Simply put any videos or images you wish to convert into the `input` folder, then run `./vid3d.sh` or `python3 vid3d.py`, and
check the `output` folder for the converted videos after the program has finished.


# Dependencies
See `requirements.txt` for `pip` dependencies. In addition to those, you will need CUDA installed (have only tested with version 12.1) along with ffmpeg.
