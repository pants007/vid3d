# vid3d
Generate stereoscopic 3d videos with the power of AI!
vid3d is a simple commandline tool for batch processing regular flat video into stereoscopic 3d video.

It uses ZoeDepth (https://github.com/isl-org/ZoeDepth) to generate a depthmap, which is then used
to create an approximate right frame from a real left frame. The real left frame and the faux right frame are then combined 
into a single left-right stereoscopic 3d image.

# Usage
Simply put any videos you wish to convert into the `input` folder, wait for the program to finish, and
check the `output` folder for the converted videos.


# Dependencies
See `requirements.txt` for `pip` dependencies.