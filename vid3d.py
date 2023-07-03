import torch
import ffmpeg
import argparse
import os
import subprocess as sp
import numpy as np
from gen_stereo import process_image


def parseArguments():
    parser = argparse.ArgumentParser(description='stereo.py')
    parser.add_argument('--output_dir', type=str, nargs='?', default='./output/',
                        help='Optional output directory path')
    parser.add_argument('--divergence', type=float, nargs='?', default=4.0,
                        help='Determines how strong the 3D effect is')
    parser.add_argument('--sample_rate', type=float, nargs='?', default=30.0,
                        help='The FPS of the output video.')
    args = parser.parse_args()
    return args


def get_video_metadata(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = video_stream = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = video_stream['width']
    height = video_stream['height']
    framerate = eval(video_stream['r_frame_rate'])
    return width, height, framerate


def load_model():
    repo = "isl-org/ZoeDepth"
    model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)
    zoe.eval()
    return zoe


def get_file_list(directory):
    file_list = []
    extensions = ('.mp4', '.webm')
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):  # Modify extensions as needed
                filename = file.split('.')[0]
                found = False
                for extension in extensions:
                    stereo_file = os.path.join(
                        './output', f'{filename}_stereo{extension}')
                    if os.path.exists(stereo_file):
                        print(f'Found {stereo_file}. Skipping')
                        found = True
                        break
                if not found:
                    file_list.append(file)

    return file_list


args = parseArguments()
model = load_model()
files = get_file_list('./input')

for file in files:
    video_path = f'./input/{file}'

    width, height, framerate = get_video_metadata(video_path)
    filename = os.path.splitext(file)[0]
    output_filename = filename + '_stereo.mp4'
    output_file = f'./output/{output_filename}'
    print(f'Beginning to process {output_file}..')
    command_in = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', video_path,
        '-vf', 'fps=30',
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo', '-'
    ]
    pipe_in = sp.Popen(command_in, stdout=sp.PIPE)

    # Start a subprocess that runs ffmpeg and takes its input from a pipe
    command_out = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-stats',
        '-y',  # overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', f'{width*2}x{height}',  # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '30',  # output frame rate
        '-i', '-',  # take input from stdin
        '-an',  # no audio
        '-vcodec', 'libx264',
        output_file
    ]
    pipe_out = sp.Popen(command_out, stdin=sp.PIPE)

    # Process video frame by frame
    try:
        while True:
            # Read one frame's worth of data from the input pipe
            raw_image = pipe_in.stdout.read(width * height * 3)
            if not raw_image:
                break

            # Convert the raw image data to a numpy array
            image = np.frombuffer(raw_image, dtype='uint8').reshape(
                [height, width, 3]).copy()
            # ignore this for now. it's just extra processing that isn't related to my width-doubling woes.
            # Do some image processing here...
            depth_numpy = model.infer_pil(image)
            lr_frame = process_image(
                image, depth_numpy, args.divergence)
            # Write the processed frame to the output pipe
            pipe_out.stdin.write(np.array(lr_frame).astype(np.uint8).tobytes())
            # pipe_out.stdin.write(processed_image.astype(np.uint8).tobytes())
    except KeyboardInterrupt:
        pipe_out.stdin.flush()
        pipe_out.terminate()
        pipe_out.stdin.close()
        pipe_out.wait()
        pipe_in.flush()
        pipe_in.terminate()
        pipe_in.stdout.close()
        pipe_in.wait()

    pipe_out.terminate()
    pipe_out.stdin.close()
    pipe_out.wait()
    pipe_in.terminate()
    pipe_in.stdout.close()
    pipe_in.wait()
    print('Done!')
