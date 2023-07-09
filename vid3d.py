import torch
import ffmpeg
import argparse
import os
import subprocess as sp
import numpy as np
from models import DepthModel
from gen_stereo import process_image_element_wise, process_image_correct
import PIL.Image as Image
VIDEO_EXTS = ['.mp4', '.webm']
IMAGE_EXTS = ['.png', '.jpg', '.jpeg']
EXTENSIONS = tuple(VIDEO_EXTS + IMAGE_EXTS)
process_image = process_image_correct


def parseArguments():
    parser = argparse.ArgumentParser(description='vid3d.py')
    parser.add_argument('--output_dir', type=str, nargs='?', default='./output/',
                        help='Optional output directory path')
    parser.add_argument('-d', type=float, nargs='?', default=4.0,
                        help='Determines how strong the 3D effect is')
    parser.add_argument('--quality', type=str, default="medium",
                        choices=['lowest', 'low', 'medium', 'high', 'highest'])
    parser.add_argument('--sample_rate', type=float, nargs='?', default=30.0,
                        help='The FPS of the output video.')
    parser.add_argument('--save-depth', action="store_true",
                        help='replaces the right-eye view with the generated depth map')
    parser.add_argument('--reprocess', action='store_true',
                        help='Reprocess all input files, regardless of whether a stereo conversion is found in ./output')
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


def get_file_lists(directory):
    img_list, vid_list = [], []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(EXTENSIONS):
                filename, ext = os.path.splitext(file)
                found = False
                for extension in EXTENSIONS:
                    stereo_file = os.path.join(
                        './output', f'{filename}_stereo{extension}')
                    if os.path.exists(stereo_file):
                        if args.reprocess:
                            os.remove(stereo_file)
                        else:
                            print(f'Found {stereo_file}. Skipping')
                            found = True
                            break
                if not found:
                    if ext in IMAGE_EXTS:
                        img_list.append(file)
                    if ext in VIDEO_EXTS:
                        vid_list.append(file)

    return img_list, vid_list


if __name__ == '__main__':
    args = parseArguments()
    # Best quality / terrible speed: ZoeDepth_N
    # Good quality / decent speed: DPT_Swin_L_384
    # Decent quality / good speed: DPT_SwinV2_T_256
    # Bad quality / best speed: MiDaS_small
    model = DepthModel('ZoeDepth', '') if args.quality == 'highest' else \
        DepthModel('MiDaS', 'DPT_Swin_L_384') if args.quality == 'high' else \
        DepthModel('MiDaS', 'DPT_SwinV2_T_256') if args.quality == 'medium' else \
        DepthModel('MiDaS', 'MiDaS_small') if args.quality == 'low' else \
        DepthModel('MiDaS', 'DPT_LeViT_224')
    imgs, vids = get_file_lists('./input')
    print(f'images: {imgs}\nvideos: {vids}')

    print('Processing images now..')
    for file in imgs:
        print(f'\t{file}..')
        img_path = f'./input/{file}'
        filename, ext = os.path.splitext(file)
        image = Image.open(img_path)
        # ignore this for now. it's just extra processing that isn't related to my width-doubling woes.
        # Do some image processing here...
        depth_numpy = model.infer(image)
        if not args.save_depth:
            lr_frame = process_image(
                image, depth_numpy, args.d)
            output_filename = f'{filename}_stereo.png'
            output_file = f'./output/{output_filename}'
            Image.fromarray(lr_frame, "RGB").save(output_file)
        else:
            img_np = np.array(image)
            depth_numpy = (depth_numpy * 255).astype(np.uint8)
            depth = np.stack((depth_numpy,)*3, axis=-1)
            lr_frame = np.hstack((img_np, depth))
            output_filename = f'{filename}_stereo.png'
            output_file = f'./output/{output_filename}'
            Image.fromarray(lr_frame, "RGB").save(output_file)

    print('Processing videos now..')
    for file in vids:
        print(f'\t{file}..')
        video_path = f'./input/{file}'

        width, height, framerate = get_video_metadata(video_path)
        filename, ext = os.path.splitext(file)
        output_filename = f'{filename}_stereo.mp4'
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
                depth_numpy = model.infer(image)
                lr_frame = np.zeros_like((height, width * 2, 3))
                if not args.save_depth:
                    lr_frame = process_image(
                        image, depth_numpy, args.d)
                else:
                    img_np = np.array(image)
                    depth_numpy = (depth_numpy * 255).astype(np.uint8)
                    depth = np.stack((depth_numpy,)*3, axis=-1)
                    lr_frame = np.hstack((img_np, depth))
                # Write the processed frame to the output pipe
                pipe_out.stdin.write(lr_frame.astype(np.uint8).tobytes())
                # pipe_out.stdin.write(processed_image.astype(np.uint8).tobytes())
        except KeyboardInterrupt:
            pipe_out.stdin.flush()
            pipe_out.terminate()
            pipe_out.stdin.close()
            pipe_out.wait()
            pipe_in.stdout.flush()
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
