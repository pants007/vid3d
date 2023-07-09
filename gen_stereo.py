import numpy as np
from numba import cuda
'''
    CUDA-based stereo-image generation

    Heavily inspired by the naive method found in:

    https://github.com/thygate/stable-diffusion-webui-depthmap-script/blob/main/scripts/stereoimage_generation.py
'''


@cuda.jit
def fix_holes(right_img, deviation, output_img):
    h, w, c = right_img.shape
    idy, idx = cuda.grid(2)
    if idy < h and idx < w:
        r = right_img[idy, idx, 0]
        g = right_img[idy, idx, 1]
        b = right_img[idy, idx, 2]
        # check if pixel is black after right-eye projection (ie was not copied to)
        if (r + g + b) == 0:
            l_idx = -1
            r_idx = -1
            offset = 0
            l_r, l_g, l_b = 0, 0, 0
            r_r, r_g, r_b = 0, 0, 0
            # find nearest pixel left of idx that is colored
            while l_idx == -1:
                offset -= 1
                offset_idx = idx + offset
                if offset_idx > 0:
                    l_r = right_img[idy, offset_idx, 0]
                    l_g = right_img[idy, offset_idx, 1]
                    l_b = right_img[idy, offset_idx, 2]
                    if (l_r + l_g + l_b) != 0:
                        l_idx = offset_idx
                else:
                    l_idx = 0
            offset = 0
            # find nearest pixel right of idx that is colored
            while r_idx == -1:
                offset += 1
                offset_idx = idx + offset
                if offset_idx < w:
                    r_r = right_img[idy, offset_idx, 0]
                    r_g = right_img[idy, offset_idx, 1]
                    r_b = right_img[idy, offset_idx, 2]
                    if (r_r + r_g + r_b) != 0:
                        r_idx = offset_idx
                else:
                    r_idx = w
            # if the search reached the right-hand edge of the image, only use left pixel
            if r_idx == w:
                r = l_r
                g = l_g
                b = l_b
            # if the search reached the left-hand edge of the image, only use right pixel
            elif l_idx == 0:
                r = r_r
                g = r_g
                b = r_b
            # interpolate between the two found neighbors
            else:
                length = r_idx - l_idx
                l_weight = (idx - l_idx) / length
                r_weight = (r_idx - idx) / length
                r = int(r_r * l_weight + l_r * r_weight)
                g = int(r_g * l_weight + l_g * r_weight)
                b = int(r_b * l_weight + l_b * r_weight)
        output_img[idy, idx, 0] = r
        output_img[idy, idx, 1] = g
        output_img[idy, idx, 2] = b


def run_fix_holes(img, right_cu, deviation):
    h, w, c = img.shape
    output_img = np.zeros_like(img)
    output_img_cu = cuda.to_device(output_img)
    threadsperblock = (16, 16)
    blockspergrid_x = np.ceil(h / threadsperblock[0]).astype(np.int32)
    blockspergrid_y = np.ceil(w / threadsperblock[1]).astype(np.int32)
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    fix_holes[blockspergrid, threadsperblock](
        right_cu, deviation, output_img_cu)
    return output_img_cu


@cuda.jit
def shift_pass_row_wise(left_img, depth, deviation, right_img):
    h, w, c = left_img.shape
    row = cuda.grid(1)
    if row < h:
        for col in range(w):
            col_r = col - int((1 - depth[row, col] ** 2) * deviation)
            if 0 <= col_r and col_r < w:
                for channel in range(c):
                    right_img[row, col_r,
                              channel] = left_img[row, col, channel]


def run_shift_pass_row_wise(left_img, depth, deviation):
    left_img = np.array(left_img)
    h, w, c = left_img.shape
    img_cu = cuda.to_device(left_img)
    depth_cu = cuda.to_device(depth)
    h, w, c = left_img.shape
    deviation = (deviation / 100) * left_img.shape[1]
    right_cu = cuda.to_device(np.zeros_like(left_img, dtype=left_img.dtype))
    block_size = 1
    grid_size = h
    shift_pass_row_wise[grid_size, block_size](
        img_cu, depth_cu, deviation, right_cu)
    return right_cu


@cuda.jit
def shift_pass_element_wise(left_img, depth, deviation, right_img, locks, max_depth):
    h, w, c = left_img.shape
    row, col = cuda.grid(2)
    if row < h and col < w:
        depth_val = depth[row, col]
        d = (1 - depth_val ** 2) * deviation
        col_r = col - int(d)
        if 0 <= col_r and col_r < w:
            lock = 0
            while lock != 1:
                lock = cuda.atomic.compare_and_swap(locks[row, col_r], 1, 0)
            if max_depth[row, col_r] < d:
                for channel in range(c):
                    right_img[row, col_r,
                              channel] = left_img[row, col, channel]
                max_depth[row, col_r] = d
            cuda.atomic.compare_and_swap(locks[row, col_r], 0, 1)


def run_shift_pass_element_wise(left_img, depth, deviation):
    img_cu = cuda.to_device(left_img)
    depth_cu = cuda.to_device(depth)
    h, w, c = left_img.shape
    deviation = (deviation / 100) * left_img.shape[1]
    right_cu = cuda.to_device(np.zeros_like(left_img, dtype=left_img.dtype))
    locks = np.ones(depth.shape)[:, :, np.newaxis].astype(np.int32)
    locks_cu = cuda.to_device(locks)
    max_depth = np.zeros_like(depth, dtype=depth.dtype)
    max_depth_cu = cuda.to_device(max_depth)

    threadsperblock = (16, 16)
    blockspergrid_x = np.ceil(h / threadsperblock[0]).astype(np.int32)
    blockspergrid_y = np.ceil(w / threadsperblock[1]).astype(np.int32)
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    shift_pass_element_wise[blockspergrid, threadsperblock](
        img_cu, depth_cu, deviation, right_cu, locks_cu, max_depth_cu)
    return right_cu


def process_image_correct(left_img, depth, deviation):
    left_img = np.array(left_img)
    depth = np.array(depth)
    shifted_row = run_shift_pass_row_wise(left_img, depth, deviation)
    shifted_row_fixed = run_fix_holes(left_img, shifted_row, deviation)
    output_img1 = shifted_row_fixed.copy_to_host()
    return np.hstack([output_img1, left_img])


def process_image_element_wise(left_img, depth, deviation):
    left_img = np.array(left_img)
    depth = np.array(depth)
    shifted_element = run_shift_pass_element_wise(left_img, depth, deviation)
    shifted_element_fixed = run_fix_holes(left_img, shifted_element, deviation)
    output_img2 = shifted_element_fixed.copy_to_host()
    return np.hstack([output_img2, left_img])
