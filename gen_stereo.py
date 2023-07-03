import cupy as cp
import numpy as np
import cupyx.jit
'''
    CUDA-based stereo-image generation

    Heavily inspired by the naive method found in:

    https://github.com/thygate/stable-diffusion-webui-depthmap-script/blob/main/scripts/stereoimage_generation.py
'''


@cupyx.jit.rawkernel()
def fix_right(right_img, deviation, output_img):
    h, w, c = right_img.shape
    idy, idx = cupyx.jit.grid(2)
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


@cupyx.jit.rawkernel()
def generate_stereo_row_wise(left_img, depth, deviation, right_img):
    h, w, c = left_img.shape
    row = cupyx.jit.grid(1)
    if row < h:
        for col in range(w):
            col_r = col - int((1 - depth[row, col] ** 2) * deviation)
            if 0 <= col_r and col_r < w:
                for channel in range(c):
                    right_img[row, col_r,
                              channel] = left_img[row, col, channel]


def process_image(left_img, depth, deviation):
    img_cp = cp.array(left_img)
    depth_cp = cp.array(depth)
    depth_min = depth_cp.min()
    depth_max = depth_cp.max()
    depth_cp = (depth_cp - depth_min) / (depth_max - depth_min)
    h, w, c = img_cp.shape
    deviation = (deviation / 100) * img_cp.shape[1]
    right_cp = cp.zeros_like(img_cp, dtype=img_cp.dtype)

    block_size = 512
    grid_size = (h + block_size - 1) // block_size
    generate_stereo_row_wise[grid_size, block_size](
        img_cp, depth_cp, deviation, right_cp)

    threadsperblock = (16, 16)
    blockspergrid_x = np.ceil(h / threadsperblock[0]).astype(np.int32)
    blockspergrid_y = np.ceil(w / threadsperblock[1]).astype(np.int32)
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    output_img = cp.zeros_like(img_cp, dtype=img_cp.dtype)
    fix_right[blockspergrid, threadsperblock](right_cp, deviation, output_img)

    return cp.asnumpy(cp.hstack([img_cp, output_img]))
