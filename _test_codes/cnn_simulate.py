from cv2 import cv2
import numpy as np

# src = cv2.imread("lenna.png", cv2.IMREAD_COLOR)
src = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)
channels = cv2.split(src)

def convolution(src, filter_arr, add_val=0):
    if len(src.shape) == 2:
        channel = 1
    else:
        channel = src.shape[2]
    dst = np.zeros(src.shape, dtype=np.uint8)
    for row_idx in range(src.shape[0]):
        if row_idx+filter_arr.shape[0] < src.shape[0]:
            # print("{}번째 행".format(row_idx))
            for col_idx in range(src.shape[1]):
                if col_idx+filter_arr.shape[1] < src.shape[1]:
                    check_area = src[row_idx:row_idx+filter_arr.shape[0], col_idx:col_idx+filter_arr.shape[1]]
                    # print(check_area.shape)
                    filtered_result = np.zeros((channel), dtype=np.float32)
                    for this_row_idx, row in enumerate(check_area):
                        for this_col_idx, _ in enumerate(row):
                            filtered_result += check_area[this_row_idx][this_col_idx] * filter_arr[this_row_idx][this_col_idx]
                    filtered_result.astype(np.uint8)
                    # filtered_result = filtered_result + add_val
                    if filtered_result < 0: #relu와 같은 효과
                        filtered_result = 0
                    if filtered_result > 255:
                        filtered_result = 255

                    dst[row_idx][col_idx] = filtered_result
    return dst

def maxPooling(src, pooling_kernel_size=2):
    if len(src.shape) == 2:
        channel = 1
        dst_size = (int(src.shape[0]/pooling_kernel_size), int(src.shape[1]/pooling_kernel_size))
    else:
        channel = src.shape[2]
        dst_size = (int(src.shape[0]/pooling_kernel_size), int(src.shape[1]/pooling_kernel_size), channel)
    dst = np.zeros(dst_size, dtype=np.uint8)
    dst_h_cnt = 0
    for row_idx in range(src.shape[0]):
        if row_idx % pooling_kernel_size == 0 and row_idx+pooling_kernel_size < src.shape[0]:
            # print("{}번째 행".format(row_idx))
            dst_w_cnt = 0
            for col_idx in range(src.shape[1]):
                if col_idx % pooling_kernel_size == 0 and col_idx+pooling_kernel_size < src.shape[1]:
                    check_area = src[row_idx:row_idx+pooling_kernel_size, col_idx:col_idx+pooling_kernel_size]
                    this_max_val = 0
                    for this_row_idx, row in enumerate(check_area):
                        for this_col_idx, _ in enumerate(row):
                            if check_area[this_row_idx][this_col_idx] > this_max_val:
                                this_max_val = check_area[this_row_idx][this_col_idx]
                    dst[dst_h_cnt][dst_w_cnt] = this_max_val
                    dst_w_cnt += 1
            dst_h_cnt += 1
    return dst

# filter_arr = np.array(
#     [[1/9, 1/9, 1/9],
#      [1/9, 1/9, 1/9],
#      [1/9, 1/9, 1/9]],
#     dtype=np.float32
# )
filter_arr = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]],
    dtype=np.float32
)

#step1 : 합성곱 수행
if len(channels) > 1:
    bgr_list = ["blue", "green", "red"]
else:
    bgr_list = ["gray"]
for idx, c in enumerate(channels):
    dst = convolution(c, filter_arr, add_val=0)
    print("dst_{}_size : {}".format(bgr_list[idx], dst.shape))
    cv2.imshow("dst_{}".format(bgr_list[idx]), dst)
    #step2 : Max Pooling
    dst = maxPooling(dst, pooling_kernel_size=32)
    # dst = maxPooling(dst, pooling_kernel_size=2)
    # dst = maxPooling(dst, pooling_kernel_size=2)
    print("dst_{}_max_pooling_size : {}".format(bgr_list[idx], dst.shape))
    cv2.namedWindow("dst_{}_max_pooling".format(bgr_list[idx]), cv2.WINDOW_KEEPRATIO)
    cv2.imshow("dst_{}_max_pooling".format(bgr_list[idx]), dst)

cv2.waitKey(0)
cv2.destroyAllWindows()