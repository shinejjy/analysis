import numpy as np
import cv2


def dct2d(image):
    M, N = image.shape
    dct = np.zeros((M, N))

    for u in range(M):
        for v in range(N):
            sum_val = 0
            for x in range(M):
                for y in range(N):
                    cu = np.sqrt(1 / M) if u == 0 else np.sqrt(2 / M)
                    cv = np.sqrt(1 / N) if v == 0 else np.sqrt(2 / N)
                    sum_val += cu * cv * image[x, y] * np.cos((x + 1 / 2) * u * np.pi / M) * np.cos(
                        (y + 1 / 2) * v * np.pi / N)
            dct[u, v] = sum_val

    return dct


def idct2d(dct):
    M, N = dct.shape
    signal = np.zeros((M, N))

    for x in range(M):
        for y in range(N):
            sum_val = 0
            for u in range(M):
                for v in range(N):
                    cu = np.sqrt(1 / M) if u == 0 else np.sqrt(2 / M)
                    cv = np.sqrt(1 / N) if v == 0 else np.sqrt(2 / N)
                    sum_val += cu * cv * dct[u, v] * np.cos((x + 1 / 2) * u * np.pi / M) * np.cos(
                        (y + 1 / 2) * v * np.pi / N)
                signal[x, y] = sum_val

    return signal


def mse(original, compressed):
    squared_diff = (original - compressed) ** 2
    mse_value = np.mean(squared_diff)
    return mse_value


def psnr(original, compressed):
    _mse = mse(original, compressed)
    max_pixel_value = 255.0
    psnr_value = 10 * np.log10((max_pixel_value ** 2) / _mse)
    return psnr_value


# ZigZag扫描
def zigzag_scan(matrix):
    rows, cols = matrix.shape
    result = np.zeros(rows * cols)

    # 初始化行和列索引
    row, col = 0, 0

    for i in range(rows * cols):
        result[i] = matrix[row, col]

        # 判断是往右上还是左下移动
        if (row + col) % 2 == 0:  # 往右上
            if col == cols - 1:
                row += 1
            elif row == 0:
                col += 1
            else:
                row -= 1
                col += 1
        else:  # 往左下
            if row == rows - 1:
                col += 1
            elif col == 0:
                row += 1
            else:
                row += 1
                col -= 1

    return result


def zigzag_reverse(arr, rows, cols):
    result = np.zeros((rows, cols))

    # 初始化行和列索引
    row, col = 0, 0

    for i in range(rows * cols):
        result[row, col] = arr[i]

        # 判断是往右上还是左下移动
        if (row + col) % 2 == 0:  # 往右上
            if col == cols - 1:
                row += 1
            elif row == 0:
                col += 1
            else:
                row -= 1
                col += 1
        else:  # 往左下
            if row == rows - 1:
                col += 1
            elif col == 0:
                row += 1
            else:
                row += 1
                col -= 1

    return result


# DPCM差分编码
def dpcm_encode(signal):
    diff_signal = np.zeros_like(signal)
    diff_signal[0] = signal[0]  # 第一个值直接复制
    for i in range(1, len(signal)):
        diff_signal[i] = signal[i] - signal[i - 1]
    return diff_signal


def dpcm_decode(diff_signal):
    signal = np.zeros_like(diff_signal)
    signal[0] = diff_signal[0]  # 第一个值直接复制
    for i in range(1, len(diff_signal)):
        signal[i] = signal[i - 1] + diff_signal[i]
    return signal


def generate_slant_mask(n, m=8):
    mask = np.zeros((m, m), dtype=int)

    for i in range(m):
        for j in range(m):
            if i + j < n:
                mask[i][j] = 1

    return mask, np.sum(mask) / (m * m)


if __name__ == '__main__':
    # 示例
    # image = np.random.rand(6, 6)
    #
    # dct_result = dct2d(image)
    # print("DCT Result:")
    # print(dct_result)
    #
    # print("DCT Result:")
    # print(cv2.dct(image))
    #
    # idct_result = idct2d(dct_result)
    # print("IDCT Result:")
    # print(idct_result)
    #
    # print("IDCT Result:")
    # print(cv2.idct(dct_result))

    matrix = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]])
    print("Original Signal:")
    print(matrix)

    zigzag_result = zigzag_scan(matrix)
    print("\nZigZag Scan Result:")
    print(zigzag_result)

    # 使用DPCM差分编码
    diff_encoded_signal = dpcm_encode(zigzag_result)
    print("\nDPCM Encoded Signal:")
    print(diff_encoded_signal)

    # 使用DPCM差分解码
    decoded_signal = dpcm_decode(diff_encoded_signal)
    print("\nDPCM Decoded Signal:")
    print(decoded_signal)

    # 可以再通过 zigzag_reverse 进行还原
    restored_matrix = zigzag_reverse(decoded_signal, 4, 4)
    print("\nRestored Matrix:")
    print(restored_matrix)
