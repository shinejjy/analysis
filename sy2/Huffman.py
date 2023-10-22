import heapq
import collections
import cv2
from matplotlib import pyplot as plt


class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(frequencies):
    heap = [HuffmanNode(sym, freq) for sym, freq in frequencies.items() if freq > 0]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        heapq.heappush(heap, parent)
    return heap[0]


def build_huffman_codes(root, prefix, code_dict):
    if root.symbol is not None:
        code_dict[root.symbol] = prefix
    if root.left:
        build_huffman_codes(root.left, prefix + '0', code_dict)
    if root.right:
        build_huffman_codes(root.right, prefix + '1', code_dict)


def main():
    # 读取灰度图像
    image = cv2.imread('lina.jpg', cv2.IMREAD_GRAYSCALE)
    height, width = image.shape

    # 统计灰度级别频率
    frequencies = collections.Counter(image.ravel())

    # 画出频率图
    gray_levels = list(frequencies.keys())
    counts = list(frequencies.values())

    plt.bar(gray_levels, counts)
    plt.xlabel('Gray Level')
    plt.ylabel('Frequency')
    plt.title('Gray Level Frequency Distribution')
    plt.show()

    # 构建哈夫曼树
    huffman_root = build_huffman_tree(frequencies)

    # 构建哈夫曼编码
    huffman_codes = {}
    build_huffman_codes(huffman_root, '', huffman_codes)

    huffman_codes = dict(sorted(huffman_codes.items(), key=lambda x: len(x[1])))

    # Calculate Huffman code lengths
    code_lengths = [len(huffman_codes.get(pixel, '')) for pixel in range(256)]

    # Plot the histogram
    plt.bar(range(256), code_lengths)
    plt.xlabel('Pixel Value')
    plt.ylabel('Code Length')
    plt.title('Huffman Code Lengths for Each Pixel Value')
    plt.show()

    # 压缩图像并计算压缩比
    encoded_image = ''.join(huffman_codes[pixel] for pixel in image.ravel())
    original_size = height * width * 8  # 8 bits per pixel
    compressed_size = len(encoded_image)
    compression_ratio = original_size / compressed_size

    print(f"Original size: {original_size / 8} Byte")
    print(f"Compressed size: {compressed_size / 8} Byte")
    print(f"Compression ratio: {compression_ratio:.2f}")


if __name__ == '__main__':
    main()
