import sys
import numpy as np
def int_add2(a, b):
    if a == -b:  # 当a = -b时 该方法效率不高，直接返回0
        return 0
    else:
        while b:
            a, b = a ^ b, (a & b) << 1
        return a


if __name__ == '__main__':
    print(int_add2(22,63))

