import math
import numpy as np
import torch


def dilate(x: torch.Tensor, dilation_factor: int,
           pad_value: float = 0.) -> torch.Tensor:

    if dilation_factor == 1:
        return x

    n, c, l = x.shape

    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)

    x = torch.permute(x, (1, 2, 0))  # (n, c, l) --> (c, l, n)
    new_l = math.ceil(l / dilation_factor)
    new_n = math.ceil(n * dilation_factor)
    x = torch.reshape(x, (c, new_l, new_n))

    x = torch.permute(x, (2, 0, 1))  # (c, l, n) --> (n, c, l)
    return x


if __name__ == '__main__':
    x = torch.arange(24).view(2, 3, 4)
    print(x)

    y = dilate(x, 2)
    print(y)

    z = dilate(y, 1 / 2)
    print(z)
