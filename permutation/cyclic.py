import math
import torch

if __name__ == '__main__':
    x = torch.arange(24).view((4, 6))
    print(x)

    y = torch.roll(x, 2)
    print(y)

    z = torch.roll(y, -2)
    print(z)
