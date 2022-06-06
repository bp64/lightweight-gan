from lightweight_gan.augmentations import AUGMENT_FNS


def DiffAugment(x, types=[]):
    for p in types:
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x.contiguous()
