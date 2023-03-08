import imgaug.augmenters as iaa

####
# Defining our augmenters
####


sometimes = lambda aug: iaa.Sometimes(0.5, aug)
simple_aug = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
], random_order=True) # apply augmenters in random order
augmentation = iaa.SomeOf((0, 2), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
               iaa.Affine(rotate=180),
               iaa.Affine(rotate=270)],
             ),
    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
    iaa.GaussianBlur(sigma=(0.0, 5.0)),
    iaa.Grayscale(alpha=(0.0, 1.0))

])