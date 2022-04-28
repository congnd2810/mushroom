import os
epoch = 0
if not os.path.exists(f'checkpoints/vgg/{epoch + 1}'):
    os.mkdir(f'checkpoints/vgg/{epoch + 1}')