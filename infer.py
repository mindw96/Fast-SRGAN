from argparse import ArgumentParser
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import os
import time

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str, help='Directory where images are kept.')
parser.add_argument('--output_dir', type=str, help='Directory where to output high res images.')


def main():
    t1 = time.time()
    args = parser.parse_args()

    # Get all image paths
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir)]

    # Change model input shape to accept all size inputs
    model = keras.models.load_model('models/generator.h5', compile=False)
    inputs = keras.Input((None, None, 3))
    outputs = model(inputs)
    model = keras.models.Model(inputs, outputs)
    num = len(image_paths)
    count = 1
    j = 1
    # Loop over all images
    for image_path in image_paths:
        if count % 65 == 0:
            print('Model Reloading')
            model = keras.models.load_model('models/generator.h5', compile=False)
            inputs = keras.Input((None, None, 3))
            outputs = model(inputs)
            model = keras.models.Model(inputs, outputs)
            num = len(image_paths)

        start = time.time()

        # Read image
        low_res = cv2.imread(image_path, 1)

        # Convert to RGB (opencv uses BGR as default)
        low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

        # Resize
        low_res = cv2.resize(low_res, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Rescale to 0-1.
        low_res = low_res / 255.0

        # Get super resolution image
        sr = model.predict(np.expand_dims(low_res, axis=0))[0]

        # Rescale values in range 0-255
        sr = ((sr + 1) / 2.) * 255

        # Convert back to BGR for opencv
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

        # Save the results:
        inference_time = time.time() - start
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(image_path)), sr)
        print('Processing : %s' % image_path, "{:02d}/{:02d}".format(count, num),
              'FPS {}'.format(round(1.0 / inference_time)))
        count += 1
    t2 = time.time()
    total_time = t2 - t1
    print('Done!')
    print(t1, t2, total_time, 'Total FPS : {}'.format(round(1.0 / total_time)))


if __name__ == '__main__':
    main()
