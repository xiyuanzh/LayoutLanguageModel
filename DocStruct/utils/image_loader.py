import cv2
import os

from PIL import Image

image_dirs = {'hospital': "data/raw/hospital_1000_20020401/image/",
              'funsd': "data/raw/funsd/image/"}


def load_image_parts_cv2(tag, name, rel_pos):
    path = os.path.join(image_dirs[tag], name + '.jpg')
    image = cv2.imread(path)
    Y, X = image.shape[:2]
    parts = []
    for p in rel_pos:
        x0, x1 = int(X*min(p[0::2])), int(X*max(p[0::2]))
        y0, y1 = int(Y*min(p[1::2])), int(Y*max(p[1::2]))
        part = image[y0:y1, x0:x1, :]
        parts.append(part)
    return parts


def load_image_parts_pil(tag, name, rel_pos, new_h):
    end = '.jpg' if tag == 'hospital' else '.png'
    path = os.path.join(image_dirs[tag], name + end)
    image = Image.open(path)
    X, Y = image.size
    parts = []
    for p in rel_pos:
        x0, x1 = int(X*min(p[0::2])), int(X*max(p[0::2]))
        y0, y1 = int(Y*min(p[1::2])), int(Y*max(p[1::2]))
        part = image.crop((x0, y0, x1, y1))

        if new_h != -1:
            width, height = part.size
            new_w = int(width / height * new_h)
            part = part.resize((new_w, new_h), Image.ANTIALIAS)

        parts.append(part)

        # part.show()

    return parts


def load_image_parts(name, rel_pos, new_h, mode='pil'):
    tag, name = name.split('-', 1)
    img = load_image_parts_pil(tag, name, rel_pos, new_h)
    return img

