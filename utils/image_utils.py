def get_bbox(img, margin=0):
    mask = (img == 255)[..., :-1].all(axis=-1)

    # If multiple images, need to further compute the common mask
    if img.ndim == 4:
        mask = mask.all(axis=0)

    row_mask = mask.all(axis=0)
    min_x = row_mask.argmin() - 1 - margin
    max_x = mask.shape[1] - row_mask[..., ::-1].argmin() + margin
    col_mask = mask.all(axis=1)
    min_y = col_mask.argmin() - 1 - margin
    max_y = mask.shape[0] - col_mask[..., ::-1].argmin() + margin

    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(mask.shape[1] - 1, max_x)
    max_y = min(mask.shape[0] - 1, max_y)

    return min_x, min_y, max_x, max_y


# def get_common_bbox(imgs, margin=0):
#
#     mask = (imgs == 255)[..., :-1].all(axis=-1).all(axis=0)
#     row_mask = mask.all(axis=0)
#     min_x = row_mask.argmin() - 1 - margin
#     max_x = mask.shape[1] - row_mask[..., ::-1].argmin() + margin
#     col_mask = mask.all(axis=1)
#     min_y = col_mask.argmin() - 1 - margin
#     max_y = mask.shape[0] - col_mask[..., ::-1].argmin() + margin
#
#     min_x = max(0, min_x)
#     min_y = max(0, min_y)
#     max_x = min(mask.shape[1] - 1, max_x)
#     max_y = min(mask.shape[0] - 1, max_y)
#
#     return min_x, min_y, max_x, max_y


def crop_img(img, min_x, min_y, max_x, max_y):
    return img[min_y:max_y, min_x:max_x]