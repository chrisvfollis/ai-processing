# standard dependencies
pass

# 3rd-party dependencies
import numpy as np
import cv2

# internal dependencies
pass


def text_with_shadow(
    image: np.ndarray,
    text: str,
    xy_org: tuple[int, int],
    font,
    fontscale: int,
    color: tuple[int, int, int],
    thickness: int,
    shadow_color: tuple[int, int, int] = (0, 0, 0),
    offset: tuple[int, int] = (-2, 2),
):
    x, y = xy_org
    ox, oy = offset

    # shadow
    cv2.putText(
        image,
        text,
        (x + ox, y + oy),
        font,
        fontscale,
        shadow_color,
        thickness + 1,
        lineType=cv2.LINE_AA,
    )
    
    # foreground
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        fontscale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )

    return image
