import OpenEXR as exr
import Imath
import array
import logging
from pathlib import Path
import numpy as np
from typing import Literal


def write_exr(
    filepath: Path,
    channels: dict[str, np.ndarray],
):
    """write dict of Channels to a multilayer EXR file. Only np arrays with np.float16, np.float32 or np.uint32 are supported"""
    if len(channels) == 0:
        raise ValueError("No channels to write")

    img = list(channels.values())[0]
    header = exr.Header(
        img.shape[1],
        img.shape[0],
    )
    header["compression"] = Imath.Compression(Imath.Compression.NO_COMPRESSION)

    if img.dtype == np.float32:
        exr_type = Imath.PixelType(Imath.PixelType.FLOAT)
    elif img.dtype == np.uint32:
        exr_type = Imath.PixelType(Imath.PixelType.UINT)
    elif img.dtype == np.float16:
        exr_type = Imath.PixelType(Imath.PixelType.HALF)
    else:
        raise ValueError(f"Unsupported dtype do write to EXR: {img.dtype}")

    header["channels"] = dict({k: Imath.Channel(exr_type) for k in channels.keys()})

    logging.debug(f"Writing {filepath} with header: {header}")

    exr_file = exr.OutputFile(str(filepath), header)
    try:
        exr_file.writePixels({k: img.tobytes() for k, img in channels.items()})
    finally:
        exr_file.close()
