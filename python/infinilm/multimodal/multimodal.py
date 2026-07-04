from typing import List, Union
from PIL import Image


def resolve_multimodal_inputs(messages: Union[List[dict], dict]):
    """Get images, videos, audios from the messages."""
    if isinstance(messages, dict):
        messages = [messages]

    images = []
    videos = []
    audios = []

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") == "text":
                pass
            elif item.get("type") == "image":
                # TODO support other image url formats
                images.append(Image.open(item["image_url"]))

            elif item.get("type") == "video":
                # Pass the source path through; the processor decodes, samples
                # frames, burns timestamps, and patchifies (see
                # Ernie4_5_VLMoeProcessor._decode_and_sample_frames).
                videos.append(item["video_url"])

            else:  # TODO support audio
                raise NotImplementedError("Only image and video inputs are supported")

    return images, videos, audios
