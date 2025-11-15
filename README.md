# Qwen Image Generator - RunPod Serverless

[![RunPod](https://img.shields.io/badge/RunPod-Serverless-blueviolet)](https://runpod.io)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

High-quality image generation using Qwen-Image model with ComfyUI backend, deployed as a serverless API on RunPod.

## Features

- **Fast Generation**: 3-5 seconds per image (warm start)
- **High Quality**: Qwen-Image with Lightning LoRA (8-step)
- **Flexible Resolutions**: Portrait, Landscape, Square presets
- **Auto-Scaling**: Scale to zero when idle, scale up on demand
- **Cost Effective**: ~$0.0003 per image

## API Usage

### Endpoint
```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
```

### Request Format
```json
{
  "input": {
    "prompt": "a beautiful sunset over mountains",
    "negative_prompt": "blurry, low quality",
    "width": 1440,
    "height": 1920,
    "seed": -1,
    "steps": 25,
    "cfg": 1.0
  }
}
```

### Response Format
```json
{
  "id": "request-id",
  "status": "COMPLETED",
  "output": {
    "image": "base64-encoded-image-data",
    "seed": 123456,
    "width": 1440,
    "height": 1920,
    "prompt": "a beautiful sunset over mountains"
  }
}
```

## Quick Start

### Python Example
```python
import requests
import base64
from PIL import Image
from io import BytesIO

def generate_image(prompt, endpoint_id, api_key):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "prompt": prompt,
            "width": 1440,
            "height": 1920,
            "seed": -1
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    result = response.json()

    # Decode image
    img_data = base64.b64decode(result["output"]["image"])
    img = Image.open(BytesIO(img_data))
    return img

# Usage
img = generate_image(
    "a cyberpunk city at night",
    "your-endpoint-id",
    "your-api-key"
)
img.save("generated.png")
```

### Testing
```bash
python test_api.py \
  --endpoint YOUR_ENDPOINT_ID \
  --api-key YOUR_API_KEY \
  --prompt "a beautiful landscape"
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of desired image |
| `negative_prompt` | string | (default) | What to avoid in the image |
| `width` | integer | 1440 | Image width (multiple of 64) |
| `height` | integer | 1920 | Image height (multiple of 64) |
| `seed` | integer | -1 | Random seed (-1 for random) |
| `steps` | integer | 25 | Generation steps (Lightning optimized) |
| `cfg` | float | 1.0 | CFG scale |

## Preset Resolutions

- **Portrait**: 1440x1920 (3:4 ratio)
- **Landscape**: 1920x1440 (4:3 ratio)
- **Square**: 1328x1328 (1:1 ratio)

## Performance

| Metric | Value |
|--------|-------|
| Cold Start | 45-60 seconds |
| Warm Generation | 3-5 seconds |
| Cost per Image | ~$0.0003 |
| GPU Required | RTX 5090 or equivalent |

## Deployment

This project is configured for automatic deployment via RunPod Hub:

1. Push to GitHub
2. Create a release (e.g., `v1.0.0`)
3. RunPod automatically builds and deploys
4. Endpoint ready to use!

## Files

- `Dockerfile` - Container definition with all dependencies
- `handler.py` - RunPod serverless handler
- `qwen_sfw_workflow_api.json` - ComfyUI workflow configuration
- `.runpod/hub.json` - RunPod Hub configuration
- `.runpod/tests.json` - Automated tests
- `test_api.py` - API testing script

## Models Included

- **Qwen Image BF16** - Main diffusion model
- **Qwen 2.5 VL 7B** - Text encoder
- **Qwen Image VAE** - Image decoder
- **Lightning LoRA** - 8-step acceleration

## License

MIT License - see LICENSE file for details

## Support

For issues or questions:
- RunPod Discord: [discord.gg/runpod](https://discord.gg/runpod)
- GitHub Issues: Open an issue in this repository

## Credits

- [Qwen-Image](https://huggingface.co/Qwen) by Alibaba
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflow engine
- [RunPod](https://runpod.io) serverless platform
