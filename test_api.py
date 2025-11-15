#!/usr/bin/env python3
"""
Test script for RunPod Serverless API
Tests the deployed Qwen-Image serverless endpoint
"""

import json
import time
import base64
import requests
import argparse
from pathlib import Path
from datetime import datetime

def test_runpod_endpoint(endpoint_id, api_key, prompt, output_dir="./test_outputs"):
    """Test the RunPod serverless endpoint
    
    Args:
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        prompt: Text prompt for generation
        output_dir: Directory to save generated images
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare the request
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted",
            "width": 1440,
            "height": 1920,
            "seed": -1,
            "steps": 25,
            "cfg": 1.0
        }
    }
    
    print("=" * 60)
    print("RunPod Serverless API Test")
    print("=" * 60)
    print(f"Endpoint: {url}")
    print(f"Prompt: {prompt}")
    print(f"Resolution: {payload['input']['width']}x{payload['input']['height']}")
    print("-" * 60)
    
    # Send request
    print("📤 Sending request...")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        elapsed = time.time() - start_time
        
        print(f"✅ Request completed in {elapsed:.1f} seconds")
        
        # Check if job was queued (async) or completed (sync)
        if "output" in result:
            # Sync response - image is ready
            output = result["output"]
            print("✅ Image generated successfully!")
            
        elif "id" in result and "status" in result:
            # Async response - need to poll
            job_id = result["id"]
            print(f"⏳ Job queued with ID: {job_id}")
            
            # Poll for completion
            output = poll_for_completion(endpoint_id, job_id, api_key)
            
        else:
            print(f"❌ Unexpected response format: {result}")
            return
            
        # Save image
        if "image" in output:
            # Decode base64 image
            img_data = base64.b64decode(output["image"])
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qwen_{timestamp}_{output.get('seed', 'unknown')}.png"
            filepath = Path(output_dir) / filename
            
            # Save image
            with open(filepath, "wb") as f:
                f.write(img_data)
                
            print(f"💾 Image saved to: {filepath}")
            
            # Print metadata
            print("\n📊 Generation Details:")
            print(f"  - Seed: {output.get('seed', 'N/A')}")
            print(f"  - Size: {output.get('width', 'N/A')}x{output.get('height', 'N/A')}")
            print(f"  - Prompt: {output.get('prompt', 'N/A')[:50]}...")
            
        elif "error" in output:
            print(f"❌ Generation failed: {output['error']}")
            
        else:
            print(f"❌ No image in response: {output}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
    print("=" * 60)

def poll_for_completion(endpoint_id, job_id, api_key, max_wait=300, poll_interval=2):
    """Poll for async job completion
    
    Args:
        endpoint_id: RunPod endpoint ID
        job_id: Job ID to poll
        api_key: RunPod API key
        max_wait: Maximum seconds to wait
        poll_interval: Seconds between polls
        
    Returns:
        dict: Job output or error
    """
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    start_time = time.time()
    
    while (time.time() - start_time) < max_wait:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            status = result.get("status")
            
            if status == "COMPLETED":
                print("✅ Job completed!")
                return result.get("output", {})
                
            elif status == "FAILED":
                print("❌ Job failed!")
                return {"error": result.get("error", "Unknown error")}
                
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                elapsed = time.time() - start_time
                print(f"⏳ Status: {status} ({elapsed:.0f}s elapsed)", end="\r")
                
        time.sleep(poll_interval)
        
    return {"error": "Timeout waiting for job completion"}

def batch_test(endpoint_id, api_key, num_images=5):
    """Run a batch test with multiple prompts
    
    Args:
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        num_images: Number of images to generate
    """
    
    test_prompts = [
        "a cyberpunk city at night with neon lights",
        "a serene mountain landscape at sunrise",
        "a futuristic robot in a garden",
        "an ancient temple in a jungle",
        "a steampunk airship above clouds",
        "a magical forest with glowing mushrooms",
        "a desert oasis under starry sky",
        "a underwater city with bioluminescent creatures",
        "a medieval castle during sunset",
        "a space station orbiting earth"
    ]
    
    print("\n🚀 Starting batch test...")
    print(f"Generating {num_images} images\n")
    
    successful = 0
    failed = 0
    total_time = 0
    
    for i in range(min(num_images, len(test_prompts))):
        print(f"\n[{i+1}/{num_images}] Generating image...")
        
        start = time.time()
        
        try:
            test_runpod_endpoint(
                endpoint_id,
                api_key,
                test_prompts[i],
                output_dir=f"./batch_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            successful += 1
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            failed += 1
            
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"Time for this image: {elapsed:.1f}s")
        
    # Summary
    print("\n" + "=" * 60)
    print("📊 Batch Test Summary")
    print("=" * 60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"⚡ Average time per image: {total_time/num_images:.1f}s")
    print("=" * 60)

def stress_test(endpoint_id, api_key, concurrent=3):
    """Run concurrent requests to test scaling
    
    Args:
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        concurrent: Number of concurrent requests
    """
    import concurrent.futures
    
    print(f"\n⚡ Starting stress test with {concurrent} concurrent requests...")
    
    prompts = [f"test prompt {i}" for i in range(concurrent)]
    
    def generate_single(prompt):
        url = f"https://api.runpod.ai/v2/{endpoint_id}/run"  # Using async endpoint
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": {
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "steps": 8  # Less steps for stress test
            }
        }
        
        start = time.time()
        response = requests.post(url, json=payload, headers=headers)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            return {"success": True, "time": elapsed, "job_id": response.json().get("id")}
        else:
            return {"success": False, "time": elapsed, "error": response.text}
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [executor.submit(generate_single, p) for p in prompts]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
    # Print results
    successful = sum(1 for r in results if r["success"])
    avg_time = sum(r["time"] for r in results) / len(results)
    
    print(f"\n📊 Stress Test Results:")
    print(f"  Successful: {successful}/{concurrent}")
    print(f"  Average response time: {avg_time:.2f}s")
    
    for i, r in enumerate(results):
        status = "✅" if r["success"] else "❌"
        print(f"  {status} Request {i+1}: {r['time']:.2f}s")

def main():
    parser = argparse.ArgumentParser(description='Test RunPod Serverless Qwen-Image API')
    
    parser.add_argument('--endpoint', '-e', required=True,
                        help='RunPod endpoint ID')
    parser.add_argument('--api-key', '-k', required=True,
                        help='RunPod API key')
    parser.add_argument('--prompt', '-p',
                        default='a beautiful cyberpunk city at night',
                        help='Text prompt for generation')
    parser.add_argument('--output', '-o',
                        default='./test_outputs',
                        help='Output directory for images')
    parser.add_argument('--batch', '-b', type=int,
                        help='Run batch test with N images')
    parser.add_argument('--stress', '-s', type=int,
                        help='Run stress test with N concurrent requests')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_test(args.endpoint, args.api_key, args.batch)
    elif args.stress:
        stress_test(args.endpoint, args.api_key, args.stress)
    else:
        test_runpod_endpoint(args.endpoint, args.api_key, args.prompt, args.output)

if __name__ == '__main__':
    main()
