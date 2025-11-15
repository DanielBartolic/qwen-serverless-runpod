#!/usr/bin/env python3
"""
RunPod Serverless Handler for ComfyUI Qwen-Image
Handles incoming requests and processes them through ComfyUI
"""

import runpod

import os
import sys
import json
import time
import base64
import logging
import traceback
import uuid
import asyncio
import threading
from pathlib import Path
from io import BytesIO
from PIL import Image



# Add ComfyUI to path
sys.path.insert(0, '/app/ComfyUI')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for ComfyUI server
COMFYUI_SERVER = None
COMFYUI_THREAD = None
COMFYUI_READY = False

def start_comfyui_server():
    """Start ComfyUI server in the background"""
    global COMFYUI_READY
    
    try:
        logger.info("Starting ComfyUI server...")
        
        # Import ComfyUI components
        import main
        import server
        import execution
        import folder_paths
        
        # Configure paths
        folder_paths.models_dir = "/app/ComfyUI/models"
        folder_paths.output_directory = "/tmp/outputs"
        os.makedirs("/tmp/outputs", exist_ok=True)
        
        # Start the server
        import argparse
        args = argparse.Namespace(
            listen='127.0.0.1',
            port=8188,
            enable_cors_header=None,
            extra_model_paths_config=None,
            output_directory='/tmp/outputs',
            temp_directory=None,
            auto_launch=False,
            disable_auto_launch=True,
            cuda_device=0,
            dont_upcast_attention=False,
            force_upcast_attention=False,
            disable_ipex_optimize=False,
            preview_method='auto',
            disable_cuda_malloc=False
        )
        
        # Initialize server
        server.BinaryEventTypes = ["preview"]
        server.PromptServer.instance = server.PromptServer(None)
        
        # Start execution loop
        import execution
        execution.PromptExecutor(server.PromptServer.instance)
        
        COMFYUI_READY = True
        logger.info("ComfyUI server ready!")
        
    except Exception as e:
        logger.error(f"Failed to start ComfyUI server: {e}")
        logger.error(traceback.format_exc())

class ComfyUIHandler:
    """Handler for ComfyUI workflow execution"""
    
    def __init__(self):
        self.workflow_path = "/app/workflow.json"
        self.output_dir = "/tmp/outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load workflow template
        with open(self.workflow_path, 'r') as f:
            self.workflow_template = json.load(f)
            
        # Import ComfyUI components after path is set
        self._import_comfyui_components()
        
    def _import_comfyui_components(self):
        """Import required ComfyUI components"""
        try:
            import execution
            import folder_paths
            import nodes
            import server
            
            self.execution = execution
            self.folder_paths = folder_paths
            self.nodes = nodes
            
            # Initialize prompt executor
            if not hasattr(server.PromptServer, 'instance') or server.PromptServer.instance is None:
                server.PromptServer.instance = server.PromptServer(None)
            
            self.prompt_executor = execution.PromptExecutor(server.PromptServer.instance)
            
            logger.info("ComfyUI components imported successfully")
            
        except Exception as e:
            logger.error(f"Failed to import ComfyUI components: {e}")
            raise
            
    def update_workflow(self, workflow, params):
        """Update workflow with custom parameters
        
        Args:
            workflow: The workflow dict to modify
            params: Parameters dict with prompt, width, height, seed, etc.
        """
        # Update prompt (node 231)
        if "231" in workflow and "prompt" in params:
            workflow["231"]["inputs"]["String"] = params["prompt"]
            logger.info(f"Set prompt: {params['prompt'][:50]}...")
            
        # Update negative prompt (node 7)
        if "7" in workflow and "negative_prompt" in params:
            workflow["7"]["inputs"]["text"] = params["negative_prompt"]
            
        # Update resolution (nodes 91, 92)
        if "91" in workflow and "width" in params:
            workflow["91"]["inputs"]["Number"] = str(params["width"])
            logger.info(f"Set width: {params['width']}")
            
        if "92" in workflow and "height" in params:
            workflow["92"]["inputs"]["Number"] = str(params["height"])
            logger.info(f"Set height: {params['height']}")
            
        # Update seed (node 75 - ClownsharKSampler_Beta)
        if "75" in workflow and "seed" in params:
            seed = params["seed"]
            if seed == -1:
                import random
                seed = random.randint(0, 2**32 - 1)
            workflow["75"]["inputs"]["seed"] = seed
            logger.info(f"Set seed: {seed}")
            
        # Update steps if provided
        if "75" in workflow and "steps" in params:
            workflow["75"]["inputs"]["steps"] = params["steps"]
            
        # Update CFG if provided
        if "75" in workflow and "cfg" in params:
            workflow["75"]["inputs"]["cfg"] = params["cfg"]
            
        return workflow
        
    def execute_workflow(self, params):
        """Execute ComfyUI workflow with given parameters
        
        Args:
            params: Dict with generation parameters
            
        Returns:
            Dict with base64 encoded image and metadata
        """
        try:
            # Copy workflow template
            workflow = json.loads(json.dumps(self.workflow_template))
            
            # Update workflow with parameters
            workflow = self.update_workflow(workflow, params)
            
            # Generate unique prompt ID
            prompt_id = str(uuid.uuid4())
            
            logger.info(f"Executing workflow with prompt_id: {prompt_id}")
            
            # Execute the workflow
            # Create prompt structure for ComfyUI
            prompt_data = {
                "prompt": workflow,
                "extra_data": {
                    "extra_pnginfo": {"workflow": workflow}
                },
                "client_id": prompt_id
            }
            
            # Validate prompt
            valid, error, outputs = self.execution.validate_prompt(prompt_data["prompt"])
            
            if not valid:
                logger.error(f"Invalid prompt: {error}")
                return {
                    "success": False,
                    "error": f"Invalid workflow: {error}"
                }
                
            # Execute prompt
            logger.info("Starting execution...")
            
            # Add to execution queue
            self.prompt_executor.server.last_prompt_id = prompt_id
            outputs = self.prompt_executor.execute(
                prompt_data["prompt"],
                prompt_id,
                prompt_data.get("extra_data", {}),
                outputs
            )
            
            logger.info("Execution completed, looking for outputs...")
            
            # Find generated images
            images = []
            
            # Check the outputs directory for images
            output_path = Path(self.output_dir)
            
            # Look for Qwen subdirectory (from SaveImage node)
            qwen_dir = output_path / "Qwen"
            if qwen_dir.exists():
                for img_file in qwen_dir.glob("*.png"):
                    if img_file.stat().st_mtime > time.time() - 60:  # Recent files only
                        images.append(img_file)
                        
            # Also check root output directory
            for img_file in output_path.glob("*.png"):
                if img_file.stat().st_mtime > time.time() - 60:
                    images.append(img_file)
                    
            if not images:
                logger.error("No images found in outputs")
                return {
                    "success": False,
                    "error": "No images generated"
                }
                
            # Get the most recent image
            latest_image = max(images, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found image: {latest_image}")
            
            # Convert to base64
            with open(latest_image, 'rb') as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
            # Get image metadata
            img = Image.open(latest_image)
            
            # Clean up generated file
            try:
                os.remove(latest_image)
            except:
                pass
                
            return {
                "success": True,
                "image": img_base64,
                "seed": workflow["75"]["inputs"]["seed"] if "75" in workflow else None,
                "width": img.width,
                "height": img.height,
                "prompt": params.get("prompt", ""),
                "format": "png"
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

# Initialize handler globally
handler = None

def initialize_handler():
    """Initialize the ComfyUI handler"""
    global handler
    
    if handler is None:
        logger.info("Initializing ComfyUI handler...")
        handler = ComfyUIHandler()
        logger.info("Handler initialized successfully")
        
    return handler

def runpod_handler(job):
    """RunPod serverless handler function
    
    Expected input format:
    {
        "input": {
            "prompt": "string",
            "negative_prompt": "string (optional)",
            "width": int (optional, default: 1440),
            "height": int (optional, default: 1920),
            "seed": int (optional, default: -1),
            "steps": int (optional, default: 25),
            "cfg": float (optional, default: 1.0)
        }
    }
    """
    try:
        logger.info(f"Received job: {job.get('id', 'unknown')}")
        
        # Get input parameters
        job_input = job.get("input", {})
        
        # Set defaults
        params = {
            "prompt": job_input.get("prompt", "a beautiful landscape"),
            "negative_prompt": job_input.get("negative_prompt", 
                "cinematic, glossy finish, shallow depth of field, cinematic bokeh, uncanny anatomy, frame-perfect symmetry, blurred background, fat, big lips, thic lips, bad anatomy"),
            "width": job_input.get("width", 1440),
            "height": job_input.get("height", 1920),
            "seed": job_input.get("seed", -1),
            "steps": job_input.get("steps", 25),
            "cfg": job_input.get("cfg", 1.0)
        }
        
        logger.info(f"Processing with params: {params}")
        
        # Initialize handler if needed
        h = initialize_handler()
        
        # Execute workflow
        result = h.execute_workflow(params)
        
        if result["success"]:
            logger.info(f"Successfully generated image: {result['width']}x{result['height']}")
            
            # Return in RunPod format
            return {
                "image": result["image"],
                "seed": result["seed"],
                "width": result["width"],
                "height": result["height"],
                "prompt": result["prompt"]
            }
        else:
            logger.error(f"Generation failed: {result.get('error', 'Unknown error')}")
            raise Exception(result.get("error", "Generation failed"))
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(traceback.format_exc())
        
        # Return error in RunPod format
        return {
            "error": str(e)
        }

def test_local():
    """Test function for local development"""
    logger.info("Running local test...")
    
    # Initialize handler
    h = initialize_handler()
    
    # Test parameters
    params = {
        "prompt": "a cyberpunk city at night, neon lights",
        "negative_prompt": "blurry, low quality",
        "width": 1440,
        "height": 1920,
        "seed": 123456,
        "steps": 25,
        "cfg": 1.0
    }
    
    # Execute
    result = h.execute_workflow(params)
    
    if result["success"]:
        # Save test image
        img_data = base64.b64decode(result["image"])
        with open("/tmp/test_output.png", "wb") as f:
            f.write(img_data)
        logger.info(f"Test successful! Image saved to /tmp/test_output.png")
        logger.info(f"Image size: {result['width']}x{result['height']}")
        logger.info(f"Seed: {result['seed']}")
    else:
        logger.error(f"Test failed: {result.get('error')}")

if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        # Run local test
        test_local()
    else:
        # Run as RunPod serverless handler
        logger.info("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": runpod_handler})
