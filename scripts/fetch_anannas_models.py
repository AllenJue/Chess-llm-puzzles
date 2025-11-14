"""
Utility script to fetch available models from Anannas API.
"""

import os
import requests
from typing import List

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
except ImportError:
    pass  # dotenv not available, continue without it


def fetch_free_models(api_key: str = None, base_url: str = "https://api.anannas.ai/v1") -> List[str]:
    """Fetch all available free models from Anannas API."""
    if api_key is None:
        api_key = os.getenv("ANANNAS_API_KEY")
    
    if not api_key:
        raise ValueError("ANANNAS_API_KEY not provided")
    
    try:
        response = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract all models with :free in their ID
        free_models = [
            model["id"] 
            for model in data.get("data", [])
            if ":free" in model.get("id", "")
        ]
        
        # Sort for consistent output
        free_models.sort()
        return free_models
    except Exception as e:
        print(f"Error fetching models: {e}")
        raise


if __name__ == "__main__":
    import sys
    try:
        models = fetch_free_models()
        print(f"Found {len(models)} free models:")
        for model in models:
            print(f"  - {model}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

