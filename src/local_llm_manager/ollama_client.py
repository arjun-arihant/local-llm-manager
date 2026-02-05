"""Ollama API client."""

import requests
import json
import time
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass


OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass
class OllamaModel:
    """Ollama model information."""
    name: str
    modified_at: str
    size: int
    digest: str
    details: Dict[str, Any]


@dataclass
class PullProgress:
    """Model pull progress."""
    status: str
    completed: Optional[int]
    total: Optional[int]
    percent: float


class OllamaClient:
    """Client for Ollama API."""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def _get(self, endpoint: str) -> Any:
        """Make GET request to Ollama API."""
        response = self.session.get(f"{self.base_url}/api/{endpoint}")
        response.raise_for_status()
        return response.json()
    
    def _post(self, endpoint: str, data: Dict[str, Any]) -> Any:
        """Make POST request to Ollama API."""
        response = self.session.post(
            f"{self.base_url}/api/{endpoint}",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def is_running(self) -> bool:
        """Check if Ollama is running."""
        try:
            self._get("tags")
            return True
        except:
            return False
    
    def list_models(self) -> List[OllamaModel]:
        """List all installed models."""
        data = self._get("tags")
        models = []
        for m in data.get("models", []):
            models.append(OllamaModel(
                name=m.get("name", "unknown"),
                modified_at=m.get("modified_at", ""),
                size=m.get("size", 0),
                digest=m.get("digest", ""),
                details=m.get("details", {})
            ))
        return models
    
    def pull_model(self, name: str) -> Iterator[PullProgress]:
        """Pull a model with progress updates."""
        response = self.session.post(
            f"{self.base_url}/api/pull",
            json={"name": name},
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status", "")
                completed = data.get("completed")
                total = data.get("total")
                
                percent = 0.0
                if total and completed:
                    percent = (completed / total) * 100
                elif "pulling" in status:
                    percent = -1  # Indeterminate
                elif "complete" in status.lower() or "success" in status.lower():
                    percent = 100.0
                
                yield PullProgress(
                    status=status,
                    completed=completed,
                    total=total,
                    percent=percent
                )
    
    def delete_model(self, name: str) -> bool:
        """Delete a model."""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/delete",
                json={"name": name}
            )
            return response.status_code == 200
        except:
            return False
    
    def generate(self, model: str, prompt: str, stream: bool = False) -> Dict[str, Any]:
        """Generate text with a model."""
        return self._post("generate", {
            "model": model,
            "prompt": prompt,
            "stream": stream
        })
    
    def generate_stream(self, model: str, prompt: str) -> Iterator[str]:
        """Generate text with streaming output."""
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                yield data.get("response", "")
    
    def show_model(self, name: str) -> Optional[Dict[str, Any]]:
        """Show model information."""
        try:
            return self._post("show", {"name": name})
        except:
            return None
    
    def copy_model(self, source: str, destination: str) -> bool:
        """Copy a model."""
        try:
            self._post("copy", {"source": source, "destination": destination})
            return True
        except:
            return False
    
    def create_model(self, name: str, modelfile: str) -> Iterator[Dict[str, Any]]:
        """Create a model from Modelfile."""
        response = self.session.post(
            f"{self.base_url}/api/create",
            json={"name": name, "modelfile": modelfile},
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                yield json.loads(line)
