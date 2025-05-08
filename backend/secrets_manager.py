# secrets_manager.py
from google.cloud import secretmanager
import os
from functools import lru_cache

class SecretManager:
    def __init__(self):
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = os.getenv("GCP_PROJECT_ID") or "fitai-459007"
    
    @lru_cache(maxsize=128)
    def get_secret(self, secret_id: str, version: str = "latest") -> str:
        """
        Retrieve a secret from Google Cloud Secret Manager
        
        Args:
            secret_id: The ID of the secret
            version: The version of the secret (default: "latest")
            
        Returns:
            The secret value as a string
        """
        try:
            # Build the resource name
            name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
            
            # Access the secret version
            response = self.client.access_secret_version(request={"name": name})
            
            # Decode the secret payload
            payload = response.payload.data.decode("UTF-8")
            return payload
            
        except Exception as e:
            print(f"Error accessing secret {secret_id}: {str(e)}")
            raise
    
    def get_openai_key(self) -> str:
        """Get the OpenAI API key from Secret Manager"""
        return self.get_secret("openai-api-key")