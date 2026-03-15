import requests
import json
from typing import List, Dict, Any, Optional

class EndeeClient:
    """ Python client for interacting with the Endee Vector Database API. """

    def __init__(self, base_url: str = "http://localhost:8080", auth_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json"
        }
        if auth_token:
            self.headers["Authorization"] = auth_token

    def health_check(self) -> bool:
        """ Check if the Endee server is healthy. """
        try:
            response = requests.get(f"{self.base_url}/api/v1/health")
            return response.status_code == 200
        except Exception:
            return False

    def list_indexes(self) -> List[Dict[str, Any]]:
        """ List all indexes in the database. """
        response = requests.get(f"{self.base_url}/api/v1/index/list", headers=self.headers)
        if response.status_code == 200:
            return response.json().get("indexes", [])
        return []

    def create_index(self, name: str, dimension: int, space_type: str = "cosine"):
        """ Create a new search index. """
        payload = {
            "index_name": name,
            "dim": dimension,
            "space_type": space_type
        }
        response = requests.post(f"{self.base_url}/api/v1/index/create", headers=self.headers, json=payload)
        return response.status_code == 200

    def delete_index(self, name: str):
        """ Delete a search index. """
        response = requests.delete(f"{self.base_url}/api/v1/index/{name}/delete", headers=self.headers)
        return response.status_code == 200

    def insert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]):
        """ Insert or upsert vectors into an index. """
        # Endee expects a list of objects or a single object
        response = requests.post(
            f"{self.base_url}/api/v1/index/{index_name}/vector/insert",
            headers=self.headers,
            json=vectors
        )
        return response.status_code == 200

    def search(self, index_name: str, query_vector: List[float], k: int = 5):
        """ Perform a similarity search in an index. """
        payload = {
            "vector": query_vector,
            "k": k
        }
        response = requests.post(
            f"{self.base_url}/api/v1/index/{index_name}/search",
            headers=self.headers,
            json=payload
        )
        if response.status_code == 200:
            return response.json().get("hits", [])
        return []
