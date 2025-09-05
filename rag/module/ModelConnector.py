import requests
import json

class OllamaConnector:
    def __init__(self, base_url):
        """
        Initialize the Ollama connection.
        :param base_url: URL of the Ollama API (default: local server)
        """
        self.base_url = base_url.rstrip("/")

    def generate(self, model, prompt, stream=False, options=None):
        """
        Send a prompt to the Ollama API and return the response.
        :param model: Name of the model (e.g., 'llama2', 'mistral')
        :param prompt: The input text prompt
        :param stream: Whether to stream the response (default: False)
        :param options: Additional options (temperature, top_p, etc.)
        :return: Response text (or a generator if streaming)
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        if options:
            payload["options"] = options

        if stream:
            return self._stream_response(url, payload)
        else:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

    def _stream_response(self, url, payload):
        """
        Handle streaming response from Ollama API.
        """
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
