# autogen/oai/nvidia.py
import time
from typing import Any, Dict, List, Union
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.completion_usage import CompletionUsage
from autogen import ModelClient
from autogen.oai.client_utils import validate_parameter

class NvidiaClient(ModelClient):
    """Client for Nvidia's API."""

    def __init__(self, **kwargs):
        """Requires api_key and base_url or environment variable to be set

        Args:
            api_key (str): The API key for using Nvidia (or environment variable NVIDIA_API_KEY needs to be set)
            base_url (str): The base URL for the Nvidia API (or default to "https://integrate.api.nvidia.com/v1")
        """
        # Ensure we have the api_key upon instantiation
        self.api_key = kwargs.get("api_key", None)
        if not self.api_key:
            self.api_key = os.getenv("NVIDIA_API_KEY", None)
        assert self.api_key, "Please specify the 'api_key' in your config list entry for Nvidia or set the NVIDIA_API_KEY env variable."
        
        # Ensure we have the base_url upon instantiation
        self.base_url = kwargs.get("base_url", None)
        if not self.base_url:
            self.base_url = os.getenv("NVIDIA_BASE_URL", None)
        assert self.api_key, "Please specify the 'base_url' in your config list entry for Nvidia or set the NVIDIA_BASE_URL env variable."
        

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def create(self, params: Dict[str, Any]) -> ChatCompletion:
        # Validate and convert parameters
        nvidia_params = self.parse_params(params)

        # Call Nvidia API
        response = self.client.chat.completions.create(**nvidia_params)

        # Convert Nvidia response to OpenAI-compatible format
        message = ChatCompletionMessage(
            role="assistant",
            content=response.choices[0].delta.content,
            function_call=None,
            tool_calls=None,
        )
        choices = [ChatCompletionMessageToolCall(finish_reason="stop", index=0, message=message)]

        return ChatCompletion(
            id=response.id,
            model=response.model,
            created=int(time.time()),
            object="chat.completion",
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
            cost=self.calculate_cost(response)
        )

    def message_retrieval(self, response: ChatCompletion) -> Union[List[str], List[ChatCompletionMessage]]:
        """Retrieve the messages from the response."""
        return [choice.message for choice in response.choices]

    def calculate_cost(self, response) -> float:
        # Assuming cost calculation is based on tokens used
        usage = response.usage
        cost_per_token = 0.0001  # Example cost per token, adjust based on Nvidia's pricing
        return usage.total_tokens * cost_per_token

    @staticmethod
    def get_usage(response: ChatCompletion) -> Dict:
        return {
            "prompt_tokens": response.usage.prompt_tokens if response.usage is not None else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage is not None else 0,
            "total_tokens": response.usage.total_tokens if response.usage is not None else 0,
            "cost": response.cost if hasattr(response, 'cost') else 0,
            "model": response.model,
        }

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Loads and validates parameters for Nvidia API."""
        nvidia_params = {}
        nvidia_params["model"] = params.get("model", None)
        assert nvidia_params["model"], "Please specify the 'model' in your config list entry to nominate the Nvidia model to use."

        nvidia_params["temperature"] = validate_parameter(params, "temperature", (int, float), True, 0.7, None, None)
        nvidia_params["top_p"] = validate_parameter(params, "top_p", (int, float), True, None, None, None)
        nvidia_params["max_tokens"] = validate_parameter(params, "max_tokens", int, True, None, (0, None), None)
        nvidia_params["messages"] = params["messages"]
        
        return nvidia_params




