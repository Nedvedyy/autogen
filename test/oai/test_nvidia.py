from unittest.mock import MagicMock, patch

import pytest

try:
    from autogen.oai.nvidia import NvidiaClient, calculate_nvidia_cost

    skip = False
except ImportError:
    NvidiaClient = object
    InternalServerError = object
    skip = True


# Fixtures for mock data
@pytest.fixture
def mock_response():
    class MockResponse:
        def __init__(self, text, choices, usage, cost, model):
            self.text = text
            self.choices = choices
            self.usage = usage
            self.cost = cost
            self.model = model

    return MockResponse


@pytest.fixture
def nvidia_client():
    return NvidiaClient(api_key="fake_api_key")


# Test initialization and configuration
@pytest.mark.skipif(skip, reason="NvidiaClient dependency is not installed")
def test_initialization():
    # Missing any api_key
    with pytest.raises(AssertionError) as assertinfo:
        NvidiaClient()  # Should raise an AssertionError due to missing api_key

    assert (
        "Please specify the 'api_key' in your config list entry for Nvidia or set the NVIDIA_API_KEY env variable."
        in str(assertinfo.value)
    )

    # Creation works
    NvidiaClient(api_key="fake_api_key")  # Should create okay now.


# Test standard initialization
@pytest.mark.skipif(skip, reason="NvidiaClient dependency is not installed")
def test_valid_initialization(nvidia_client):
    assert nvidia_client.api_key == "fake_api_key", "Config api_key should be correctly set"


# Test cost calculation
@pytest.mark.skipif(skip, reason="NvidiaClient dependency is not installed")
def test_cost_calculation(mock_response):
    response = mock_response(
        text="Example response",
        choices=[{"message": "Test message 1"}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        cost=None,
        model="nvidia-large-latest",
    )
    assert (
        calculate_nvidia_cost(response.usage["prompt_tokens"], response.usage["completion_tokens"], response.model)
        == 0.0001
    ), "Cost for this should be $0.0001"


# Test text generation
@pytest.mark.skipif(skip, reason="NvidiaClient dependency is not installed")
@patch("autogen.oai.nvidia.NvidiaClient.chat")
def test_create_response(mock_chat, nvidia_client):
    # Mock NvidiaClient.chat response
    mock_nvidia_response = MagicMock()
    mock_nvidia_response.choices = [
        MagicMock(finish_reason="stop", message=MagicMock(content="Example Nvidia response", tool_calls=None))
    ]
    mock_nvidia_response.id = "mock_nvidia_response_id"
    mock_nvidia_response.model = "nvidia-small-latest"
    mock_nvidia_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)  # Example token usage

    mock_chat.return_value = mock_nvidia_response

    # Test parameters
    params = {
        "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "World"}],
        "model": "nvidia-small-latest",
    }

    # Call the create method
    response = nvidia_client.create(params)

    # Assertions to check if response is structured as expected
    assert (
        response.choices[0].message.content == "Example Nvidia response"
    ), "Response content should match expected output"
    assert response.id == "mock_nvidia_response_id", "Response ID should match the mocked response ID"
    assert response.model == "nvidia-small-latest", "Response model should match the mocked response model"
    assert response.usage.prompt_tokens == 10, "Response prompt tokens should match the mocked response usage"
    assert response.usage.completion_tokens == 20, "Response completion tokens should match the mocked response usage"


# Test functions/tools
@pytest.mark.skipif(skip, reason="NvidiaClient dependency is not installed")
@patch("autogen.oai.nvidia.NvidiaClient.chat")
def test_create_response_with_tool_call(mock_chat, nvidia_client):
    # Mock `nvidia_response = client.chat(**nvidia_params)`
    mock_function = MagicMock(name="currency_calculator")
    mock_function.name = "currency_calculator"
    mock_function.arguments = '{"base_currency": "EUR", "quote_currency": "USD", "base_amount": 123.45}'

    mock_function_2 = MagicMock(name="get_weather")
    mock_function_2.name = "get_weather"
    mock_function_2.arguments = '{"location": "Chicago"}'

    mock_chat.return_value = MagicMock(
        choices=[
            MagicMock(
                finish_reason="tool_calls",
                message=MagicMock(
                    content="Sample text about the functions",
                    tool_calls=[
                        MagicMock(id="gdRdrvnHh", function=mock_function),
                        MagicMock(id="abRdrvnHh", function=mock_function_2),
                    ],
                ),
            )
        ],
        id="mock_nvidia_response_id",
        model="nvidia-small-latest",
        usage=MagicMock(prompt_tokens=10, completion_tokens=20),
    )

    # Construct parameters
    converted_functions = [
        {
            "type": "function",
            "function": {
                "description": "Currency exchange calculator.",
                "name": "currency_calculator",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "base_amount": {"type": "number", "description": "Amount of currency in base_currency"},
                    },
                    "required": ["base_amount"],
                },
            },
        }
    ]
    nvidia_messages = [
        {"role": "user", "content": "How much is 123.45 EUR in USD?"},
        {"role": "assistant", "content": "World"},
    ]

    # Call the create method
    response = nvidia_client.create(
        {"messages": nvidia_messages, "tools": converted_functions, "model": "nvidia-medium-latest"}
    )

    # Assertions to check if the functions and content are included in the response
    assert response.choices[0].message.content == "Sample text about the functions"
    assert response.choices[0].message.tool_calls[0].function.name == "currency_calculator"
    assert response.choices[0].message.tool_calls[1].function.name == "get_weather"
