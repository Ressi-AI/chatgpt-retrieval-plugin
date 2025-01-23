from typing import List
import os
import decouple
import openai
from loguru import logger
import airouter.models
from tenacity import retry, wait_random_exponential, stop_after_attempt

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", 256))


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using OpenAI's ada model.

    Args:
        texts: The list of texts to embed.

    Returns:
        A list of embeddings, each of which is a list of floats.

    Raises:
        Exception: If the OpenAI API call fails.
    """

    if decouple.config("DEFAULT_LLM_PROVIDER", default=None) == airouter.models.ProviderName.AZURE_OPENAI:
        # azure deployment created with the same name as model name
        client = openai.AzureOpenAI(azure_deployment=EMBEDDING_MODEL)
    else:
        client = openai.OpenAI()

    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSION)
    data = response.data

    # Return the embeddings as a list of lists of floats
    return [result.embedding for result in data]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_chat_completion(
    messages,
    model="gpt-3.5-turbo",  # use "gpt-4" for better results
    deployment_id=None,
):
    """
    Generate a chat completion using OpenAI's chat completion API.

    Args:
        messages: The list of messages in the chat history.
        model: The name of the model to use for the completion. Default is gpt-3.5-turbo, which is a fast, cheap and versatile model. Use gpt-4 for higher quality but slower results.

    Returns:
        A string containing the chat completion.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    # call the OpenAI chat completion API with the given messages
    # Note: Azure Open AI requires deployment id
    response = {}
    if deployment_id == None:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
        )
    else:
        response = openai.ChatCompletion.create(
            deployment_id=deployment_id,
            messages=messages,
        )

    choices = response["choices"]  # type: ignore
    completion = choices[0].message.content.strip()
    logger.info(f"Completion: {completion}")
    return completion
