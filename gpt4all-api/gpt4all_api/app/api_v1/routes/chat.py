import json
import logging
import time
from typing import Dict, Iterable, List, Union, Optional

from uuid import uuid4
import aiohttp
import asyncio

from api_v1.settings import settings
from fastapi import APIRouter, Depends, Response, Security, status, HTTPException
from fastapi.responses import StreamingResponse
from gpt4all import GPT4All
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

### This should follow https://github.com/openai/openai-openapi/blob/master/openapi.yaml


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionStreamMessage(BaseModel):
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(settings.model, description='The model to generate a completion from.')
    messages: List[ChatCompletionMessage] = Field(..., description='The model to generate a completion from.')
    temperature: float = Field(settings.temp, description='Model temperature')
    top_p: Optional[float] = Field(settings.top_p, description='top_p')
    top_k: Optional[int] = Field(settings.top_k, description='top_k')
    n: int = Field(1, description='How many completions to generate for each prompt')
    stream: bool = Field(False, description='Stream responses')
    max_tokens: int = Field(None, description='Max tokens to generate')
    presence_penalty: float = Field(0.0, description='Positive values penalize new tokens based on whether they appear in the text so far, increasing the model\'s likelihood to talk about new topics.')
    frequency_penalty: float = Field(0.0, description='Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model\'s likelihood to repeat the same line verbatim.')


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatCompletionStreamMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = 'chat.completion'
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = 'chat.completion'
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


router = APIRouter(prefix="/chat", tags=["Completions Endpoints"])

def stream_completion(output: Iterable, base_response: ChatCompletionStreamResponse):
    """
    Streams a GPT4All output to the client.

    Args:
        output: The output of GPT4All.generate(), which is an iterable of tokens.
        base_response: The base response object, which is cloned and modified for each token.

    Returns:
        A Generator of CompletionStreamResponse objects, which are serialized to JSON Event Stream format.
    """
    for token in output:
        chunk = base_response.copy()
        chunk.choices = [dict(ChatCompletionStreamChoice(
            index=0,
            delta=ChatCompletionStreamMessage(
                content=token
            ),
            finish_reason='stop'
        ))]
        yield f"data: {json.dumps(dict(chunk))}\n\n"

async def gpu_infer(payload, header):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                settings.hf_inference_server_host, headers=header, data=json.dumps(payload)
            ) as response:
                resp = await response.json()
            return resp

        except aiohttp.ClientError as e:
            # Handle client-side errors (e.g., connection error, invalid URL)
            logger.error(f"Client error: {e}")
        except aiohttp.ServerError as e:
            # Handle server-side errors (e.g., internal server error)
            logger.error(f"Server error: {e}")
        except json.JSONDecodeError as e:
            # Handle JSON decoding errors
            logger.error(f"JSON decoding error: {e}")
        except Exception as e:
            # Handle other unexpected exceptions
            logger.error(f"Unexpected error: {e}")


@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    '''
    Completes a GPT4All model response.
    '''
    if request.model != settings.model:
        raise HTTPException(status_code=400,
                            detail=f"The GPT4All inference server is booted to only infer: `{settings.model}`")
    if isinstance(request.messages, list):
        if len(request.messages) > 1:
            raise HTTPException(status_code=400, detail="Can only infer one inference per request in CPU mode.")
        else:
            request.messages = request.messages[0]

    model = GPT4All(model_name=settings.model, model_path=settings.gpt4all_path)

    output = model.generate(prompt=request.messages,
                            max_tokens=request.max_tokens,
                            temp=request.temperature,
                            top_k=request.top_k,
                            top_p=request.top_p,
                            repeat_penalty=request.presence_penalty,
                            repeat_last_n=request.frequency_penalty,
                            n_batch=request.n,
                            streaming=request.stream,
                            )

    if request.stream:
        base_chunk = ChatCompletionStreamResponse(
            id=str(uuid4()),
            created=time.time(),
            model=request.model,
            choices=[] #not implemented 
        )

        return StreamingResponse((response for response in stream_completion(output, base_chunk)),
                                            media_type='text/event-stream')
    else:
        return ChatCompletionResponse(
            id=str(uuid4()),
            created=time.time(),
            model=request.model,
            choices=[dict(ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role='assistant',
                    content=output
                ),
                finish_reason='stop'
            ))],
            usage={
                'prompt_tokens': 0,  # TODO how to compute this?
                'completion_tokens': 0,
                'total_tokens': 0
            }
        )
