import os
import time
from typing import Any

import openai
from openai import OpenAI

from ..types import MessageList, SamplerBase, SamplerResponse, Message

HOSTED_MODEL_SYSTEM_MESSAGE = "You are a helpful assistant."


class HostedModelChatCompletionSampler(SamplerBase):
    """
    Sample from a hosted model.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 4096,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
        qwen3_no_thinking: bool = False,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.reasoning_model = reasoning_model
        self.reasoning_effort = reasoning_effort
        self.qwen3_no_thinking = qwen3_no_thinking

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def _add_no_thinking_instruction(self, message: Message):
        return {
            "role": message["role"],
            "content": message["content"] + " /no_think",
        }

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.qwen3_no_thinking:
            message_list = [
                self._add_no_thinking_instruction(msg) for msg in message_list
            ]
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
