from __future__ import annotations

import os
import itertools
import time
import openai

from datetime import timedelta
import random

from typing import List, Dict, Union, Any, Generator

from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from src.model.model import Model
from src import cache


class RitsModel(Model):
    """
    Wrapper for calling OpenAI with some safety and retry loops as well as a somewhat advanced caching mechanism.
    """

    engine: str
    api_max_attempts: int
    api_endpoint: str

    max_tokens: int
    stop_token: str

    log_probs: int
    num_samples: int

    echo: bool

    temperature: float

    def __init__(
            self,
            engine: str = 'text-davinci-003',
            api_max_attempts: int = 60,
            api_endpoint: str = 'completion',

            temperature: float = 1.0,
            top_p: float = 1.0,
            max_tokens: int = 2049,
            stop_token: str = None,
            log_probs: int = 1,
            num_samples: int = 1,
            echo: bool = True,

            prompt_cost: float = None,
            completion_cost: float = None

    ):
        """

        :param engine: The model you are calling
        :param api_max_attempts: Retry the api call N times.
        :param api_endpoint: Usually differs between completion or chat endpoint
        :param temperature: https://platform.openai.com/docs/api-reference/chat/create#chat/create-temperature
        :param top_p: https://platform.openai.com/docs/api-reference/chat/create#chat/create-top_p
        :param max_tokens: https://platform.openai.com/docs/api-reference/chat/create#chat/create-max_tokens
        :param stop_token: https://platform.openai.com/docs/api-reference/chat/create#chat/create-stop
        :param log_probs: (only for completion) https://platform.openai.com/docs/api-reference/completions/create#completions/create-logprobs
        :param num_samples: https://platform.openai.com/docs/api-reference/chat/create#chat/create-n
        :param echo: (only for completion) https://platform.openai.com/docs/api-reference/completions/create#completions/create-echo
        :param prompt_cost: Pass in the current cost of the api you are calling to track costs (optional)
        :param completion_cost: Pass in the current cost of the api you are calling to track costs (optional)
        """

        self.engine = engine

        self.api_max_attempts = api_max_attempts
        self.api_endpoint = api_endpoint.lower()

        self.max_tokens = max_tokens
        self.stop_token = stop_token

        self.log_probs = log_probs
        self.num_samples = num_samples

        self.echo = echo

        self.temperature = temperature
        self.top_p = top_p

        self.gpt_waittime = 60

        self.prompt_cost = prompt_cost
        self.completion_cost = completion_cost
        self.total_cost = 0.0

        print(f"At rits.py openai key is {openai.api_key}")
        if not openai.api_key:
            openai.api_key = os.getenv("RITS_API_KEY")


        if engine == "mistralai/mixtral-8x22B-instruct-v0.1":
            self.base_url = 'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x22b-instruct-v01/v1'
        elif engine in ["ibm-granite/granite-3.0-8b-instruct"]:
            self.base_url = 'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-0-8b-instruct/v1'
        elif engine in ["meta-llama/llama-3-1-70b-instruct"]:
            self.base_url = 'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-1-70b-instruct/v1'
        elif engine == "microsoft/phi-4":
            self.base_url = 'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/microsoft-phi-4/v1'
        else:
            assert False, f'Unsupported model {engine}'

        self.client = openai.OpenAI(
            api_key="EMPTY",
            base_url=self.base_url,
            default_headers={"RITS_API_KEY": os.environ.get("RITS_API_KEY")},
        )

    def __update_cost__(self, raw):
        if self.prompt_cost and self.completion_cost:
            cost = raw.usage.completion_tokens * self.completion_cost + raw.usage.prompt_tokens * self.prompt_cost
            self.total_cost += cost

    @cache.cached(data_ex=timedelta(days=30), no_data_ex=timedelta(hours=1), prepended_key_attr='engine,num_samples,log_probs,echo,temperature=float(0),top_p=float(1.0),stop_token,max_tokens')
    def inference(self, prompt: str, *args, **kwargs) -> Any:
        if self.api_endpoint == 'completion':
            out = self.__safe_openai_completion_call__(
                prompt,
                *args,
                **kwargs
            )
        elif self.api_endpoint == 'chat':
            out = self.__safe_openai_chat_call__(
                prompt,
                *args,
                **kwargs
            )
        else:
            raise Exception(f"Unknown api endpoint for openai model: {self.api_endpoint}")

        self.__update_cost__(out)
        return out

    def __safe_openai_completion_call__(
            self,
            prompt: str,
            temperature: float = None,
            max_tokens: int = None,
            stop_token: str = None,
            logprobs: int = None,
            num_samples: int = None,
            echo: bool = None
    ) -> ChatCompletion | Stream[ChatCompletionChunk]: # Dict[str, Union[str, bool]]:
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if stop_token is None:
            stop_token = self.stop_token
        if logprobs is None:
            logprobs = self.log_probs
        if num_samples is None:
            num_samples = self.num_samples
        if echo is None:
            echo = self.echo

        last_exc = None
        for i in range(self.api_max_attempts):
            try:
                return self.client.chat.completions.create(
                    engine=self.engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                    n=num_samples,
                    echo=echo,
                    stop=stop_token
                )
            except openai.RateLimitError as e:
                last_exc = e
                print(f"ERROR: OPENAI Rate Error: {e}")
                time.sleep(self.gpt_waittime + int(random.randint(1, 10)))
            except openai.APIConnectionError as e:
                last_exc = e
                print(f"ERROR: OPENAI APIConnection Error: {e}")
            except openai.APIError as e:
                last_exc = e
                print(f"ERROR: OPENAI API Error: {e}")
            except openai.Timeout as e:
                last_exc = e
                print(f"ERROR: OPENAI Timeout Error: {e}")
            except openai.AuthenticationError as e:
                last_exc = e
                print(f"ERROR: OPENAI Authentication Error: {e}")
        # make a fake response
        return {
                "text": prompt + " OPENAI Error - " + str(last_exc),
                "API Error": True,
        }

    def __safe_openai_chat_call__(
            self,
            prompt: str,
            system_prompt: str = None,
            temperature: float = None,
            top_p: float = None,
            max_tokens: int = None,
            stop_token: str = None,
            num_samples: int = None,
    ) -> ChatCompletion | dict[str, bool | str]:
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if stop_token is None:
            stop_token = self.stop_token
        if num_samples is None:
            num_samples = self.num_samples

        last_exc = None
        for i in range(self.api_max_attempts):
            try:
                # TODO - look at different roles?
                messages = [
                    {"role": "user", "content": prompt}
                ]

                if system_prompt:
                    messages = [{'role': 'system', 'content': system_prompt}, {"role": "user", "content": prompt}]

                return self.client.chat.completions.create(
                    model=self.engine,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n=num_samples,
                    stop=stop_token
                )
            except openai.RateLimitError as e:
                last_exc = e
                print(f"ERROR: OPENAI Rate Error: {e}")
                time.sleep(self.gpt_waittime)
            except openai.APIError as e:
                last_exc = e
                print(f"ERROR: OPENAI API Error: {e}")
            except openai.Timeout as e:
                last_exc = e
                print(f"ERROR: OPENAI Timeout Error: {e}")
            except openai.error.APIConnectionError as e:
                last_exc = e
                print(f"ERROR: OPENAI APIConnection Error: {e}")
            except openai.error.ServiceUnavailableError as e:
                last_exc = e
                print(f"ERROR: OPENAI Service Error: {e}")
        # make a fake response
        return {
            "text": prompt + " OPENAI Error - " + str(last_exc),
            "API Error": True,
        }
