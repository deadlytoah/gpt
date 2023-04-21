import os
import sys
from typing import Any, Dict, List, Optional, Union

import openai

import pyservice
from pyservice import Metadata, ProtocolException
from pyservice.gpt import (AssistantMessage, Message, SystemMessage,
                           UserMessage, build_message)
from pyservice.metadata import Arguments, Argument

OPENAI_MODEL = 'gpt-3.5-turbo'


def send(arguments: List[str]) -> List[str]:
    """
    Sends a chat completion request to OpenAI. The request consists of
    a system message and alternating user and assistant messages.

    Args:
        arguments: A list of strings representing the chat completion
                   request.
                   The first message should be a system message.
                   Alternating user and assistant messages should
                   follow it.

    Returns:
        A list of strings representing the response to the
        conversation. The response consists of alternating system and
        assistant messages.
    Raises:
        ProtocolException: Argument passed included less than two
                           messages.
    """
    if len(arguments) < 2:
        raise ProtocolException("Expect at least system and user messages.")
    else:
        messages: List[Union[UserMessage, AssistantMessage]] = []
        for i, text in enumerate(arguments[1:]):
            if i % 2 == 0:
                messages.append(UserMessage(text))
            else:
                messages.append(AssistantMessage(text))
        response = __send_impl(SystemMessage(arguments[0]), messages)
        result: List[str] = []
        for item in response:
            if isinstance(item, Message):
                result.append(item.role)
                result.append(item.text)
            else:
                result.append(item)
        return result


def __send_impl(system_message: SystemMessage, messages: List[Union[UserMessage, AssistantMessage]]) -> List[Union[str, Message]]:
    dict_messages = [system_message.to_dictionary()] + [message.to_dictionary()
                                                        for message in messages]
    dictionary_of_response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=dict_messages,
    )
    choices: List[Dict[str, Any]] = dictionary_of_response.get('choices')
    response: List[Union[str, Message]] = []
    if choices and len(choices) > 0:
        for choice in choices:
            finish_reason: Optional[str] = choice.get('finish_reason')
            dictionary_of_message: Optional[Dict[str, str]] = choice.get(
                'message')
            if dictionary_of_message:
                role = dictionary_of_message.get('role')
                content = dictionary_of_message.get('content')
                if finish_reason and role and content:
                    response.append(finish_reason)
                    response.append(build_message(role, content))
                else:
                    raise ProtocolException(
                        'missing or empty: finish_reason, role or content')
            else:
                raise ProtocolException('missing or empty message')
        return response
    else:
        raise ProtocolException('choices key missing or empty')


def main() -> None:
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        print("Please set OPENAI_API_KEY environment variable.", file=sys.stderr)
        exit(1)
    else:
        openai.api_key = api_key

    pyservice.register(
        "send",
        send,
        Metadata(
            name='send',
            description='Sends a chat completion request to the OpenAI service.',
            timeout=pyservice.Timeout.LONG,
            arguments=Arguments.variable_length(Argument(
                "Message", '''Message to the *system* and then alternating
                *user* and *assistant* messages''')),
            returns='A list of strings representing the response to the conversation.',
            errors='*ProtocolException* - Argument passed included less than two messages.'))
    pyservice.service_main()


if __name__ == '__main__':
    main()
