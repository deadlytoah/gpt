import openai
import os
import subprocess
import sys
import zmq

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

OPENAI_MODEL = 'gpt-3.5-turbo'

class Message:
    """
    Represents a message in the chat.

    Use the `build_message` function to create a message of the
    appropriate role.

    :param role: The role of the message, which can be one of "system", "user", or "assistant".
    :type role: str
    :param text: The text content of the message.
    :type text: str
    """
    def __init__(self, role: str, text: str):
        self.role = role
        self.text = text

    def to_dictionary(self) -> Dict[str, str]:
        """
        Converts the message to a dictionary containing the message
        role and text content.

        Returns:
            A dictionary containing the message role and text content.
        """
        return {"role": self.role, "content": self.text}

class SystemMessage(Message):
    """Represents a system message."""
    def __init__(self, text: str):
        super().__init__("system", text)

class UserMessage(Message):
    """Represents a user message."""
    def __init__(self, text: str):
        super().__init__("user", text)

class AssistantMessage(Message):
    """Represents an assistant message."""
    def __init__(self, text: str):
        super().__init__("assistant", text)

def build_message(role: str, content: str) -> Message:
    """
    Returns a new instance of a message object that matches the given
    role and contains the provided content.

    Args:
        role (str): The role of the message, which can be 'system',
        'user', or 'assistant'.
        content (str): The content of the message.

    Returns:
        Message: A new instance of message with the given role that
        contains the provided content.

    Raises:
        ValueError: If the role provided is invalid.
    """
    if role == 'system':
        return SystemMessage(content)
    elif role == 'user':
        return UserMessage(content)
    elif role == 'assistant':
        return AssistantMessage(content)
    else:
        raise ValueError(f"Invalid role: {role}")

class ProtocolException(Exception):
    """
    An exception that indicates unexpected data format in the external
    API request or response.

    Attributes:
        message (str): The error message associated with the
        exception.
    """
    def __init__(self, message: str):
        """
        Initializes a new instance of the ProtocolException class.

        Args:
            message (str): The error message associated with the
            exception.
        """
        super(ProtocolException, self).__init__(message)

class ErrorCode(Enum):
    UNKNOWN_COMMAND = "ERROR_UNKNOWN_COMMAND"
    UNCATEGORISED = "ERROR_UNCATEGORISED"

class State(Enum):
    SENDING = 0
    RECEIVING = 1

class StateException(Exception):
    def __init__(self, state):
        self.state = state

def ok(socket, array):
    socket.send_multipart([b"OK"] + [arg.encode() for arg in array])

def error(socket, code, message):
    socket.send_multipart([b"ERROR", code.value.encode(), message.encode()])

def list_commands(arguments: List[str]) -> List[str]:
    return list(command_map().keys())

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
    dict_messages = [system_message.to_dictionary()] + [message.to_dictionary() for message in messages]
    dictionary_of_response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=dict_messages,
    )
    choices: List[Dict[str, Any]] = dictionary_of_response.get('choices')
    response: List[Union[str, Message]] = []
    if choices and len(choices) > 0:
        for choice in choices:
            finish_reason: Optional[str] = choice.get('finish_reason')
            dictionary_of_message: Optional[Dict[str, str]] = choice.get('message')
            if dictionary_of_message:
                role = dictionary_of_message.get('role')
                content = dictionary_of_message.get('content')
                if finish_reason and role and content:
                    response.append(finish_reason)
                    response.append(build_message(role, content))
                else:
                    raise ProtocolException('missing or empty: finish_reason, role or content')
            else:
                raise ProtocolException('missing or empty message')
        return response
    else:
        raise ProtocolException('choices key missing or empty')

def command_map() -> Dict[str, Callable[[List[str]], List[str]]]:
    return {
        "help": list_commands,
        "send": send,
    }

def main() -> None:
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        print("Please set OPENAI_API_KEY environment variable.", file=sys.stderr)
        exit(1)
    else:
        openai.api_key = api_key

    context: zmq.Context = zmq.Context()

    # Create a socket for the server
    socket: zmq.Socket = context.socket(zmq.REP)
    socket.bind("tcp://*:0")

    # Print the port number to stdout
    port_bytes = socket.getsockopt(zmq.LAST_ENDPOINT)
    assert(isinstance(port_bytes, bytes))
    port: str = port_bytes.decode().rsplit(":", 1)[-1]
    print(port)
    subprocess.call(f'/bin/echo -n {port} | pbcopy', shell=True)

    state: State = State.RECEIVING

    while True:
        try:
            # Wait for a request from a client
            if state == State.RECEIVING:
                message = socket.recv_multipart()
                state = State.SENDING
            else:
                raise StateException(state)

            command = message[0].decode()
            arguments = [arg.decode() for arg in message[1:]]

            print("received command", command, file=sys.stderr)

            # Process the request
            if command in command_map():
                response = command_map()[command](arguments)

                # Send the response back to the client
                if state == State.SENDING:
                    ok(socket, response)
                    state = State.RECEIVING
                else:
                    raise StateException(state)
            else:
                if state == State.SENDING:
                    error(socket, ErrorCode.UNKNOWN_COMMAND, "unknown command")
                    state = State.RECEIVING
                else:
                    raise StateException(state)

        except KeyboardInterrupt:
            break
        except StateException as e:
            print("Illegal state: ", e.state, file=sys.stderr)
            exit(1)
        except Exception as e:
            # Handle any errors that occur during processing
            error_response = str(e)
            if state == State.SENDING:
                error(socket, ErrorCode.UNCATEGORISED, error_response)
                state = State.RECEIVING
            else:
                print("Illegal state: ", state, file=sys.stderr)
                print("While trying to respond with error message: ", error_response, file=sys.stderr)

if __name__ == '__main__':
    main()
