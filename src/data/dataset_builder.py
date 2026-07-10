# ==============================================================================
# Cleaned CSVs → conversations JSON → prompt/response training examples.
# Context is selected with a time-gap heuristic: only messages close in time
# to the current one are included, and consecutive same-role messages are
# grouped into a single turn.
# ==============================================================================

import json
from datetime import datetime, timedelta

import pandas as pd

from src.prompts import TRAINING_SYSTEM_PROMPT
from src.tokens import SpecialTokens


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parses a '[DD/MM/YY, HH:MM:SS]' WhatsApp timestamp into a datetime."""
    cleaned_ts = timestamp_str.strip('[]').strip()
    return datetime.strptime(cleaned_ts, '%d/%m/%y, %H:%M:%S')


def process_conversation(csv_file: str, chat_id: int, assistant_name: str) -> dict:
    """Converts a cleaned chat CSV into {chat_id, messages} with user/assistant roles.

    Messages from assistant_name get the "assistant" role, everything else "user".
    """
    df = pd.read_csv(csv_file)
    messages = []

    for _, row in df.iterrows():
        role = "assistant" if row['Sender'] == assistant_name else "user"
        messages.append({
            "role": role,
            "content": row['Message'],
            "timestamp": row['Timestamp'],
        })

    return {"chat_id": chat_id, "messages": messages}


def create_dataset(csv_files: list, output_json: str, assistant_name: str) -> list:
    """Merges all chats in csv_files into a single conversations JSON file."""
    all_conversations = []

    for chat_id, csv_file in enumerate(csv_files, start=1):
        conversation = process_conversation(csv_file, chat_id, assistant_name)
        all_conversations.append(conversation)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)

    return all_conversations


def create_formatted_prompt(messages, current_user_msg_idx, tokens: SpecialTokens,
                            max_context_messages=5, time_gap_seconds=1800,
                            system_prompt=TRAINING_SYSTEM_PROMPT):
    """Builds the formatted prompt for one training example.

    The prompt starts with the system message, then the context window (messages
    within time_gap_seconds of the current one, up to max_context_messages) and
    finally the current user message. Consecutive messages with the same role are
    grouped with '\\n' as separator instead of re-using the role token.
    """
    prompt_parts = []

    if system_prompt:
        prompt_parts.append(f"{tokens.bos}{system_prompt}")

    current_msg = messages[current_user_msg_idx]
    current_msg_ts = parse_timestamp(current_msg['timestamp'])

    relevant_context_messages = []

    # iterate backwards from the message right before the current user message
    for i in range(current_user_msg_idx - 1, -1, -1):
        msg = messages[i]
        msg_ts = parse_timestamp(msg['timestamp'])
        time_diff = current_msg_ts - msg_ts

        if time_diff <= timedelta(seconds=time_gap_seconds) and len(relevant_context_messages) < max_context_messages:
            relevant_context_messages.insert(0, msg)  # keep chronological order
        else:
            # message too old or max context reached: stop
            break

    messages_to_process = relevant_context_messages + [current_msg]

    def role_token(role):
        return tokens.user_start if role == "user" else tokens.assistant_start

    # group consecutive same-role messages into a single turn
    current_role = None
    current_contents = []
    for msg in messages_to_process:
        if msg["role"] != current_role:
            if current_role is not None:
                prompt_parts.append(
                    role_token(current_role) + '\n'.join(str(c) for c in current_contents) + tokens.end_turn
                )
            current_role = msg["role"]
            current_contents = [msg["content"]]
        else:
            current_contents.append(msg["content"])

    if current_role is not None:
        prompt_parts.append(
            role_token(current_role) + '\n'.join(str(c) for c in current_contents) + tokens.end_turn
        )

    return "\n".join(prompt_parts) + f"\n{tokens.assistant_start}"


def create_dataset_list(json_file_path, tokens: SpecialTokens, max_context_messages=5,
                        time_gap_seconds=1800, system_prompt=TRAINING_SYSTEM_PROMPT):
    """Extracts {prompt, response} pairs from the conversations JSON.

    Only user → assistant pairs within the time gap become examples. Subsequent
    assistant messages sent within a quarter of the gap are merged into the
    response, mimicking the multi-message texting style.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset_list = []

    for conversation in data:
        messages = conversation['messages']

        if not messages:
            print(f"Warning: Conversation {conversation.get('chat_id', 'N/A')} is empty. Skipping.")
            continue

        for i in range(0, len(messages) - 1):
            current_message = messages[i]
            next_message = messages[i + 1]  # the immediate assistant response

            current_msg_ts = parse_timestamp(current_message['timestamp'])
            next_msg_ts = parse_timestamp(next_message['timestamp'])
            time_diff = next_msg_ts - current_msg_ts
            max_time_diff = timedelta(seconds=time_gap_seconds)

            if current_message['role'] == 'user' and next_message['role'] == 'assistant' and time_diff <= max_time_diff:

                prompt = create_formatted_prompt(
                    messages,
                    current_user_msg_idx=i,
                    tokens=tokens,
                    max_context_messages=max_context_messages,
                    time_gap_seconds=time_gap_seconds,
                    system_prompt=system_prompt,
                )

                response_parts = [next_message['content']]
                response_base_timestamp = parse_timestamp(next_message['timestamp'])

                # group subsequent assistant messages close in time to the base response
                for j in range(i + 2, len(messages)):
                    subsequent_msg = messages[j]

                    if subsequent_msg['role'] == 'assistant':
                        subsequent_msg_ts = parse_timestamp(subsequent_msg['timestamp'])
                        time_diff = subsequent_msg_ts - response_base_timestamp

                        # time gap divided by 4 for assistant responses
                        if time_diff <= timedelta(seconds=time_gap_seconds / 4):
                            response_parts.append(subsequent_msg['content'])
                        else:
                            break
                    else:
                        break

                response = '\n'.join(response_parts) + tokens.eos

                dataset_list.append({'prompt': prompt, 'response': response})

    return dataset_list


def build_splits(dataset_list, test_size=0.2, seed=42):
    """Creates HF train/val/test splits (80/10/10 with the default test_size)."""
    from datasets import Dataset  # imported lazily: only needed here

    dataset = Dataset.from_list(dataset_list)

    train_test = dataset.train_test_split(test_size=test_size, seed=seed)
    test_valid = train_test['test'].train_test_split(test_size=0.5, seed=seed)

    return train_test['train'], test_valid['train'], test_valid['test']
