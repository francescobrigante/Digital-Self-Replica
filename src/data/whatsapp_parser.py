# ==============================================================================
# WhatsApp .txt export → cleaned CSV.
# Removes metadata and noise (omitted media, encryption notices, calls),
# decodes escaped Latin-1 characters and normalizes whitespace.
# ==============================================================================

import csv
import re

# maps WhatsApp "omitted" attachment types to a natural-language placeholder
OMITTED_PLACEHOLDERS = {
    "audio": "*manda un audio*",
    "document": "*manda un documento*",
    "image": "*manda un'immagine*",
    "video": "*manda un video*",
    "sticker": "*manda uno sticker*",
    "contact": "*manda un contatto*",
}
FALLBACK_PLACEHOLDER = "*manda un allegato*"

# group 1: timestamp [DD/MM/YY, HH:MM:SS] — group 2: sender — group 3: message
MESSAGE_PATTERN = re.compile(
    r'^.*?(\[\d{2}/\d{2}/\d{2},\s*\d{2}:\d{2}:\d{2}\])\s*(.*?):\s*(.*)$'
)


def decode_latin1_escapes(text: str) -> str:
    """Converts special \\<hex> characters into the corresponding Latin-1 characters."""
    def repl(match):
        hex_str = match.group(1)
        return bytes.fromhex(hex_str).decode('latin-1')

    return re.sub(r"\\'([0-9a-fA-F]{2})", repl, text)


def clean_message(message: str, sender: str, assistant_name: str) -> str | None:
    """Cleans a single message; returns None if the message should be dropped.

    Messages with omitted content are dropped when sent by the assistant
    (nothing to learn from) and replaced with a placeholder otherwise, so the
    model still sees that the other person sent an attachment.
    """
    omitted_match = re.search(
        r'\b(audio|document|image|video|sticker|contact) omitted\b',
        message, re.IGNORECASE,
    )
    if omitted_match:
        if sender.lower() == assistant_name.lower():
            return None
        omitted_type = omitted_match.group(1).lower()
        message = OMITTED_PLACEHOLDERS.get(omitted_type, FALLBACK_PLACEHOLDER)

    if "Messages and calls are end-to-end encrypted" in message:
        return None
    if "Voice call." in message:
        return None

    # removes unfiltered special characters and decodes Latin-1 escapes
    message = re.sub(r'\\[a-z]+\d*', '', message)
    message = decode_latin1_escapes(message)

    # removes remaining backslashes (e.g., from original escapes that weren't Latin-1)
    message = message.replace("\\", "")

    # normalizes multiple spaces to a single space and removes artifacts
    message = re.sub(r'\s+', ' ', message).strip().rstrip('}')

    return message if message else None


def process_data(input_path, output_path, assistant_name, only_sender_name=None):
    """Reads the WhatsApp chat at input_path and saves cleaned messages to a CSV.

    Args:
        input_path: path to the exported WhatsApp .txt chat.
        output_path: destination CSV with columns [Timestamp, Sender, Message].
        assistant_name: sender whose messages will become the assistant role.
        only_sender_name: if set, keep only messages from this sender.
    """
    data = []  # processed [timestamp, sender, message] rows

    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the input file: {e}")
        return

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = MESSAGE_PATTERN.match(line)
        if not match:
            continue

        timestamp = match.group(1).strip()
        sender = match.group(2).strip()
        message = match.group(3).strip()

        if only_sender_name and sender.lower() != only_sender_name.lower():
            continue

        message = clean_message(message, sender, assistant_name)
        if message:
            data.append([timestamp, sender, message])

    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'Sender', 'Message'])
            writer.writerows(data)
        print(f"File saved successfully in: {output_path}")
    except IOError as e:
        print(f"Error: Could not write to output file {output_path}. {e}")
