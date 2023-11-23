def format_speech_to_dialogue(speech_text):
    """
    Formats the given text into a dialogue format.

    Args:
        speech_text (str): The dialogue text to be formatted.

    Returns:
        str: Formatted text in dialogue format.
    """
    # Parse the given text appropriately
    dialog_list = eval(str(speech_text))
    dialog_text = ""

    for i, turn in enumerate(dialog_list):
        speaker = f"Speaker {i % 2 + 1}"
        text = turn['text']
        dialog_text += f"{speaker}: {text}\n"

    return dialog_text
