# ==============================================================================
# Special tokens used to format conversations for DeepSeek-R1-Distill-Qwen.
# Centralized here because they are needed by the data pipeline, inference
# and evaluation alike.
# ==============================================================================

from dataclasses import dataclass

# NOTE: we're using deepseek's special tokens for user and assistant roles, which have ｜ instead of |,
# don't confuse (｜|) this costed me 2hrs of bug fixing
USER_TOKEN_START = '<｜User｜>'
ASSISTANT_TOKEN_START = '<｜Assistant｜>'
END_TURN_TOKEN = "<|turn_end|>"

# fallbacks used only if the tokenizer does not define them
# NOTE: same thing here for deepseek's eos token, which is <｜end▁of▁sentence｜>, using also ▁ instead of _
DEFAULT_BOS_TOKEN = '<begin_of_sentence>'
DEFAULT_EOS_TOKEN = "<|end_of_sentence|>"


@dataclass(frozen=True)
class SpecialTokens:
    """Resolved set of special tokens for a given tokenizer."""
    bos: str
    eos: str
    user_start: str = USER_TOKEN_START
    assistant_start: str = ASSISTANT_TOKEN_START
    end_turn: str = END_TURN_TOKEN


def get_special_tokens(tokenizer) -> SpecialTokens:
    """Resolves BOS/EOS from the tokenizer, falling back to explicit defaults."""
    return SpecialTokens(
        bos=tokenizer.bos_token if tokenizer.bos_token else DEFAULT_BOS_TOKEN,
        eos=tokenizer.eos_token if tokenizer.eos_token else DEFAULT_EOS_TOKEN,
    )
