"""Prompt templates for benchmark evaluation"""

from typing import List

# Zero-shot prompt: just the question and options
ZERO_SHOT = """{question}

{options}

Answer with only the letter (A, B, C, D, E, F, G, H, I, or J). Do not explain."""

# Few-shot prompt with examples (supports multi-choice A-J)
FEW_SHOT_EXAMPLES = """Here are some examples:

Question: What is the chemical symbol for gold?
A. Go
B. Gd
C. Au
D. Ag
Answer: C

Question: Which planet is known as the Red Planet?
A. Venus
B. Mars
C. Jupiter
D. Saturn
Answer: B

"""

FEW_SHOT = FEW_SHOT_EXAMPLES + """Now answer this question:

{question}

{options}

Answer with only the letter. Do not explain."""

# Chain-of-thought prompt
COT = """{question}

{options}

Think carefully, then end your response with: ANSWER: X (where X is the letter)"""

# Chinese-style prompt
ZERO_SHOT_CN = """{question}

{options}

只回答选项字母(A-J)，不要解释。"""

# Chinese few-shot prompt with examples
FEW_SHOT_CN_EXAMPLES = """以下是一些示例：

问题：以下哪个是中国的首都？
A. 上海
B. 北京
C. 广州
D. 深圳
答案：B

"""

FEW_SHOT_CN = FEW_SHOT_CN_EXAMPLES + """请回答以下问题：

{question}

{options}

只回答选项字母，不要解释。"""

# Chinese Chain-of-thought prompt
COT_CN = """{question}

{options}

仔细思考，最后用"答案：X"的格式回答（X是选项字母）。"""

PROMPT_STYLES = {
    "zero_shot": ZERO_SHOT,
    "few_shot": FEW_SHOT,
    "cot": COT,
    "zero_shot_cn": ZERO_SHOT_CN,
    "few_shot_cn": FEW_SHOT_CN,
    "cot_cn": COT_CN,
}


def format_prompt(
    question: str,
    choices: List[str],
    style: str = "zero_shot",
) -> str:
    """Format a question into a prompt

    Args:
        question: The question text
        choices: List of choices (can be 4 or 10 options)
        style: Prompt style (zero_shot, few_shot, cot)

    Returns:
        Formatted prompt string
    """
    template = PROMPT_STYLES.get(style, ZERO_SHOT)

    # Format options with letters A, B, C, D, E, F, G, H, I, J
    options_text = "\n".join(
        f"{chr(ord('A') + i)}. {choice}"
        for i, choice in enumerate(choices)
    )

    return template.format(
        question=question,
        options=options_text,
    )
