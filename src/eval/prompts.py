"""Prompt templates for benchmark evaluation"""

from typing import List

# MMLU-Pro style prompt (recommended) - uses CoT format
MMLU_PRO = """The following are multiple choice questions (with answers) about {category}. Think step by step and then output the answer in the format of "The answer is (X)" at the end.

{examples}
{question}

{options}
Answer: Let's think step by step."""

# Zero-shot prompt: just the question and options
ZERO_SHOT = """{question}

{options}

Answer with only the letter (A, B, C, D, E, F, G, H, I, or J). Do not explain."""

# Few-shot prompt with examples
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

# Chain-of-thought prompt (MMLU-Pro style)
COT = """{question}

{options}

Think step by step and then output the answer in the format of "The answer is (X)" at the end."""

# Chinese-style prompts
ZERO_SHOT_CN = """{question}

{options}

只回答选项字母(A-J)，不要解释。"""

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

COT_CN = """{question}

{options}

请逐步思考，最后以"答案是(X)"的格式给出答案。"""

PROMPT_STYLES = {
    "zero_shot": ZERO_SHOT,
    "few_shot": FEW_SHOT,
    "cot": COT,
    "mmlu_pro": MMLU_PRO,
    "zero_shot_cn": ZERO_SHOT_CN,
    "few_shot_cn": FEW_SHOT_CN,
    "cot_cn": COT_CN,
}


def format_options(choices: List[str]) -> str:
    """Format options with letters A-J"""
    return "\n".join(
        f"{chr(ord('A') + i)}. {choice}"
        for i, choice in enumerate(choices)
    )


def format_prompt(
    question: str,
    choices: List[str],
    style: str = "zero_shot",
    category: str = "",
    examples: str = "",
) -> str:
    """Format a question into a prompt

    Args:
        question: The question text
        choices: List of choices (can be 4 or 10 options)
        style: Prompt style (zero_shot, few_shot, cot, mmlu_pro)
        category: Category name for mmlu_pro style
        examples: Few-shot examples for mmlu_pro style

    Returns:
        Formatted prompt string
    """
    template = PROMPT_STYLES.get(style, ZERO_SHOT)

    # Format options with letters A, B, C, D, E, F, G, H, I, J
    options_text = format_options(choices)

    return template.format(
        question=question,
        options=options_text,
        category=category,
        examples=examples,
    )
