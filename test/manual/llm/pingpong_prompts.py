from typing import Dict


def build_prompts(user_prompt: str) -> Dict[str, str]:
    """
    返回每一轮要喂给模型的 prompt（字符串）。
    外部可以只改这里来调整 demo 行为。
    """
    p0 = f"""You are a helpful assistant.

TASK: Turn the following into a short demo story about "two model servers ping-ponging prompts".
Constraints:
- Keep it concrete and technical.
- Mention Ray actors and separate conda envs.
- Keep it under 120 words.

INPUT:
{user_prompt}
"""

    p1 = f"""Rewrite the INPUT into:
1) A 2-sentence English summary
2) Exactly 3 bullet points
Be concise. No extra text.

INPUT:
{p0}
"""

    # p2/p3/... 里会引用上一轮输出，所以这里只放“壳”
    return {
        "seed": p0,
        "r1_en_summary": p1,
        "r2_zh_title_tpl": """把下面内容翻译成中文，并添加一个简短标题。
要求：标题一行，正文保持原有结构（2句总结 + 3个要点）。

TEXT:
{prev}
""",
        "r3_ja_polite_tpl": """次の内容を日本語（丁寧語）に変換してください。
さらに末尾に「注意事項」を1行だけ追加してください（例：実行環境のnumpyを揃える等）。
形式は読みやすく。

TEXT:
{prev}
""",
        "r4_es_tweet_tpl": """Convierte el texto al español con estilo de tweet:
- Máx 2 tweets (separados por '---')
- Mantén términos técnicos (Ray, actor, conda env)
- Incluye 2 hashtags relevantes

TEXT:
{prev}
""",
        "r5_en_actions_tpl": """Convert the content into:
- Key takeaways (3 bullets)
- Action items (2 bullets)
English only, crisp.

TEXT:
{prev}
""",
        "r6_zh_spoken_tpl": """把下面英文内容改写成中文「口播脚本」：
- 适合30秒讲完
- 用【停顿】标记停顿
- 语气自然，面向工程师demo

TEXT:
{prev}
""",
        "r7_yaml_tpl": """Turn this into YAML with keys:
title, context, steps (list), risks (list), envs (dict with sglang, vllm)
Keep values short.

TEXT:
{prev}
""",
        "r8_final_tpl": """You are preparing a demo README snippet.
1) Write a short, polished final summary (<=120 words)
2) Then a plain-text comparison table with 3 rows:
   - Runtime env isolation
   - Prompt ping-pong depth
   - Output diversity
Columns: Feature | SGLang | vLLM

Use the YAML as the source of truth.

YAML:
{prev}
""",
    }
