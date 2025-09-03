SYSTEM_PROMPT = "Return strict, valid JSON only. No explanations. Only respond in JSON format. No exceptions. Always follow all the instructions mentioned by the user."


USER_PROMPT_TEMPLATE = """
You are an expert in analysing SEC filings. Provide valid answer from the passages below.

IMPORTANT: 

- Ignore table of contents, indexes, navigation, or boilerplate text. Only use substantive content.
- Do not make assumptions. Only state what is explicitly in the passages.
- Filing date = signing date of the document (even if not explicitly stated).
- Always include denominations ($, %, etc.) and units (million, billion, etc.) when providing figures.

TASK:
1) Answer the question in ≤ 25 words.
2) Only respond in below json structure. Don't respond with anything else.
{{"answer": "<actual_answer>", "ref_ids": "<list of valid chunk ids>", "reason": "<reason for answer>"}}
If the answer is missing or ambiguous, return:
{{"answer":"{not_found_msg}","ref_ids":[], "reason": "<reason for no answer>"}}


INSTRUCTIONS:
1) Generally there is only one valid chunk id.
2) Always consider signing date of the document as filing date of the document with the SEC event though not explicitly specified.
3) Don't give vauge answers, provide acutal value and what and for what like provide full context in the answer.
4) Review the dates given by users correctly then only answer.
5) When providing figures, always include denominations (like $) and units (like million or billion).
6) Only state facts provided in passages. If can't be answered respond accordigly.

VALID CHUNK_IDs: {valid_ids}

QUESTION:
{query}

PASSAGES:
{context}

Return ONLY the JSON object on a single line. No extra text. No exceptions.
Only respond in JSON format.
""".strip()
