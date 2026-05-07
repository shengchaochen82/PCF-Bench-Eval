def format_prompt(query, response, ground_truth):
    prompt = f'''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{str(query)}"""

Output sentence: """{str(response)}"""

Correct answer: {str(ground_truth)}

Judgement:
'''
    return prompt

def eval_func_xverify(item, llm):
    try:
        prompt = format_prompt(item['query'], item['response'], item['gt'])
        response = llm.inference({"query": prompt})['response']
        valid_label = ['correct', 'incorrect']
        
        if isinstance(response, str):
            item_label = response.strip().lower()
            if item_label in valid_label:
                if item_label == "correct":
                    return item_label, 1
                else:
                    return item_label, 0
            else:
                return f"Eval Error: {item_label}", None
        else:
            return "Eval Error: response is not a string", None
    except Exception as e:
        return f"Eval Error: {str(e)}", None