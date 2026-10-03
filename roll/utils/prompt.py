Qwen2_5_7B_Instruct_RLVR_Prompt = """
Given a problem, determine whether the final answer in the provided (incomplete) solution process matches the reference answer.  
The reference answer may be one single option character (e.g., A, B, C, D), a numerical value, an expression, or a list of answers if multiple questions are involved.  
**The reference answer may be in Chinese or another language, but your evaluation should be language-agnostic.**  

Your task:  
- Compare the final output of the solution process with the reference answer.  
- If they **match exactly**, output **YES**.  
- If they **do not match**, output **NO**.  
- If the solution process is unclear, incomplete, or ambiguous, assume it is incorrect and output **NO**.  

Your output must be strictly **'YES'** or **'NO'**, with no additional words, punctuation, or explanation.  

---

**Question:**  
{question}  

**Solution Process (Final Step Only):**  
{response}  

**Reference Answer:**  
{reference}  

**Output:**  
"""

LONGCOT_QWEN_2_5_SYSTEM = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"
# template name -> default system
MATH_DEFAULT_SYSTEMS = {
    "qwen2_5": "Please reason step by step, and put your final answer within \\boxed{}.",
}
CODE_DEFAULT_SYSTEMS = {
    "qwen2_5": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
}

BASE_CHAT_FORMAT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it."
    " The assistant first thinks about the reasoning process in the mind and then provides the user "
    "with the answer, ensuring that the final result in the answer is enclosed in \\boxed{{}}. The "
    "reasoning process and answer are enclosed within '<think>' '</think>' and '<answer>' '</answer>' "
    "tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    "User: {{content}} Assistant:"
)

# StateTree / LoCoMo binary judge (paper Appendix). Placeholders:
#   {question}, {gold_answer}/{reference}, {generated_answer}/{response}
StateTree_LoCoMo_Judge_Prompt = """\
Your task is to label an answer to a question as "CORRECT" or "WRONG".

You will be given the following data: (1) a question (posed by one user to another user), \
(2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user \
based on their prior conversations. The gold answer will usually be a concise and short answer \
that includes the referenced topic, for example:

Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace

The generated answer might be much longer, but you should be generous with your grading --- as \
long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The \
generated answer might be much longer or use relative time references (like 'last Tuesday' or \
'next month'), but you should be generous with your grading --- as long as it refers to the same \
date or time period as the gold answer, it should be counted as CORRECT. Even if the format \
differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.

Now it's time for the real question:

Question: {question}
Gold answer: {reference}
Generated answer: {response}

Return the label CORRECT or WRONG in a json format with the key as "label". Do NOT include both \
CORRECT and WRONG in your response, or it will break the evaluation script.
"""

prompt_maps = {
    "Qwen2.5-7B-Instruct-RLVR-prompt": Qwen2_5_7B_Instruct_RLVR_Prompt,
    "StateTree-LoCoMo-judge": StateTree_LoCoMo_Judge_Prompt,
}
