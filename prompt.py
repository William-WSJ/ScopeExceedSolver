from langchain_core.prompts import PromptTemplate

direct_answer_prompt = PromptTemplate(
    input_variable=["question"],
    template="""
        请解答下列问题：
        {question}
    """
)

direct_answer_with_limitations_prompt = PromptTemplate(
    input_variable=["question", "limitations"],
    template="""
        请解答下列问题：
        {question}\n
        你的解答过程中严禁出现以下内容或方法：
        {limitations}
    """
)

generate_idea_and_solution_prompt = PromptTemplate(
    input_variable=["question", "idea"],
    template="""
        请根据我的解题思路解决以下问题：
        {question}\n
        我的解题思路如下：
        {idea}\n
        请注意，如果我的解题思路有明显的错误，请纠正后再解答。
    """
)

generate_idea_with_limitations_and_solution_prompt = PromptTemplate(
    input_variable=["question", "idea", "limitations"],
    template="""
        请根据我的解题思路解决以下问题：
        {question}\n
        我的解题思路如下：
        {idea}\n
        你的解答过程中严禁出现以下内容或方法：
        {limitations}\n
        请注意：
        1. 如果我的解题思路有明显的错误，请纠正后再解答。
        2. 如果我没有指明严禁出现的方法，请遵循第一条规则解答。
    """
)