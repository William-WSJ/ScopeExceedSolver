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

generate_idea_prompt = PromptTemplate(
    input_variable=["question", "idea"],
    template="""
        请根据我的解题思路解决以下问题：
        {question}\n
        我的解题思路如下：
        {idea}\n
        请注意，如果我的解题思路有明显的错误，请纠正后再解答。
    """
)

generate_idea_with_limitations_prompt = PromptTemplate(
    input_variable=["question", "idea", "limitations"],
    template="""
        请根据我的解题思路解决以下问题：
        {question}\n
        我的解题思路如下：
        {idea}\n
        你的解答过程中严禁出现以下内容或方法：
        {limitations}\n
        请注意，如果我的解题思路有明显的错误，请纠正后再解答。
    """
)

answer_check_prompt = PromptTemplate(
    input_variable=["question", "answer", "solution"],
    template="""
        请帮我判断这个问题的答案是否与我给出的答案一致

        问题：{question}\n
        正确答案：{answer}\n
        我的解答：{solution}

        注意：
        - 正确答案和我的答案可能在形式上不一样，你需要仔细辨别后再给出回答；
        - 如果我的解答没有明确给出答案或者解答中出现乱码等与题目无关的内容且影响最终判断，则返回False。
        请不要返回任何其他内容，只返回True或者False即可。True表示答案一致，False表示答案不一致。
    """
)

exceeds_scope_check_prompt = PromptTemplate(
    input_variables=["solution", "limitations"],
    template="""
        请帮我检查一下我的解答过程中是否出现了超纲现象，可能出现的超纲现象我都写在列表中了。
        解答过程：{solution}\n
        超纲列表：{limitations}\n

        要求：
        - 请只返回True和False，True表示解答过程中出现了超纲列表的内容或方法，False表示没有出现。
        - 请仔细检查我的解答过程，但凡出现一点超纲列表中的内容都算超纲。
        - 如果我的解答没有明确给出答案或者解答中出现乱码等与题目无关的内容且影响最终判断，则返回True。

        再三强调，不要返回任何其他内容，只返回True或False即可，True和False的定义在上面有说明。
    """
)

idea_checklist = PromptTemplate(
    input_variables=["question", "idea", "solution"],
    template="""
        问题描述：{question}\n
        【解题思路开始】\n
        {idea}\n
        【解题思路结束】\n
        【完整解答开始】\n
        {solution}\n
        【完整解答结束】\n
        请依次完成对解题思路的得分评估：
        思路评估（0-5分）：从【逻辑完整性】【步骤连贯性】【表述清晰度】三维度打分
        关键点覆盖（0-5分）：判断是否触及核心公式/定理/解题突破口
        引导有效性（0-10分）：评估思路对构建解答的路径指引程度,不需要考虑正确性，考虑引导性即可 
        请严格返回JSON字符串,包含三个属性："思路评分":n,"关键点":n,"引导力":n，禁止任何额外内容
    """
)