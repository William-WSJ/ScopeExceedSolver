from langchain_core.prompts import PromptTemplate

direct_answer_prompt = PromptTemplate(
    input_variable=["question"],
    template="""
        Please answer the following questions:
        {question}
    """
)

direct_answer_with_limitations_prompt = PromptTemplate(
    input_variable=["question", "limitations"],
    template="""
        Please answer the following questions:
        {question}\n
        Your solution process strictly prohibits the following content or methods:
        {limitations}
    """
)

generate_idea_prompt = PromptTemplate(
    input_variable=["question", "idea"],
    template="""
        Please answer the following questions based on my problem-solving path:
        {question}\n
        My solution path is as follows:
        {idea}\n
        Note that if there is a significant error in my solution path, please correct it before answering.
    """
)

generate_idea_with_limitations_prompt = PromptTemplate(
    input_variable=["question", "idea", "limitations"],
    template="""
        Please answer the following questions based on my problem-solving path:
        {question}\n
        My solution path is as follows:
        {idea}\n
        Your solution process strictly prohibits the following content or methods:
        {limitations}\n
        Note that if there is a significant error in my solution path, please correct it before answering.
    """
)

answer_check_prompt = PromptTemplate(
    input_variable=["question", "answer", "solution"],
    template="""
        Please help me determine if the answer to this question is consistent with the answer I provided.

        Question: {question}\n
        Correct Answer: {answer}\n
        My Answer: {solution}

        Note:
        - The correct answer and my answer may differ in form, so you need to carefully distinguish them before giving your response;
        - If my solution does not clearly provide an answer or contains garbled content unrelated to the question that affects the final judgment, please return False.
        
        Please do not return any other content, only return True or False. True indicates the answers are consistent, False indicates they are inconsistent.
    """
)

exceeds_scope_check_prompt = PromptTemplate(
    input_variables=["solution", "limitations"],
    template="""
        Please help me check if there are any scope exceedances in my solution process. The possible scope exceedances are listed below.
        Solution Process: {solution}\n
        Scope Exceedance List: {limitations}\n

        Requirements:
        - Please only return True or False. True indicates that the solution process contains content or methods from the scope exceedance list, and False indicates that it does not.
        - Please carefully inspect my solution process; any presence of content from the out-of-scope list constitutes a violation.
        - If my solution fails to provide a clear answer, or contains garbled/irrelevant content that impedes the final judgment, please return True.

        I emphasize once again: do not return any other content. Only return True or False, as defined above.
    """
)

idea_checklist = PromptTemplate(
    input_variables=["question", "idea", "acc", "solution"],
    template="""
        You shall act as an extremely rigorous expert in mathematics education assessment. Please conduct an in-depth evaluation of the Problem-Solving Thought Process (thought) by integrating the [Question], the provided [Full Solution], and its [Correctness Label].\n
        Evaluation Benchmarks:\n
        Question: {question} \n
        Full Solution: {solution} \n
        Solution Correctness: {acc} \n
        Thought Process to Be Evaluated: {idea} \n
        Output Requirements \n
        Return scoring results in JSON format. No additional text is permitted. The specified format is shown below: \n
        {\n
        ``Idea Score": integer (0-5),\n
        ``Key Points": integer (0-5),\n
        ``Guidance Ability": integer (0-10),\n
        ``Accuracy": integer (0-10)\n
        }\n
        """
)

# generate_idea_with_limitations_and_knowledge_prompt = PromptTemplate(
#     input_variable=["question", "idea", "limitations", "knowledge"],
#     template="""
#         Please solve the following problem based on my solution strategy and the solution templates in the knowledge base:
#         {question}\n
#         My solution strategy is as follows:
#         {idea}\n
#         The knowledge base template is as follows:
#         {knowledge}\n
#         Your solution process strictly prohibits the following content or methods:
#         {limitations}\n
#         Note that if there is a significant error in my solution strategy, please correct it before answering.
#     """
# )