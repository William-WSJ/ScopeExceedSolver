import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Enter your OpenAI API Key here
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()


test_template = PromptTemplate(
    input_variables=[],
    template="""
        Who are you?
    """
)
test_chain = test_template | llm | parser


def test_env():
    print(test_chain.invoke({}))



if __name__=="__main__":
    test_env()