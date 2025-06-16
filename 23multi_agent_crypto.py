# 환경 변수에서 API 키 가져오기
import os
from dotenv import load_dotenv

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# CrewAI 라이브러리에서 필요한 클래스 가져오기
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI
# Search Tool
from langchain_community.tools.tavily_search import TavilySearchResults
import gradio as gr


# LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

#search_tool = TavilySearchResults()
@tool("tavily_search_wrapper")
def tavily_search_wrapper(query: str) -> str:
    """Tavily 검색 도구 래퍼"""
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY)
    return search_tool.run(query)

def run_crypto_crew(topic):

    # Agent
    researcher = Agent(
    role='Market Researcher',
    goal=f'Uncover emerging trends and investment opportunities in the cryptocurrency market in 2024. Focus on the topic: {topic}.',
    backstory='Identify groundbreaking trends and actionable insights.',
    verbose=True,
    tools=[tavily_search_wrapper],
    allow_delegation=False,
    llm=llm,
    max_iter=3,
    max_rpm=10,
    )


    analyst = Agent(
    role='Investment Analyst',
    goal=f'Analyze cryptocurrency market data to extract actionable insights and investment leads. Focus on the topic: {topic}.',
    backstory='Draw meaningful conclusions from cryptocurrency market data.',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    )


    # Tasks
    research_task = Task(
    description=f'Explore the internet to pinpoint emerging trends and potential investment opportunities. Focus on the topic: {topic}.',
    agent=researcher,
    expected_output='A detailed summary of the reserch results in string format'
    )


    analyst_task = Task(
    description=f'Analyze the provided cryptocurrency market data to extract key insights and compile a concise report in Korean Hangul. Focus on the topic: {topic}.',
    agent=analyst,
    expected_output='A refined finalized version of the report in string format'
    )


    #`Crew` is a group of agents working together to accomplish a task
    crypto_crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analyst_task],
    process=Process.sequential  
    )

    #`kickoff` method starts the crew's process
    result = crypto_crew.kickoff()

    return result.raw    # raw text 속성을 출력 


def process_query(message, history):
    return run_crypto_crew(message)


if __name__ == '__main__':
    app = gr.ChatInterface(
        fn=process_query,
        type="messages",     
        title="Crypto Investment Advisor Bot",
        description="암호화폐 관련 트렌드를 파악하여 투자 인사이트를 제공해 드립니다."
    )

    app.launch()
