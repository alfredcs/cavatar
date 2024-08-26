import os
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain.tools import DuckDuckGoSearchRun
import boto3
from langchain_aws import BedrockLLM, ChatBedrock, ChatBedrockConverse
from botocore.config import Config
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

load_dotenv()
search_tool = DuckDuckGoSearchRun()
web_rag_tool = WebsiteSearchTool()


## Setup LLMs
def get_llm(model_id):
    session = boto3.Session(profile_name='default')
    credentials = session.get_credentials()
    
    # Get the access key ID and secret access key
    access_key_id = credentials.access_key
    secret_access_key = credentials.secret_key
    aws_region = 'us-west-2'
    
    config = Config(
        retries = dict(
            max_attempts = 10,
            total_max_attempts = 25,
        )
    )
    
    bedrock_client = boto3.client("bedrock-runtime", config=config, 
                                  aws_access_key_id=access_key_id,
                                  aws_secret_access_key=secret_access_key,)

    inference_modifier = {
        "max_tokens": 4096,
        "temperature": 0.01,
        "top_k": 50,
        "top_p": 0.95,
        "stop_sequences": ["\n\n\nHuman"],
    }

    if 'claude-3-5' in model_id:
        inference_modifier = {
            "max_tokens": 4096,
            "temperature": 0.01,
            "top_k": 50,
            "top_p": 0.95,
            "stop_sequences": ["\n\n\nHuman"],
        }
        llm = ChatBedrock(
            model_id=model_id,
            client=bedrock_client,
            model_kwargs=inference_modifier,
            region_name=aws_region,
        ) 
    elif 'claude-3' in model_id or 'mistral' in model_id:
        llm = ChatBedrockConverse(
            model=model_id,
            client=bedrock_client,
            temperature=0.01,
            max_tokens=4096,
            region_name=aws_region,
        )
    elif 'llama3-1' in model_id:
        llm = ChatBedrockConverse(
            model=model_id,
            client=bedrock_client,
            temperature=0.01,
            max_tokens=4096,
            region_name=aws_region,
        )
    else:
        llm = BedrockLLM(
            model_id=model_id,
            client=bedrock_client,
            model_kwargs={"temperature": 0.1, "max_gen_len":4096},
        )  
    return llm

class blogAgents():
    def __init__(self, topic, model_id):
        self.topic = topic
        self.model_id = model_id
    
    def planner(self, topic, model_id):
        return Agent(
            role="Content Planner",
            goal=f"""lan engaging and factually accurate content on {topic}""", 
            backstory=f"""You're working on planning a blog article about the topic: {topic}. \n
                      You collect information by searhing the web for the latest developements that directly relate to the {topic}. \n
                      audience learn something and make informed decisions. Your work is the basis for the Content Writer to write an article on this {topic}.""",
            allow_delegation=False,
            tools=[search_tool, web_rag_tool],
            llm=get_llm(self.model_id),
            verbose=True
        )
        
    def writer(self, topic, model_id):
        return Agent(
            role="Content Writer",
            goal=f"Write insightful and factually accurate opinion piece about the topic: {topic}", 
            backstory=f"""You're working on a writing a new opinion piece about the topic: {topic}. You base your writing on the work of \n
                      the Content Planner, who provides an outline \n
                      and relevant context about the topic. \n
                      You follow the main objectives and \n
                      direction of the outline, \n
                      as provide by the Content Planner. \n 
                      You also provide objective and impartial insights \n 
                      and back them up with information \n
                      provide by the Content Planner. \n
                      You acknowledge in your opinion piece \n 
                      when your statements are opinions \n
                      as opposed to objective statements.""",
            allow_delegation=False,
            llm=get_llm(model_id),
            verbose=True
        )

    def editor(self, model_id):
        return Agent(
            role="Editor",
            goal="Edit a given blog post to align with "
                 "the writing style of the organization. ",
            backstory="You are an editor who receives a blog post from the Content Writer. "
                      "Your goal is to review the blog post to ensure that it follows journalistic best practices,"
                      "provides balanced viewpoints when providing opinions or assertions, "
                      "and also avoids major controversial topics or opinions when possible.",
            allow_delegation=False,
            llm=get_llm(model_id),
            verbose=True
        )


class blogTasks():
    def __init__(self, topic, model_id):
        self.topic = topic
        self.model_id = model_id

    def plan(self, planner, topic, model_id):  
        return Task(
            description=(
                f"""1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n
                  2. Identify the target audience, considering their interests and pain points.\n
                  3. Develop a detailed content outline including an introduction, key points, and a call to action.\n
                  4. Include SEO keywords and relevant data or sources."""
            ),
            expected_output=f"""Covert the latest developments of the {topic} with sufficient depth as a domain expert.
                A comprehensive content plan document with an outline, audience analysis,
                SEO keywords, and resources.""",
            agent=planner,
        )
    def write(self, writer, topic, model_id):  
        return Task(
            description=(
                f"""1. Use the content plan to craft a compelling blog post on {topic}.\n
                2. Incorporate SEO keywords naturally.\n
                3. Sections/Subtitles are properly named in an engaging manner.\n
                4. Ensure the post is structured with an engaging introduction, insightful body, and a summarizing conclusion.\n
                5. Proofread for grammatical errors and alignment with the brand's voice"""
            ),
            expected_output="A well-written blog post like a professional writer."
                "You are a domain expert and your blog is for other subject experts"
                "in markdown format, ready for publication, "
                "each section should have 2 or 3 paragraphs.",
            agent=writer,
        )
        
    def edit(self, editor, model_id):
        return Task(
            description=("Proofread the given blog post for "
                         "grammatical errors and "
                         "alignment with the brand's voice."),
            expected_output="A well-written blog post in markdown format, "
                            "ready for publication, "
                            "each section should have 2 or 3 paragraphs.",
            agent=editor,
            file_path='./blogPost.txt',
        )

class blogCrew():
  def __init__(self, topic, model_id):
    self.topic = topic
    self.model_id = model_id

  def run(self):
    agents = blogAgents(self.topic, self.model_id)
    tasks = blogTasks(self.topic, self.model_id)

    planner_agent = agents.planner(self.topic, self.model_id)
    writer_agent = agents.writer(self.topic, self.model_id)
    editor_agent = agents.editor(self.model_id)

    plan_task = tasks.plan(planner_agent, self.topic, self.model_id)
    write_task = tasks.write(writer_agent, self.topic, self.model_id)
    edit_task = tasks.edit(editor_agent, self.model_id)


    crew = Crew(
        agents=[planner_agent, writer_agent, editor_agent],
        tasks=[plan_task, write_task, edit_task],
        verbose=True,
        process=Process.sequential # Sequential process will have tasks executed one after the other and the outcome of the previous one is
    )

    result = crew.kickoff()
    return result