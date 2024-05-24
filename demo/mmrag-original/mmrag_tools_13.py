import requests
from PIL import Image
from bs4 import BeautifulSoup as Soup
import os, sys
import fitz
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
#import tabula
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken
from operator import itemgetter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import BedrockEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
#from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

#module_path = "../"
#sys.path.append(os.path.abspath(module_path))
#from claude_bedrock_13 import *
os.environ['AWS_PROFILE'] = 'default'
os.environ['AWS_DEFAULT_REGION'] = region = 'us-west-2'
module_path = "../"
sys.path.append(os.path.abspath(module_path))
from utils import bedrock
from claude_bedrock_13 import *


boto3_bedrock = bedrock.get_bedrock_client(
    #assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

# --- parsinf --- #
def parse_tables_images(url:str):
    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = Soup(response.content, 'html.parser')

    # Find all table elements
    tables = soup.find_all('table')
    # Find all image elements
    images = soup.find_all('img')
    dir_p = url.split('/')[-1] 
    if len(dir_p) < 1:
        dir_p = url.split('/')[-2] 
    # Create a directory to store the tables
    os.makedirs(f'./{dir_p}/tables', exist_ok=True)
    os.makedirs(f'./{dir_p}/images', exist_ok=True)
    os.makedirs(f'./{dir_p}/summaries', exist_ok=True)
     # Save each table as an HTML file
    for i, table in enumerate(tables, start=1):
        table_html = str(table)
        #Creat table summary
        with open(f'./{dir_p}/summaries/summary_table_{i}.txt', 'w') as f:
            f.write(bedrock_textGen('anthropic.claude-3-sonnet-20240229-v1:0', 
                                    prompt='You are a perfect table reader and pay great attention to detail which makes you an expert at generating a comprehensive table summary in text based on this input:'+table_html, 
                                    max_tokens=2048, 
                                    temperature=0.01, 
                                    top_p=0.95, 
                                    top_k=40, 
                                    stop_sequences='Human:'))
        with open(f'./{dir_p}/tables/table_{i}.html', 'w', encoding='utf-8') as f:
            f.write(table_html)
     # Save each image to a file
    for i, image in enumerate(images, start=1):
        image_src = image.get('src')
        if image_src.startswith('http'):
            image_url = image_src
        else:
            base_url = '/'.join(url.split('/')[:3])
            image_url = f'{base_url}/{image_src}'

        try:
            image_data = requests.get(image_url).content
            image_byteio = Image.open(io.BytesIO(image_data))
            image_sum = bedrock_get_img_description('anthropic.claude-3-sonnet-20240229-v1:0', 
                                    prompt='You are an expert at analyzing images in great detail. Your task is to carefully examine the provided \
                                                image and generate a detailed, accurate textual description capturing all of key and supporting elements as well as \
                                                context present in the image. Pay close attention to any numbers, data, or quantitative information visible, \
                                                and be sure to include those numerical values along with their semantic meaning in your description. \
                                                Thoroughly read and interpret the entire image before providing your detailed caption describing the \
                                                image content in text format. Strive for a truthful and precise representation of what is depicted',
                                    image=image_byteio, 
                                    max_token=2048, 
                                    temperature=0.01, 
                                    top_p=0.95, 
                                    top_k=40, 
                                    stop_sequences='Human:')
            #print(f'{type(image_byteio)} and {image_sum}')
            if len(image_sum) > 1:
                with open(f'./{dir_p}/summaries/summary_image_{i}.txt', 'w') as f:
                    f.write(image_sum)
            with open(f'./{dir_p}/images/image_{i}.png', 'wb') as f:
                f.write(image_data)
        except Exception as e:
            print(f'Error saving image {i}: {e}')
        loader = DirectoryLoader(f'./{dir_p}/summaries', glob="**/*.txt")
    docs_sums = loader.load()
    return docs_sums

def parse_images_tables_from_pdf(pdf_path:str, output_folder:str):
    os.makedirs(output_folder, exist_ok=True)
    # Load text content
    loader = PyPDFLoader(pdf_path)
    text_splitter = CharacterTextSplitter(chunk_size=100000, chunk_overlap=1000)
    pdf_texts = loader.load_and_split(text_splitter)

    # Open the PDF file
    pdf_file = fitz.open(pdf_path)

    # Iterate through each page
    for page_index in range(len(pdf_file)):
        # Select the page
        page = pdf_file[page_index]

        # Search for tables on the page
        tables = page.find_tables()
        for table_index, table in enumerate(tables):
            # Save the table as a CSV file
            table_path = f"{output_folder}/table_{page_index}_{table_index}.csv"
            with open(table_path, "w", encoding="utf-8") as csv_file:
                for row in table:
                    csv_file.write(",".join([str(cell) for cell in row]) + "\n")
            print(f"Table saved: {table_path}")
        loader = DirectoryLoader(f"{output_folder}", glob='**/*.csv', loader_cls=CSVLoader)
        table_csvs = loader.load()
        
        # Search for images on the page
        images = page.get_images()
        for image_index, img in enumerate(images):
            # Get the image bounding box
            xref = img[0]
            image_info = pdf_file.extract_image(xref)
            image_data = image_info["image"]
            image_ext = image_info["ext"]

            # Save the image
            image_path = f"{output_folder}/image_{page_index}_{image_index}.{image_ext}"
            with open(image_path, "wb") as image_file:
                image_file.write(image_data)
            #print(f"Image saved: {image_path}")
            # Get image caption
            image_byteio = Image.open(io.BytesIO(image_data))
            try:
                image_sum = bedrock_get_img_description('anthropic.claude-3-sonnet-20240229-v1:0', 
                                        prompt='You are an expert at analyzing images in great detail. Your task is to carefully examine the provided \
                                                image and generate a detailed, accurate textual description capturing all of the important elements and \
                                                context present in the image. Pay close attention to any numbers, data, or quantitative information visible, \
                                                and be sure to include those numerical values along with their semantic meaning in your description. \
                                                Thoroughly read and interpret the entire image before providing your detailed caption describing the \
                                                image content in text format. Strive for a truthful and precise representation of what is depicted',
                                        image=image_byteio, 
                                        max_token=4096, 
                                        temperature=0.01, 
                                        top_p=0.95, 
                                        top_k=200, 
                                        stop_sequences='Human:')
                #print(f'{type(image_byteio)} and {image_sum}')
                if len(image_sum) > 1:
                    with open(f"{output_folder}/image_{page_index}_{image_index}.txt", 'w') as f:
                        f.write(image_sum)
            except:
                print(f"Fail to process {image_path}")
                pass
        '''
        # Search for charts on the page
        charts = page.get_charts()
        for chart_index, chart in enumerate(charts):
            # Get the chart bounding box
            area = chart.area
            rect = area.rect

            # Render the chart as an image
            chart_matrix = fitz.Matrix(rect.width / 72, 0, 0, rect.height / 72, rect.x, rect.y)
            pix = page.get_pixmap(matrix=chart_matrix)
            chart_data = pix.get_pixmap_data()

            # Save the chart as an image
            chart_path = f"{output_folder}/chart_{page_index}_{chart_index}.png"
            with open(chart_path, "wb") as chart_file:
                chart_file.write(chart_data)
            print(f"Chart saved: {chart_path}")

            # Get chart caption
            image_byteio = Image.open(io.BytesIO(image_data))
            image_sum = bedrock_get_img_description('anthropic.claude-3-sonnet-20240229-v1:0', 
                                    prompt='',
                                    image=image_byteio, 
                                    max_token=2048, 
                                    temperature=0.01, 
                                    top_p=0.95, 
                                    top_k=40, 
                                    stop_sequences='Human:')
            print(f'{type(image_byteio)} and {image_sum}')
            if len(image_sum) > 1:
                with open(f"{output_folder}/chart_{page_index}_{image_index}.txt", 'w') as f:
                    f.write(image_sum)
            '''

    # Close the PDF file
    pdf_file.close()
    
    loader = DirectoryLoader(f'./{output_folder}', glob="**/*.txt", loader_cls=TextLoader)
    text_splitter = CharacterTextSplitter(chunk_size=100000, chunk_overlap=1000)
    image_chart_sums = loader.load_and_split(text_splitter)
    pdf_texts.extend([*table_csvs, *image_chart_sums])
    return pdf_texts

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def combine_lists(nested_lists):
    return [element for sublist in nested_lists for element in sublist]

def extract_from_urls_or_pdf(urls: list, pdfs:list):
    all_docs = []
    if len(urls) > 0:
        for url in urls:
            loader = RecursiveUrlLoader(
                url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
            )
            docs = loader.load()
            sums = parse_tables_images(url)
            all_docs.append([*docs, *sums])
    elif len(pdfs) > 0:
        for pdf in pdfs:
            output_dir, ext = os.path.splitext(os.path.basename(pdf))
            sums_pdf = parse_images_tables_from_pdf(pdf, output_dir)
            all_docs.append([*sums_pdf])
    else:
        return all_docs
    new_docs = combine_lists(all_docs)
    #docs_texts = [d.page_content for d in new_docs]
    return new_docs

# ------ ETL ------#

def insert_into_chroma(docs, persist_directory, embd, chunk_size_tok:int, chunk_overlap:int):
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [doc.page_content for doc in d_reversed]
    )
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tok, chunk_overlap=chunk_overlap
    )
    texts_split = text_splitter.split_text(concatenated_content)
    persist_directory = persist_directory
    db = Chroma.from_texts(texts=texts_split, embedding=embd, persist_directory=persist_directory)
    # Make sure write to disk
    db.persist() 
    return True

# --------------  Retrial part ------------------- #
def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


def retrieval_from_chroma_fusion(retriever, question, model_id, max_tokens, temperature, top_k, top_p):
    model_kwargs =  { 
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": ["\n\nHuman"],
    }
    chat_claude_v3 = BedrockChat(model_id=model_id, model_kwargs=model_kwargs)
    
    # RAG-Fusion: Related
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Understand if the input query requires or implies multimodal search and output. \n
    Generate multiple search queries related to: {question} \n
    Output (6 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    
    generate_queries = (
        prompt_rag_fusion 
        | chat_claude_v3 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion 
    #docs = retrieval_chain_rag_fusion.invoke({"question": question})

    # RAG
    template = """Answer the following question based on this context:
    
    {context}
    
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion, 
         "question": itemgetter("question")} 
        | prompt
        | chat_claude_v3 #chat_openai# bedrock_llamav2 #_titan_agile
        | StrOutputParser()
    )
    
    return final_rag_chain.invoke({"question":question})

def retrieve_and_rag(retriever, chat_model, question,prompt_rag,sub_question_generator_chain):
    """RAG on each sub-question"""

    # Use our decomposition / 
    sub_questions = sub_question_generator_chain.invoke({"question":question})
    
    # Initialize a list to hold RAG chain results
    rag_results = []
    
    for sub_question in sub_questions:
        
        # Retrieve documents for each sub-question
        retrieved_docs = retriever.get_relevant_documents(sub_question)
        
        # Use retrieved documents and sub-question in RAG chain
        answer = (prompt_rag | chat_model| StrOutputParser()).invoke({"context": retrieved_docs, 
                                                                "question": sub_question})
        rag_results.append(answer)
    
        return rag_results,sub_questions

def format_qa_pairs(questions, answers):
    """Format Qa and A pairs"""
    
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

def retrieval_from_chroma_decompose(retriever, question, model_id, max_tokens, temperature, top_k, top_p):
    model_kwargs =  { 
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": ["\n\nHuman"],
    }
    chat_claude_v3 = BedrockChat(model_id=model_id, model_kwargs=model_kwargs)
    
    # Decomposition
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries semantically related to: {question} \n
    Output (5 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    # Chain
    generate_queries_decomposition = ( prompt_decomposition | chat_claude_v3 | StrOutputParser() | (lambda x: x.split("\n")))
    
    # Run
    questions = generate_queries_decomposition.invoke({"question":question})

    # RAG prompt
    prompt_rag = hub.pull("rlm/rag-prompt")
    
    
    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
    answers, questions = retrieve_and_rag(retriever, chat_claude_v3, question, prompt_rag, generate_queries_decomposition)

    context = format_qa_pairs(questions, answers)

    # Prompt
    template = """Here is a set of Q+A pairs:
    
    {context}
    
    Use these to synthesize an answer to the question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    final_rag_chain = (
        prompt
        | chat_claude_v3
        | StrOutputParser()
    )
    
    return final_rag_chain.invoke({"context":context,"question":question})
# ----- Mian ----- #

if __name__ == "__main__":
    urls = [
        "https://python.langchain.com/docs/expression_language/",
        "https://www.anthropic.com/news/claude-3-family",
    ]
    pdfs = [
        "../notebooks/pdfs/35-2-35.pdf",
        "../notebooks/pdfs/TSLA-Q4-2023-Update.pdf",
    ]
    chroma_pers_dir = "./chroma_03_122024"
    question = "How well does Claude 3's performance comparing with GPT-4 regarding Math benchmarks?"
    
    embd = embedding_bedrock = BedrockEmbeddings(client=boto3_bedrock, model_id="amazon.titan-embed-g1-text-02")
    #docs = extract_from_urls_or_pdf(urls, pdfs)
    #insert_into_chroma(docs, chroma_pers_dir, embd, 8190, 400)

    # REtrieval
    retriever  = Chroma(persist_directory=chroma_pers_dir, embedding_function=embd).as_retriever(search_kwargs={"k": 7})
    answer = retrieval_from_chroma_fusion(retriever, question, "anthropic.claude-3-sonnet-20240229-v1:0", 2048, 0.01, 250, 0.95)
    print(answer)
    print("-------------\n")
    answer = retrieval_from_chroma_decompose(retriever, question, "anthropic.claude-3-sonnet-20240229-v1:0", 2048, 0.01, 250, 0.95)
    print(answer)
    
