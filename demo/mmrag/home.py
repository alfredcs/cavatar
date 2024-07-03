import streamlit as st
from audio_recorder_streamlit import audio_recorder
import sys
import os
import io
from PIL import Image
from io import BytesIO
import base64
import time
import hmac

module_paths = ["./", "./configs"]
file_path = "./data/"
video_file_name = "uploaded_video.mp4"

for module_path in module_paths:
    sys.path.append(os.path.abspath(module_path))

from utility import *
from utils import *
from video_captioning import *
from anthropic_tools import *

st.set_page_config(page_title="Advanced RAG",page_icon="🩺",layout="wide")
st.title("Advanced RAG Demo")

aoss_host = read_key_value(".aoss_config.txt", "AOSS_host_name")
aoss_index = read_key_value(".aoss_config.txt", "AOSS_index_name")

# Password protection
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


# Check password
if not check_password():
    st.stop()  # Do not continue if check_password is not True.
    
#@st.cache_data
#@st.cache_resource(TTL=300)
with st.sidebar:
    #----- RAG  ------ 
    st.header(':green[Search RAG] :file_folder:')
    rag_search = rag_update = rag_retrieval = video_caption = image_caption = False
    rag_on = st.select_slider(
        'Activate RAG',
        value='None',
        options=['None', 'Search', 'Multimodal', 'Insert', 'Retrieval'])
    if 'Search' in rag_on:
        doc_num = st.slider('Choose max number of documents', 1, 8, 3)
        embedding_model_id = st.selectbox('Choose Embedding Model',('amazon.titan-embed-g1-text-02', 'amazon.titan-embed-image-v1'))
        rag_search = True
    elif 'Multimodal' in rag_on:
        upload_file = st.file_uploader("Upload your image/video here.", accept_multiple_files=False, type=["jpg", "png", "mp4", "mov"])
        if upload_file is not None:
            # Check if the uploaded file is an image
            try:
                bytes_data = upload_file.read()
                image =  (io.BytesIO(bytes_data))
                st.image(image)
                image_caption = True
            except:
                # Check if the uploaded file is a video
                video_bytes = upload_file.getvalue()
                with open(video_file_name, 'wb') as f:
                    f.write(video_bytes)
                st.video(video_bytes)
                video_caption = True
                pass
    elif 'Insert' in rag_on:
        upload_docs = st.file_uploader("Upload your doc here", accept_multiple_files=True, type=['pdf', 'doc', 'jpg', 'png'])
        # Amazon Bedrock KB only supports titan-embed-text-v1 not g1-text-02
        embedding_model_id = st.selectbox('Choose Embedding Model',('amazon.amazon.titan-embed-text-v1', 'amazon.titan-embed-image-v1'))
        rag_update = True
    elif 'Retrieval' in rag_on:
        rag_retrieval = True
                
    #----- Choose models  ------ 
    st.divider()
    st.title(':orange[Model Config] :pencil2:') 
    if 'Search' in rag_on:
        option = st.selectbox('Choose Model',('anthropic.claude-3-haiku-20240307-v1:0', 
                                              'anthropic.claude-3-sonnet-20240229-v1:0',
                                              'claude-3-5-sonnet-20240620'
                                             ))
    elif 'Video' in rag_on:
         option = st.selectbox('Choose Model',('anthropic.claude-3-haiku-20240307-v1:0', 
                                              'anthropic.claude-3-sonnet-20240229-v1:0'
                                             ))
    else:
        option = st.selectbox('Choose Model',('anthropic.claude-3-haiku-20240307-v1:0', 
                                              'anthropic.claude-3-sonnet-20240229-v1:0',
                                              'claude-3-5-sonnet-20240620',
                                              #'anthropic.claude-3-opus-20240229-v1:0',
                                              'meta.llama3-70b-instruct-v1:0',
                                              'finetuned:llama-3-8b-instruct'))
        
    st.write("------- Default parameters ----------")
    temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    max_token = st.number_input("Maximum Output Token", min_value=0, value=1024, step=64)
    top_p = st.number_input("Top_p: The cumulative probability cutoff for token selection", min_value=0.1, value=0.85)
    top_k = st.number_input("Top_k: Sample from the k most likely next tokens at each step", min_value=1, value=40)
    #candidate_count = st.number_input("Number of generated responses to return", min_value=1, value=1)
    stop_sequences = st.text_input("The set of character sequences (up to 5) that will stop output generation", value="\n\n\nHuman")

    # --- Audio query -----#
    st.divider()
    st.header(':green[Enable voice input]')# :microphone:')
    voice_on = st.toggle('Activate microphone')
    if voice_on:
        #record_audio_bytes = audio_recorder(icon_name="fa-solid fa-microphone-slash", recording_color="#cc0000", neutral_color="#666666",icon_size="2x",)
        record_audio_bytes = audio_recorder(text="",
                                            recording_color="#e8b62c",
                                            neutral_color="#6aa36f",
                                            icon_name="user",
                                            icon_size="6x",)
        if record_audio_bytes:
            st.audio(record_audio_bytes, format="audio/wav")#, start_time=0, *, sample_rate=None)
            with open(temp_audio_file, 'wb') as audio_file:
                audio_file.write(record_audio_bytes)
            if os.path.exists(temp_audio_file):
                voice_prompt = get_asr(temp_audio_file)
        st.caption("Press space and hit ↩️ for voice & agent activation")
        
    # ---- Clear chat history ----
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]
        record_audio = None
        voice_prompt = ""
        #del st.session_state[record_audio]


###
# Streamlist Body
###
start_time = time.time()
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

#
if rag_search:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        #Do retrieval
        #documents = search_and_convert(prompt, max_results=doc_num, filepath='pdfs')
        #documents = search_arxiv(prompt, max_results=doc_num, filepath='pdfs')
        #msg = retrieval_chroma(prompt, option, embedding_model_id, 6000, 600, max_token, temperature, top_p, top_k, doc_num
        #if 'simple general' in classify_query(prompt, 'simple general, others', 'anthropic.claude-3-haiku-20240307-v1:0'):
        
        #    msg, urls = serp_search(prompt, option, embedding_model_id, max_token, temperature, top_p, top_k, doc_num)

        ##
        # Use both search engines concurrently
        ##
        #def combine_documents(doc1: Document, doc2: Document) -> Document:
        #    combined_page_content = doc1.page_content + "\n\n" + doc2.page_content
        #    combined_metadata = {**doc1.metadata, **doc2.metadata}
        #    return Document(page_content=combined_page_content, metadata=combined_metadata)
            
        #with concurrent.futures.ThreadPoolExecutor() as executor:
        #    answer1 = executor.submit(google_search, prompt, num_results=doc_num)
        #    answer2 = executor.submit(tavily_search, prompt, num_results=doc_num)
        #    concurrent.futures.wait([answer1, answer2])
        #    docs1, urls =  answer1.result()
        #    docs2 = answer2.result()
        
        #combined_content = "\n\n".join([docs1.page_content, docs2['documents'].page_content])
        #documents = Document(
        #    page_content=docs1.page_content + "\n" + docs2['documents'].page_content,
        #    metadata={**doc1.metadata['source'], **docs['urls'][0:doc_num]}
        #)
        #documents = Document(page_content=combined_content)
        #urls += docs2['urls'][0:doc_num]

        # Tavily only
        docs =  tavily_search(prompt, num_results=doc_num)
        documents = docs['documents']
        urls = docs['urls'][0:doc_num]
        
        # Is Tavily search fails then try Google search
        if documents is None or len(urls) == 0:
            documents, urls = google_search(prompt, num_results=doc_num)
        
        if 'claude-3-5' in option:
            msg = retrieval_faiss_anthropic(prompt, documents, option, embedding_model_id, max_token, temperature, top_p, top_k, doc_num)
        else:
            msg = retrieval_faiss(prompt, documents, option, embedding_model_id, 6000, 600, max_token, temperature, top_p, top_k, doc_num)
        msg += "\n\n ✧***Sources:***\n\n" + '\n\n\r'.join(urls)
        msg += "\n\n ✒︎***Content created by using:*** " + option + f", Latency: {(time.time() - start_time) * 1000:.2f} ms" + f", Tokens In: {estimate_tokens(prompt, method='max')}, Out: {estimate_tokens(msg, method='max')}"
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🦙').write(msg)

elif video_caption:
    if "anthropic.claude-3" not in option:
        st.info("Please switch to a vision model")
        st.stop()
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        b64frames, audio_file = process_video(video_file_name, seconds_per_frame=2)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            answer1 = executor.submit(videoCaptioning_claude_nn, option, prompt, b64frames, max_token, temperature, top_p, top_k)
            answer2 =  executor.submit(get_asr, audio_file)
            captions, tokens = answer1.result()
            audio_transcribe = answer2.result()
    
        prompt2 = xml_prompt(captions, audio_transcribe, prompt)
        msg = bedrock_textGen(option, prompt2, max_token, temperature, top_p, top_k, stop_sequences)
        msg += "\n\n 🔊***Audio transcribe:*** " + audio_transcribe + "\n\n ✒︎***Content created by using:*** " + option + f", Latency: {(time.time() - start_time) * 1000:.2f} ms" + f", Tokens In: {tokens}+{estimate_tokens(prompt, method='max')}, Out: {estimate_tokens(msg, method='max')}"
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🦙').write(msg)

elif image_caption:
    if "claude-3" not in option:
        st.info("Please switch to a vision model")
        st.stop()
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        msg = bedrock_get_img_description(option, prompt, image, max_token, temperature, top_p, top_k, stop_sequences)
        width, height = Image.open(image).size
        tokens = int((height * width)/750)
        msg += "\n\n ✒︎***Content created by using:*** " + option + f", Latency: {(time.time() - start_time) * 1000:.2f} ms" + f", Tokens In: {tokens}+{estimate_tokens(prompt, method='max')}, Out: {estimate_tokens(msg, method='max')}"
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🦙').write(msg)
    
elif rag_update:        
    # Update AOSS
    if upload_docs:
        empty_directory(file_path)
        upload_doc_names = [file.name for file in upload_docs]
        for upload_doc in upload_docs:
            bytes_data = upload_doc.read()
            with open(file_path+upload_doc.name, 'wb') as f:
                f.write(bytes_data)
        stats, status, kb_id = bedrock_kb_injection(file_path)
        msg = f'Total {stats}  to Amazon Bedrock Knowledge Base: {kb_id}  with status: {status}.' + f", Latency: {(time.time() - start_time) * 1000:.2f} ms" 
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🎙️').write(msg)

elif rag_retrieval:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        #Do retrieval
        msg = bedrock_kb_retrieval(prompt, option)
        #msg = bedrock_kb_retrieval_advanced(prompt, option, max_token, temperature, top_p, top_k, stop_sequences)
        #msg = bedrock_kb_retrieval_decomposition(prompt, option, max_token, temperature, top_p, top_k, stop_sequences)
        msg += "\n\n ✒︎***Content created by using:*** " + option + f", Latency: {(time.time() - start_time) * 1000:.2f} ms" + f", Tokens In: {estimate_tokens(prompt, method='max')}, Out: {estimate_tokens(msg, method='max')}"
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🦙').write(msg)
        
else:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        if 'llama-3-8b-instruct' in option.lower():
            msg=tgi_textGen2('http://infs.cavatar.info:7861/', prompt, max_token, temperature, top_p, top_k)
        elif 'claude-3-5' in option:
            msg = anthropic_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences)
        elif 'generate imagetextGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences)' in classify_query(prompt, 'generate image, news, others', 'anthropic.claude-3-haiku-20240307-v1:0'):
            option = 'amazon.titan-image-generator-v1' #'stability.stable-diffusion-xl-v1:0' # Or 'amazon.titan-image-generator-v1'
            base64_str = bedrock_imageGen(option, prompt, iheight=1024, iwidth=1024, src_image=None, image_quality='premium', image_n=1, cfg=7.5, seed=452345)
            new_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
            st.image(new_image,width=512)#use_column_width='auto')
            msg = ' '
        else:
            msg=bedrock_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences)
        msg += "\n\n ✒︎***Content created by using:*** " + option + f", Latency: {(time.time() - start_time) * 1000:.2f} ms" + f", Tokens In: {estimate_tokens(prompt, method='max')}, Out: {estimate_tokens(msg, method='max')}"
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🦙').write(msg)
        
