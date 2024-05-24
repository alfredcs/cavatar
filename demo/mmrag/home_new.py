import streamlit as st
import sys
import os

module_paths = ["./", "./configs"]
file_path = "./data/"

for module_path in module_paths:
    sys.path.append(os.path.abspath(module_path))

from utility import *

st.set_page_config(page_title="MM-RAG Demo",page_icon="🩺",layout="wide")
st.title("Multimodal RAG Demo")

@st.cache_data

aoss_host = read_key_value("./.aoss_config.txt", "AOSS_host_name")
aoss_index = read_key_value("./.aoss_config.txt", "AOSS_index_name")



#@st.cache_resource(TTL=300)
with st.sidebar:
    #----- RAG  ------ 
    st.header(':green[Multimodal RAG] :file_folder:')
    rag_update = rag_retrieval = False
    rag_on = st.select_slider(
        'Activate RAG',
        value='None',
        options=['None', 'Update', 'Retrieval'])
    if 'Update' in rag_on:
        upload_docs = st.file_uploader("Upload your doc here", accept_multiple_files=True, type=['pdf', 'doc', 'jpg', 'png'])
        # Amazon Bedrock KB only supports titan-embed-text-v1 not g1-text-02
        embedding_model_id = st.selectbox('Choose Embedding Model',('amazon.amazon.titan-embed-text-v1', 'amazon.titan-embed-image-v1'))
        rag_update = True
    elif 'Retrieval' in rag_on:
        rag_retrieval = True

    # --- Perplexity query -----#
    st.divider()
    st.header(':green[Perplexity] :confused:')
    #perplexity_on = st.toggle('Activate Perplexity query')
    perplexity_on = st.select_slider(
        'Activate answer and source query',
        value='Naive',
        options=['Naive', 'Text', 'Multimodal'])
    if 'Text' in perplexity_on or 'Multimodal' in perplexity_on:
        embd_model_id = st.selectbox('Choose Embedding Model',('amazon.amazon.titan-embed-text-v1', 'amazon.titan-embed-image-v1'))
        rag_on = "None"
    if 'Multimodal' in perplexity_on:
        upload_image = st.file_uploader("Upload your image here", accept_multiple_files=False, type=['jpg', 'png'])
        image_url_p = st.text_input("Or Input Image URL", key="image_url", type="default")
        if upload_image:
            bytes_data = upload_image.read()
            image =  (io.BytesIO(bytes_data))
            st.image(image)
        elif image_url_p:
            try:
                stream = fetch_image_from_url(image_url)
                st.image(stream)
                image = Image.open(stream)
            except:
                msg = 'Failed to download image, please check permission.'
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("ai").write(msg)
                
    #----- Choose models  ------ 
    st.divider()
    st.title(':orange[Multimodal Config] :pencil2:') 
    if 'Naive' not in perplexity_on:
        option = st.selectbox('Choose Model',('anthropic.claude-3-haiku-20240307-v1:0', 
                                          'anthropic.claude-3-sonnet-20240229-v1:0'))
    else:
        option = st.selectbox('Choose Model',('anthropic.claude-3-haiku-20240307-v1:0', 
                                              'anthropic.claude-3-sonnet-20240229-v1:0', 
                                              'anthropic.claude-instant-v1',
                                              'anthropic.claude-v2:1'))


    st.write("------- Default parameters ----------")
    temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    max_token = st.number_input("Maximum Output Token", min_value=0, value=1024, step=64)
    top_p = st.number_input("Top_p: The cumulative probability cutoff for token selection", min_value=0.1, value=0.85)
    top_k = st.number_input("Top_k: Sample from the k most likely next tokens at each step", min_value=1, value=40)
    #candidate_count = st.number_input("Number of generated responses to return", min_value=1, value=1)
    stop_sequences = st.text_input("The set of character sequences (up to 5) that will stop output generation", value="\n\nHuman")

        
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
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

#
if rag_update:
    # Update AOSS
    if upload_docs:
        empty_directory(file_path)
        upload_doc_names = [file.name for file in upload_docs]
        for upload_doc in upload_docs:
            bytes_data = upload_doc.read()
            with open(file_path+upload_doc.name, 'wb') as f:
                f.write(bytes_data)
        stats, status, kb_id = bedrock_kb_injection(file_path)
        msg = f'Total {stats}  to Amazon Bedrock Knowledge Base: {kb_id}  with status: {status}.'
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🎙️').write(msg)

elif rag_retrieval:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        #Do retrieval
        msg = bedrock_kb_retrieval(prompt, option)
        msg += "\n\n ✒︎Content created by using: " + option
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🦙').write(msg)
        
elif 'naive' not in perplexity_on.lower():
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
            
        if 'text' in perplexity_on.lower():
            msg,_ = bedrock_textGen_perplexity_memory(option, prompt, max_token, temperature, top_p, top_k, stop_sequences, embd_model_id)
        elif 'multimodal' in perplexity_on.lower():
            if image:
                prompt_msg = bedrock_get_img_description(option, prompt, image, max_token, temperature, top_p, top_k, stop_sequences)      
                keywords = extract_keywords(prompt_msg)
                search_prompt = '+'.join(str(item) for item in keywords)
                prompt = f"{prompt}::{search_prompt}"
                #prompt = f"{prompt}::{prompt_msg}"
            msg,_ = bedrock_imageGen_perplexity(option, prompt, max_token, temperature, top_p, top_k, stop_sequences, embd_model_id)
            #images= 
        elif "mistral.mistral-large" in option.lower() and 'text' in perplexity_on.lower():
            msg,_ = bedrock_textGen_perplexity_memory(option, prompt, max_token, temperature, top_p, top_k, stop_sequences, embd_model_id)
        else:
            msg = "Please choose a correct model."

        msg += "\n\n✒︎Content created by using: Perplexity query with " + option
        if "multimodal" in perplexity_on.lower() and image:
            left_co, cent_co,last_co = st.columns(3)
            with cent_co:
                st.image(image, width=512)
else:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        msg=bedrock_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences)
        msg += "\n\n ✒︎Content created by using: " + option
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🦙').write(msg)
        