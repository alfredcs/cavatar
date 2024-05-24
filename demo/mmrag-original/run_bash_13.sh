#!/bin/bash
if [[ -n alfred ]]; then
    . /home/alfred/.bashrc
fi
/home/alfred/anaconda3/condabin/conda run -n dui /home/alfred/anaconda3/envs/dui/bin/streamlit run /home/alfred/multimodel/Gemini_Pro_Streamlit_Dashboard/home_13.py --server.port 7868 
