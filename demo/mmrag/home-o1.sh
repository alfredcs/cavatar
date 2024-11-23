#!/bin/bash
if [[ -n alfred ]]; then
    . /home/alfred/.bashrc
fi
/home/alfred/anaconda3/condabin/conda run -n dui /home/alfred/anaconda3/envs/dui/bin/streamlit run /home/alfred/demos/mmrag/home-o1.py --server.port 7867
