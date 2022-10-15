import streamlit as st
import base64
from main import main
import time
from argparse import ArgumentParser


def add_bg_from_local(image_file):
    """Set a background picture"""
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('love.png')


# parse arguments from command line
argparser = ArgumentParser()
argparser.add_argument('corpus', type=str, default='./corpus',
                       help='provide a route to your the directory with your corpus file (default: ./corpus)')
args = argparser.parse_args()

# streamlit app
st.title("L is for the way you look at me")
query = st.text_input('Введите запрос')
left_col, right_col = st.columns(2)
model_type = left_col.selectbox('Выберите метод векторизации', ['TF-IDF', 'BM25', 'BERT'])
if st.button('Поиск'):
    start = time.time()
    search_results = main(corpus_dir=args.corpus, query=query, model_type=model_type)
    end = time.time()
    st.markdown(f'*Время поиска: {round(end - start, 3)} секунд*')
    for res in search_results:
        st.markdown(f'* {res}')
