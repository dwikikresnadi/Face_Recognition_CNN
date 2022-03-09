from img_classification import teachable_machine_classification
import streamlit as st
from PIL import Image
import requests
from streamlit_lottie import st_lottie
# config
st.set_page_config(
    page_title="Face Recognition",
    page_icon="🤡",
    layout="centered",
    menu_items={
        'About': "My Github Profile : " 'https://github.com/dwikikresnadi'
    }
)

# -- hidden
hide_st_style = """
        <style>
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ----------------------------------------------------------------------

# lottie function
def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
# variable for lottie icon
lottie_discord = load_lottieurl('https://assets1.lottiefiles.com/packages/lf20_b7zhs33r.json')
lottie_email = load_lottieurl('https://assets8.lottiefiles.com/packages/lf20_9yi1cm7i.json')
lottie_git = load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_kq6zs04j.json')
with st.container():
        st.title('🤡 Face Expression Recognition')
with st.container():
    st.write('---')
    uploaded_file = st.file_uploader("Choose a picture of mood ...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Picture.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = teachable_machine_classification(image, 'saved_model/best_model')
        if label == 0:
            st.subheader("Gambar menunjukkan ekspresi angry :rage:")
        elif label == 1:
            st.subheader("Gambar menunjukkan ekspresi happy :smile:")
        elif label == 2:
            st.subheader("Gambar menunjukkan ekspresi neutral :neutral_face:")
        else:
            st.subheader("Gambar menunjukkan ekspresi sad :disappointed:")
with st.container():
    st.write('---')
    st.title('Get in Touch with Me!')
    st.write('##')
    contact_icon, contact_dis = st.columns((0.05, 1))
    with contact_icon:
        dis = st_lottie(
            lottie_discord,
            height=50,
            key='discord',
            )
    with contact_dis:
        st.subheader(f"Naufal Dwiki #8342")
with st.container():
    contact_icons, contact_email = st.columns((0.05, 1))
    with contact_icons:
        st_lottie(
            lottie_email,
            height=50,
            key='email',
            )
    with contact_email:
        st.subheader("[naufaldwiki08@gmail.com](mailto:naufaldwiki08@gmail)")
with st.container():
    contact_iconss, contact_git = st.columns((0.05, 1))
    with contact_iconss:
        st_lottie(
            lottie_git,
            height=50,
            key='git',
            )
    with contact_git:
        st.subheader("[github.com/dwikikresnadi](https://github.com/dwikikresnadi)")