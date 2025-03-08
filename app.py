import streamlit as st
from prediction_utils import ner_pipeline
from helper import Helper

label_colors = {
    "ACTIVITY OR CAUSE": "#FFA07A",
    "PLACE": "#20B2AA",
}
if "user_input" not in st.session_state:
    st.session_state.user_input = ""


def clear_text():
    st.session_state.user_input = ""


st.title("NERA - Your entity extractor")


st.sidebar.header("Entity Labels")
legend_md = Helper.generate_legend(label_colors)
st.sidebar.markdown(legend_md, unsafe_allow_html=True)


user_input = st.text_area("Enter text:", key="user_input", height=200)

if st.button("Predict"):
    if user_input:

        predictions = ner_pipeline.process(user_input)

        grouped_entities = Helper.group_entities_by_span(predictions)

        highlighted_text = Helper.highlight_entities(
            user_input, grouped_entities, label_colors
        )

        st.markdown(highlighted_text, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text for prediction.")

if st.button("Clear"):
    st.session_state.clear()
