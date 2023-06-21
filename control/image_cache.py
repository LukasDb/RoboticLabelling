import streamlit as st


class ImageCache:
    def __init__(self):
        if "cached_images" not in st.session_state:
            st.session_state.cached_images = []
        self._captured_images = st.session_state.cached_images

    def clear_cache(self):
        self._captured_images.clear()
