import streamlit as st
import replicate
import base64

st.title("🪞 Age Transformer")

tab1, tab2 = st.tabs(["Anteprima", "Applicazione"])

with tab1:
    col1, col2 = st.columns([1,1])

    with col1:
        st.image('start_image.jpg')
    with col2:
        st.image('transformation.gif')

with tab2:

    # input_method = st.radio("Input method", ["📷 Camera", "🖼 Upload photo"], horizontal=True)

    image_data = None

    # if input_method == "📷 Camera":
    camera_photo = st.camera_input("Scatta un'immagine da modificare")
    if camera_photo:
        image_data = camera_photo.getvalue()
    # else:
    #     uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
    #     if uploaded:
    #         image_data = uploaded.getvalue()

    target_age = st.slider("Età voluta", min_value=10, max_value=100, value=70, step=5)

    if st.button("Trasforma", disabled=(image_data is None)):
        b64 = base64.b64encode(image_data).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64}"

        with st.spinner("Trasformazione in corso... tempo stimato ~12 secondi"):
            try:
                output = replicate.run(
                    "yuval-alaluf/sam:9222a21c181b707209ef12b5e0d7e94c994b58f01c7b2fec075d2e892362f13c",
                    input={"image": data_uri, "target_age": str(target_age)},
                )
                result_bytes = output.read()

                st.subheader("Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_data, caption="Età: attuale", use_container_width=True)
                with col2:
                    st.image(result_bytes, caption=f"Età: {target_age} anni", use_container_width=True)

                st.download_button(
                    label="Scarica il risultato",
                    data=result_bytes,
                    file_name=f"age_transformed_{target_age}.jpg",
                    mime="image/jpeg",
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")

    # if image_data is None:
    #     st.info("Cattura o carica un'immagine per iniziare.")