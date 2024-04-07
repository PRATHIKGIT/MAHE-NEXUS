import streamlit as st
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from IPython.display import display
from streamlit_cropperjs import st_cropperjs
import io
@st.cache_data
def get_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")

    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    return processor,model



def ocr_image(src_img):
    processor, model= get_model()
    pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def split_image_into_lines(image):
    lines = []
    img = Image.open(io.BytesIO(image))
    width, height = img.size
    line_height = 60  # adjust this value according to your image
    for y in range(0, height, line_height):
        line_box = (0, y, width, min(y + line_height, height))
        line = img.crop(line_box)
        lines.append(line)
    return lines

st.title("Out TrOCR App")
st.text("Upload a image which contains English text")

pic = st.file_uploader("Upload a picture", key="uploaded_pic")



if pic:
    pic = pic.read()


    if st.button("Extract Text"):
    # st.write("Extract Text")
    #     hw_image = show_image(croped)
    #         # display(hw_image)
        lines = split_image_into_lines(pic)

        for line in lines:
            # print(ocr_image(line))cd
            output_text = ocr_image(line)
            st.write(output_text)
