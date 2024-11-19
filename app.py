import os
import requests
from PIL import Image
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")

# Directory to save images
SAVE_FOLDER = "saved_images"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Streamlit App
st.set_page_config(page_title="Image Q&A App", layout="wide", initial_sidebar_state="collapsed")

# Application Header
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        font-weight: bold;
        margin-bottom: 30px;
    }
    </style>
    <div class="main-header">Interactive Image Q&A Application</div>
    """, unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Enter a valid image URL.  
2. Add one or more questions, separated by new lines.  
3. Click the **Submit** button to get answers!  
""")

# Input fields
st.subheader("Provide Image URL and Questions")

# User Inputs
image_url = st.text_input("Enter Image URL")
questions = st.text_area("Enter Questions (One question per line)")
submit = st.button("Submit")

if submit:
    answers = []
    image_path = None

    # Download and save the image
    try:
        st.info("Processing your request...")
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            image_name = os.path.basename(image_url.split("?")[0])
            image_path = os.path.join(SAVE_FOLDER, image_name)

            # Save the image
            with open(image_path, "wb") as f:
                f.write(response.content)

            # Load the image for display
            image = Image.open(image_path)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Process questions
            for question in questions:
                if question.strip():  # Skip empty questions
                    encoding = processor(image, question.strip(), return_tensors="pt")
                    outputs = model.generate(**encoding)
                    answer = processor.decode(outputs[0], skip_special_tokens=True)
                    answers.append((question, answer))

            # Display results
            st.subheader("Q&A Results")
            for idx, (q, a) in enumerate(answers, 1):
                st.markdown(f"**Q{idx}: {q}**")
                st.markdown(f"*Answer*: {a}")

        else:
            st.error("Failed to download the image. Please check the URL.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: #6c757d;
    }
    </style>
    <div class="footer">Made with ❤️ using Streamlit</div>
    """, unsafe_allow_html=True
)
