from fpdf import FPDF
import streamlit as st
import os
import cv2
import numpy as np
import tempfile
import tensorflow as tf
from PIL import Image
import time
import pandas as pd
import matplotlib.pyplot as plt
import base64

# Load the trained model
model_path = 'plasma_model.h5'
model = tf.keras.models.load_model(model_path)

def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

background_image_base64 = get_base64_of_image("static/images/photo8.jpg")

# Function to highlight cells on the image
def highlight_cells(image, mask):
    highlighted_image = image.copy()
    highlighted_image[mask == 1] = [255, 0, 0]  # Red in BGR
    highlighted_image[mask == 2] = [0, 0, 255]  # Blue in BGR
    highlighted_image[mask == 3] = [0, 0, 255]  # Purple in BGR
    return highlighted_image

# Function to create a PDF report
def create_pdf_report(patient_name, file_number, brown_count, blue_count, purple_count, positive_ratio, original_img_path, highlighted_img_path, bar_chart_path, pie_chart_path, interpretation):
    pdf = FPDF()
    try:
        pdf.add_font("Lora", "", "fonts/Lora-VariableFont_wght.ttf", uni=True)
        pdf.add_font("Lora", "B", "fonts/static/Lora-Bold.ttf", uni=True)
    except RuntimeError as e:
        st.error(f"Failed to load font: {e}")
        return None

    pdf.add_page()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Lora", 'B', 16)
    pdf.cell(200, 10, txt="Cell Analysis Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Lora", 'B', 14)
    pdf.cell(200, 10, txt="Patient Information", ln=True)
    
    pdf.set_text_color(0, 0, 125)
    pdf.set_font("Lora", size=12)
    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(200, 10, txt=f"File Number: {file_number}", ln=True)
    pdf.ln(10)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Lora", 'B', 14)
    pdf.cell(200, 10, txt="Analysis Results", ln=True)
    
    pdf.set_text_color(0, 0, 125)
    pdf.set_font("Lora", size=12)
    pdf.cell(200, 10, txt=f"Number of Brown Cells: {brown_count}", ln=True)
    pdf.cell(200, 10, txt=f"Number of Blue Cells: {blue_count + purple_count}", ln=True)
    pdf.cell(200, 10, txt=f"Percentage of Cells: {positive_ratio:.2f}%", ln=True)
    pdf.ln(10)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Lora", 'B', 14)
    pdf.cell(200, 10, txt="Interpretation:", ln=True)
    
    pdf.set_text_color(0, 0, 125)
    pdf.set_font("Lora", size=12)
    pdf.multi_cell(200, 10, txt=interpretation)
    pdf.ln(10)

    pdf.add_page()
    pdf.set_text_color(0, 0, 0) 
    pdf.set_font("Lora", 'B', 14)
    pdf.cell(200, 10, txt="Original Image:", ln=True)

    pdf.image(original_img_path, x=10, y=pdf.get_y(), w=180)
    pdf.ln(100)

    pdf.add_page()
    pdf.set_font("Lora", 'B', 14)
    pdf.cell(200, 10, txt="Image with Detected Cells:", ln=True)
    pdf.set_text_color(0, 0, 0)  
    pdf.image(highlighted_img_path, x=10, y=pdf.get_y(), w=180)
    pdf.ln(100)

    pdf.add_page()
    pdf.set_font("Lora", 'B', 14)
    pdf.cell(200, 10, txt="Bar Chart:", ln=True)
    pdf.set_text_color(0, 0, 0)  
    pdf.image(bar_chart_path, x=10, y=pdf.get_y(), w=180)
    pdf.ln(100)

    pdf.add_page()
    pdf.set_font("Lora", 'B', 14)
    pdf.cell(200, 10, txt="Pie Chart:", ln=True)
    pdf.set_text_color(0, 0, 0)  
    pdf.image(pie_chart_path, x=10, y=pdf.get_y(), w=180)
    pdf.ln(100)

    pdf_path = f"{patient_name}_report.pdf"
    pdf.output(pdf_path)
    return pdf_path

# Custom CSS to style the app with a watermark background
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{background_image_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        
    }}
    

    .stTextInput label, .stFileUploader label {{
        font-size: 24px !important;
        font-weight: bold !important;
    }}
    .stTextInput>div>div>input {{
        border-radius: 8px;
        border: 3px solid #adadad;  
        padding: 10px;
        font-size: 16px;
    }}

    #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-yw8pof.ekr3hml4 > div > div > div > div:nth-child(5) > div > section {{
        border-radius: 8px;
        border: 3px solid #adadad;  
        padding: 10px;
    }}
    .stFileUploader>div>div>div>div>div>div>div>div>div>span {{
        font-size: 18px !important;
        font-weight: bold !important;
    }}

    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #45a049;
    }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: #2c3e50;
    }}
    .stDeployButton {{
        display: none;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Application title and logo in the same row
col1, col2 = st.columns([1, 4])

# Add logo
logo_path = "static/images/Shar.png"
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    with col1:
        st.image(logo, width=1000)
else:
    st.warning("Logo image not found!")

# Add title
with col2:
    st.title("Clever Cell Scanner")

# Input patient data
patient_name = st.text_input("Patient Name:")
file_number = st.text_input("File Number:")

# Upload images
uploaded_files = st.file_uploader("Drop images here or click to upload.", type=["tiff", "png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(tmp_file_path), caption="Original Image", use_container_width=True)

        original_img_path = "original_image.jpg"
        cv2.imwrite(original_img_path, cv2.cvtColor(cv2.imread(tmp_file_path), cv2.COLOR_BGR2RGB))

        st.write("Analyzing the image...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        img = cv2.imread(tmp_file_path)
        resized_img = cv2.resize(img, (512, 512))
        hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
        blurred_img = cv2.GaussianBlur(hsv_img, (5, 5), 0)
        expanded_img = np.expand_dims(blurred_img, axis=0)

        pred_mask = model.predict(expanded_img)
        pred_mask = np.argmax(pred_mask, axis=-1)[0]

        pred_mask_resized = cv2.resize(pred_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        highlighted_img = highlight_cells(img, pred_mask_resized)

        highlighted_img_path = "highlighted_image.jpg"
        highlighted_img_rgb = cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(highlighted_img_path, highlighted_img_rgb)

        with col2:
            st.image(highlighted_img, caption="Image with Detected Cells", use_container_width=True)

        brown_count = np.sum(pred_mask_resized == 1)
        blue_count = np.sum(pred_mask_resized == 2)
        purple_count = np.sum(pred_mask_resized == 3)

        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
            if i == 50:
                status_text.text("We're almost done...")
        status_text.text("Analysis completed successfully!")

        st.write("### Analysis Results:")
        st.write(f"Number of Brown Cells: {brown_count}")
        st.write(f"Number of Blue Cells: {blue_count + purple_count}")

        total_cells = brown_count + blue_count + purple_count
        if total_cells > 0:
            positive_ratio = (brown_count) / total_cells * 100
            st.write(f"Percentage of Cells: {positive_ratio:.2f}%")
        else:
            st.write("No cells detected.")

        if positive_ratio < 10:
            interpretation = "Typically seen in normal bone marrow or in Monoclonal Gammopathy of Undetermined Significance (MGUS) if other factors support it."
        elif 10 <= positive_ratio < 60:
            interpretation = "Smoldering Myeloma (asymptomatic but concerning) or Needs Additional Criteria (CRAB Features: Calcium elevation, Renal dysfunction, Anemia, Bone lesions) to confirm active multiple myeloma."
        else:
            interpretation = "Definitive diagnosis of multiple myeloma based on the updated International Myeloma Working Group (IMWG) criteria, even without other myeloma-defining events."

        st.write("### Interpretation:")
        st.write(interpretation)

        st.write("### Comparison Between Healthy and Cancer Cells:")
        col1, col2 = st.columns(2)

        with col1:
            st.write("#### Bar Chart")
            data = {
                "Cell Type": ["Brown Cells", "Blue Cells"],
                "Count": [brown_count, blue_count + purple_count]
            }
            df = pd.DataFrame(data)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.bar(df["Cell Type"], df["Count"], color=['#8B4513', '#0000FF'])
            ax.set_title("Bar Chart")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            bar_chart_path = "bar_chart.png"
            plt.savefig(bar_chart_path, bbox_inches='tight')
            plt.close()

        with col2:
            st.write("#### Risk Level")
            if positive_ratio < 20:
                circle_color = '#02ce52'
                risk_label = "Low Risk"
            elif 20 <= positive_ratio < 60:
                circle_color = '#ffcb19'
                risk_label = "Medium Risk"
            else:
                circle_color = '#FF0000'
                risk_label = "High Risk"

            fig, ax = plt.subplots(figsize=(5, 5))
            circle = plt.Circle((0.5, 0.5), 0.4, color=circle_color, fill=False, linewidth=5)
            ax.add_artist(circle)
            ax.text(0.5, 0.5, f"{positive_ratio:.2f}%", ha='center', va='center', fontsize=20, color='black')
            ax.text(0.5, -0.2, f"Risk Level: {risk_label}", ha='center', va='center', fontsize=12, color=circle_color)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            st.pyplot(fig)
            pie_chart_path = "pie_chart.png"
            plt.savefig(pie_chart_path, bbox_inches='tight', transparent=True)
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)

        pdf_path = create_pdf_report(patient_name, file_number, brown_count, blue_count, purple_count, positive_ratio, original_img_path, highlighted_img_path, bar_chart_path, pie_chart_path, interpretation)

        st.markdown("<h3 style='text-align: center;'>Download Report</h3>", unsafe_allow_html=True)
        with open(pdf_path, "rb") as file:
            st.download_button(
                label="Download Report",
                data=file,
                file_name=f"{patient_name}_report.pdf",
                mime="application/pdf",
                key="download_button"
            )