# =================  (FULL CODE - UPDATED)
import streamlit as st
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
##Report
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO
import matplotlib.pyplot as plt

# ================= Openai key===========================================================================

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    USE_LLM = True
else:
    USE_LLM = False
    st.warning("API Key not found. Running without AI report.")

st.set_page_config(page_title="Advanced FEA Solver", layout="wide")
st.title("Advanced Solver with Plasticity")

# ================= INPUTS ================================================================================

st.sidebar.header("Material")
E = st.sidebar.number_input("Young’s Modulus (MPa)", value=210e3)
nu = st.sidebar.number_input("Poisson Ratio", min_value=0.0, max_value=0.5, value=0.3)
rho = st.sidebar.number_input("Density (kg/mm³)", value=7.85e-9, format='%.2e')

yield_strength = st.sidebar.number_input("Yield Strength (MPa)", value=250)
ultimate_strength = st.sidebar.number_input("Ultimate Strength (MPa)", value=450)
ultimate_plastic_strain = st.sidebar.number_input("Ultimate Plastic Strain", value=0.15)

st.sidebar.header("Geometry")
width = st.sidebar.number_input("Width (mm)", value=40)
depth = st.sidebar.number_input("Depth (mm)", value=5)
L = st.sidebar.number_input("Length (mm)", value=80)

A = width * depth
I = (width * depth**3) / 12

# ================= NEW LOAD INPUTS==============================================================
st.sidebar.header("Loading")
Fz = st.sidebar.number_input("Vertical Load Fz (N)", value=1000.0)

# Boundary Condition

st.sidebar.header("Boundary Condition")
bc = st.sidebar.selectbox("Support Type", ["Cantilever", "Simply Supported"])


# ================= TIME INPUT =================
st.sidebar.header("Time Parameters")
total_time = st.sidebar.number_input("Total Time (ms)", value=100.0)
time_steps = st.sidebar.number_input("Time Steps", min_value=10, value=50)

time = np.linspace(0, total_time, time_steps)

# ================= COMPUTATION =================
x = np.linspace(0, L, 50)

K = 800
n = 0.2

Fz_time = Fz * (time / total_time)

disp_time = []
vm_time = []
plastic_time = []
failure_flag = False

for Ft in Fz_time:

    if bc == "Cantilever":
        M = Ft * (L - x)
        disp = (Ft * x**2 * (3*L - x)) / (6 * E * I)
        V = Ft
    else:
        M = Ft * x * (L - x) / L
        delta_max = (Ft * L**3) / (48 * E * I)
        disp = delta_max * (4 * x * (L - x) / L**2)
        V = Ft / 2

    sigma_x = (M * (depth/2)) / I

    tau = (1.5 * V) / A
    tau_distribution = np.full(len(x), tau)

    vm = np.sqrt(sigma_x**2 + 3 * tau_distribution**2)

    plastic = np.zeros_like(vm)

    for i in range(len(vm)):
        if vm[i] > yield_strength:
            plastic[i] = ((vm[i] - yield_strength) / K) ** (1/n)

            if plastic[i] >= ultimate_plastic_strain:
                plastic[i] = ultimate_plastic_strain
                failure_flag = True

    disp_time.append(np.max(np.abs(disp)))
    vm_time.append(np.max(vm))
    plastic_time.append(np.max(plastic))

disp_time = np.array(disp_time)
vm_time = np.array(vm_time)
plastic_time = np.array(plastic_time)

# ================= RESULTS =================---------------------------------------------------------------------------
max_disp = np.max(disp_time)
max_vm = np.max(vm_time)
max_plastic = np.max(plastic_time)
# ================= SAFETY =================---------------------------------------------------------------------------
if failure_flag:
    st.error("❌ Failure: Ultimate plastic strain exceeded")

elif max_vm > ultimate_strength:
    st.error("❌ Failure: Ultimate strength exceeded")

elif max_vm > yield_strength:
        st.warning("⚠️ Plastic deformation")

else:
    st.success("✅ Elastic")

# ================= DISPLAY =================
st.subheader("Results")
c1, c2, c3, c4 = st.columns(4)
#c1.metric("Max Stress", f"{max_stress:.2e}")
c2.metric("Max Deflection", f"{max_disp:.6f}")
c3.metric("Max von Mises stress", f"{max_vm:.2e}")
c4.metric("Max Plastic Strain", f"{max_plastic:.2e}")

# ================= PLOTS =================---------------============================================================

# ================= DEFLECTION =================
st.subheader("Deflection vs Time")

df_disp = pd.DataFrame({
    "Time (ms)": time,
    "Deflection (mm)": disp_time
}).set_index("Time (ms)")

st.line_chart(df_disp)


# ================= VON MISES =================
st.subheader("von Mises Stress vs Time")

df_vm = pd.DataFrame({
    "Time (ms)": time,
    "von Mises Stress (MPa)": vm_time
}).set_index("Time (ms)")

st.line_chart(df_vm)


# ================= PLASTIC STRAIN =================
st.subheader("Plastic Strain vs Time")

df_plastic = pd.DataFrame({
    "Time (ms)": time,
    "Plastic Strain": plastic_time
}).set_index("Time (ms)")

st.line_chart(df_plastic)
# ==============================-Plot_Images============================================================
def create_plot_image(x, y, title, xlabel="Length", ylabel=None):
    buffer = BytesIO()
    plt.figure(figsize=(6,3))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else title)
    plt.grid()

    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return buffer
# ==============================
#  AI REPORT
# ==============================
####

def create_full_pdf(report):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    pdf_elements  = []       

    # ================= TITLE =================
    pdf_elements .append(Paragraph("<b>FEA Simulation Report</b>", styles["Title"]))
    pdf_elements .append(Spacer(1, 10))

    # ================= MATERIAL TABLE =================
    material_data = [
        ["Property", "Value"],
        ["Young's Modulus", str(E)],
        ["Poisson Ratio", str(nu)],
        ["Density", str(rho)],
        ["Yield Strength", str(yield_strength)],
        ["Ultimate Strength", str(ultimate_strength)]
    ]

    table = Table(material_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))

    pdf_elements .append(Paragraph("<b>Material Properties</b>", styles["Heading2"]))
    pdf_elements .append(table)
    pdf_elements .append(Spacer(1, 10))

    # ================= GEOMETRY =================

    pdf_elements .append(Paragraph("<b>Boundary Condition</b>", styles["Heading2"]))
    pdf_elements .append(Paragraph(f"{bc}", styles["Normal"]))
    pdf_elements .append(Spacer(1, 10))


    pdf_elements .append(Paragraph("<b>Geometry</b>", styles["Heading2"]))
    pdf_elements .append(Paragraph(f"Width: {width} mm, Depth: {depth} mm, Length: {L} mm", styles["Normal"]))
    pdf_elements .append(Spacer(1, 10))

    pdf_elements .append(Paragraph("<b>Loading</b>", styles["Heading2"]))
    pdf_elements.append(Paragraph(f"Fz: {Fz} N", styles["Normal"]))
    pdf_elements .append(Spacer(1, 10))


# ================= SAFETY =================

    pdf_elements.append(Paragraph("<b>Safety Assessment</b>", styles["Heading2"]))

    if failure_flag:
        safety_msg = "Failure (Ultimate plastic strain exceeded)"
    elif max_vm > ultimate_strength:
        safety_msg = "Failure (Exceeded Ultimate Strength)"
    elif max_vm > yield_strength:
        safety_msg = "Plastic Deformation (Yield Exceeded)"
    else:
        safety_msg = "Safe (Elastic Region)"

    pdf_elements.append(Paragraph(safety_msg, styles["Normal"]))
    pdf_elements.append(Spacer(1, 10))


    # ================= RESULTS =================
    pdf_elements .append(Paragraph("<b>Results</b>", styles["Heading2"]))
#    pdf_elements .append(Paragraph(f"Max Stress: {max_stress:.2e}", styles["Normal"]))
    pdf_elements .append(Paragraph(f"Max Deflection: {max_disp:.4f}", styles["Normal"]))
    pdf_elements .append(Paragraph(f"Max von Mises stress: {max_vm:.2e}", styles["Normal"]))
    pdf_elements .append(Paragraph(f"Max Plastic Strain: {max_plastic:.2e}", styles["Normal"]))
    pdf_elements .append(Spacer(1, 10))

    # ================= STRESS-STRAIN =================
    pdf_elements .append(Paragraph("<b>Stress-Strain Curve</b>", styles["Heading2"]))

# Use time-based values instead of last-step variables
    true_strain = np.log(1 + np.maximum(plastic_time, -0.99))
    true_stress = vm_time

    img_ss = create_plot_image(true_strain, true_stress, 
                            "True Stress vs True Strain",
                            xlabel="True Strain",
                            ylabel="True Stress (MPa)")

    pdf_elements .append(Image(img_ss, width=400, height=200))
    pdf_elements .append(Spacer(1, 10))

    # ================= PLOTS =================
    pdf_elements .append(Paragraph("<b>Plots</b>", styles["Heading2"]))

    img1 = create_plot_image(time, disp_time, "Deflection vs Time", xlabel="Time (ms)")
    img3 = create_plot_image(time, vm_time, "von Mises vs Time", xlabel="Time (ms)")
    img4 = create_plot_image(time, plastic_time, "Plastic Strain vs Time", xlabel="Time (ms)")

    pdf_elements .append(Image(img1, width=400, height=200))
 #   #pdf_elements .append(Image(img2, width=400, height=200))
    pdf_elements .append(Image(img3, width=400, height=200))
    pdf_elements .append(Image(img4, width=400, height=200))

    pdf_elements .append(Spacer(1, 10))

    # ================= AI REPORT =================
    pdf_elements .append(Paragraph("<b>AI Summary</b>", styles["Heading2"]))
    for line in report.split("\n"):
        if line.strip() != "":
            pdf_elements.append(Paragraph(line, styles["Normal"]))
            pdf_elements.append(Spacer(1, 6))

    # Build PDF
    doc.build(pdf_elements )

    buffer.seek(0)
    return buffer


######

if st.button("Generate AI Report"):

    prompt = f"""
    You are a senior CAE crash and structural analyst.

    Evaluate the following FEA results:

    Beam Type: {bc}

    Material:
    E = {E}, ν = {nu}, ρ = {rho}
    Yield Strength: {yield_strength}
    Ultimate Strength: {ultimate_strength}
    Ultimate Plastic Strain: {ultimate_plastic_strain}

    Geometry:
    Width: {width}, Depth: {depth}, Length: {L}

    Loading: Fz = {Fz} N

    Results:
    Max Deflection = {max_disp}
    Max von Mises stress = {max_vm}
    Plastic Strain = {max_plastic}

    Provide output EXACTLY in this format:

    ### Engineering Interpretation
    - point
    - point

    ### Failure Assessment
    - point
    - point

    ### Nonlinearity Insights
    - point
    - point

    ### Design Risks
    - point
    - point

    ### Recommendations
    - point
    - point

    Keep answers concise and professional.
    """

    if USE_LLM:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        report = response.choices[0].message.content
    else:
        report = "Basic report generated without AI."

    st.subheader("Report")
    st.markdown(report)

    # ✅ FULL CAE PDF EXPORT
    pdf_file = create_full_pdf(report)

    st.download_button(
        label="📥 Download Full FEA Report (PDF)",
        data=pdf_file,
        file_name="FEA_Report.pdf",
        mime="application/pdf"
    )