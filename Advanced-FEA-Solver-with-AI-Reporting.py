# =================  (FULL CODE - UPDATED)
import streamlit as st
import numpy as np
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
st.sidebar.header("Mesh")
num_elements  = st.sidebar.number_input("Number of Elements", min_value=10, value=50)

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

# ================= COMPUTATION =================================================================
x = np.linspace(0, L, num_elements )



# =================  BENDING # =================================================================
if bc == "Cantilever":
    M = Fz * (L - x)
    disp = (Fz * x**2 * (3*L - x)) / (6 * E * I)

else:
    M = Fz * x * (L - x) / L
    disp = (Fz * x * (L**3 - 2*L*x**2 + x**3)) / (48 * E * I)

max_disp = np.max(np.abs(disp))


# ---------------- STRESS ----------------============================================================
y = depth / 2
sigma_x = (M * y) / I

# ---------------- SHEAR ----------------============================================================
V = Fz
tau = (1.5 * V) / A
tau_distribution = np.full(num_elements, tau)

# ---------------- STRAIN ----------------============================================================
strain_x = sigma_x / E


# von Mises
von_mises_stress = np.sqrt(sigma_x**2 + 3 * tau_distribution**2) 

# ================= PLASTIC ---------------============================================================
K = 800
n = 0.2

yield_strain = yield_strength / E

plastic_strain = np.zeros_like(von_mises_stress)
failure_flag = False

for i in range(len(von_mises_stress)):
    
    if von_mises_stress[i] > yield_strength:
        plastic_strain[i] = ((von_mises_stress[i] - yield_strength) / K) ** (1/n)
        
        # 🚨 FAILURE CONDITION
        if plastic_strain[i] >= ultimate_plastic_strain:
            plastic_strain[i] = ultimate_plastic_strain
            failure_flag = True


# ================= Mass---------------------------------------------------------------------------
volume = A * L
mass = rho * volume *10e-3
st.sidebar.write(f"Mass: {mass:.4f} tonne")

# ================= RESULTS =================---------------------------------------------------------------------------
max_vm = np.max(von_mises_stress)
max_stress = np.max(np.abs(sigma_x))
max_plastic = np.max(plastic_strain)

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
st.subheader("Deflection")
st.line_chart(disp)

#st.subheader("Stress")
#st.line_chart(sigma_x)

st.subheader("von Mises stress")
st.line_chart(von_mises_stress)

st.subheader("Plastic Strain")
st.line_chart(plastic_strain)

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

    elastic_strain = sigma_x / E
    strain = elastic_strain + plastic_strain

    true_strain = np.log(1 + strain)
    true_stress = sigma_x + E * plastic_strain

    img_ss = create_plot_image(true_strain, true_stress, 
                            "True Stress vs True Strain",
                            xlabel="True Strain",
                            ylabel="True Stress (MPa)")

    pdf_elements .append(Image(img_ss, width=400, height=200))
    pdf_elements .append(Spacer(1, 10))

    # ================= PLOTS =================
    pdf_elements .append(Paragraph("<b>Plots</b>", styles["Heading2"]))

    img1 = create_plot_image(x, disp, "Deflection")
#    img2 = create_plot_image(x, sigma_x, "Stress")
    img3 = create_plot_image(x, von_mises_stress, "von Mises")
    img4 = create_plot_image(x, plastic_strain, "Plastic Strain")

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