"""Fill the RM project template with the PINN project report (python-docx). Output: report_draft.docx (Word then
updates the table of contents and saves the final copy)."""
import re
from pathlib import Path

import docx
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

D = Path(__file__).parent
FIG = D / "figs"
doc = docx.Document(D / "template.docx")
TNR = "Times New Roman"
BLACK = RGBColor(0, 0, 0)

# ------------------------------------------------------------------ cover page
TITLE = ("Physics-Informed Neural Networks for Incompressible Flow: Benchmark Study and Non-Invasive "
         "Blood-Pressure Estimation from 4D Flow MRI")
DOMAIN = "Scientific Machine Learning for fluid dynamics and medical imaging"
STUDENTS = [("Mehul Rawat", "123B1C063"), ("Mantavya Agarwal", "123B1C020"), ("Ishwari Shendarkar", "124B2C009"),
            ("Atharva Palande", "123B1C060")]


def set_cover(p, text):
    r0 = p.runs[0]
    size, bold = r0.font.size, r0.bold
    for r in p.runs[1:]:
        r._element.getparent().remove(r._element)
    r0.text = text
    r0.font.size, r0.bold = size, bold


paras = doc.paragraphs
for p in paras[:3]:
    pPr = p._p.get_or_add_pPr()
    ol = OxmlElement("w:outlineLvl")
    ol.set(qn("w:val"), "9")
    pPr.append(ol)
for p in paras[:25]:
    t = p.text
    if t.startswith("Project Title:"):
        set_cover(p, f"Project Title: {TITLE}")
    elif t.startswith("Domain / Research Area:"):
        set_cover(p, f"Domain / Research Area: {DOMAIN}")
    elif t.startswith("Project Group No.:"):
        set_cover(p, "Project Group No.: 20")
    elif t.startswith("Student "):
        k = int(t.split()[1].rstrip(":")) - 1
        name, prn = STUDENTS[k]
        set_cover(p, f"Student {k + 1}: {name}\tPRN No: {prn}")
        p.paragraph_format.tab_stops.add_tab_stop(Inches(4.3))
    elif t.startswith("Internal Guide/Mentor:"):
        set_cover(p, "Internal Guide/Mentor: Puja Kate")
    elif t.startswith("External/Industry Guide"):
        set_cover(p, "External/Industry Guide (if applicable): Not applicable")
    elif t.startswith("Project Type:"):
        set_cover(p, "Project Type: In-house, multidisciplinary")
    elif t.startswith("SDG Mapped:"):
        set_cover(p, "SDG Mapped: SDG 3 (Good Health and Well-being), SDG 9 (Industry and Innovation)")
        sdg = p

# remove everything after the cover (the instructions page and the template's placeholder sections)
body = doc.element.body
started = False
for el in list(body):
    if el is sdg._element:
        started = True
        continue
    if started and el.tag != qn("w:sectPr"):
        body.remove(el)


# ------------------------------------------------------------------ helpers
def fmt_body(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=6, spacing=1.5):
    pf = p.paragraph_format
    pf.alignment = align
    pf.line_spacing = spacing
    pf.space_after = Pt(after)
    pf.space_before = Pt(0)


def add_runs(p, text, size=12, italic=False, font=TNR):
    """**bold** markup inside text."""
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if not part:
            continue
        bold = part.startswith("**")
        r = p.add_run(part[2:-2] if bold else part)
        r.font.name, r.font.size, r.bold, r.italic = font, Pt(size), bold, italic
        r._element.rPr.rFonts.set(qn("w:eastAsia"), font)
        r.font.color.rgb = BLACK
    return p


def para(text, **kw):
    p = doc.add_paragraph()
    fmt_body(p)
    return add_runs(p, text, **kw)


def label(text):
    """a bold run-in label paragraph, e.g. 'Problem Statement:'"""
    p = para(f"**{text}**")
    p.paragraph_format.keep_with_next = True
    return p


def bullets(items, size=12):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        fmt_body(p, after=3)
        add_runs(p, it, size=size)


def numbered(items):
    for k, it in enumerate(items, 1):
        p = doc.add_paragraph()
        fmt_body(p, after=3)
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        add_runs(p, f"{k}.  {it}")


def heading(text, level=1, page_break=True):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.page_break_before = page_break and level == 1
    p.paragraph_format.space_before = Pt(12 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(8 if level == 1 else 4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name, r.font.size, r.bold, r.font.color.rgb = TNR, Pt(16 if level == 1 else 13), True, BLACK
    r._element.rPr.rFonts.set(qn("w:eastAsia"), TNR)
    return p


def equation(text):
    p = doc.add_paragraph()
    fmt_body(p, align=WD_ALIGN_PARAGRAPH.CENTER, after=6, spacing=1.2)
    r = p.add_run(text)
    r.font.name, r.font.size, r.italic, r.font.color.rgb = "Cambria Math", Pt(11.5), False, BLACK
    return p


def code(lines):
    for k, ln in enumerate(lines):
        p = doc.add_paragraph()
        fmt_body(p, align=WD_ALIGN_PARAGRAPH.LEFT, after=0 if k < len(lines) - 1 else 8, spacing=1.0)
        p.paragraph_format.left_indent = Inches(0.3)
        r = p.add_run(ln if ln else " ")
        r.font.name, r.font.size, r.font.color.rgb = "Courier New", Pt(9.5), BLACK


fig_no = [0]
tab_no = [0]


LABELS = {}


def figure(path, caption, width=6.2, key=None):
    fig_no[0] += 1
    if key:
        LABELS[f'fig:{key}'] = fig_no[0]
    p = doc.add_paragraph()
    fmt_body(p, align=WD_ALIGN_PARAGRAPH.CENTER, after=2, spacing=1.0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG / path), width=Inches(width))
    c = doc.add_paragraph()
    fmt_body(c, align=WD_ALIGN_PARAGRAPH.CENTER, after=10, spacing=1.0)
    add_runs(c, f"**Figure {fig_no[0]}:** {caption}", size=10.5)


def set_cell_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), "000000")
        borders.append(e)
    tblPr.append(borders)


def table(rows, caption, widths=None, size=10, header_fill="D9D9D9", key=None):
    tab_no[0] += 1
    if key:
        LABELS[f"tab:{key}"] = tab_no[0]
    c = doc.add_paragraph()
    fmt_body(c, align=WD_ALIGN_PARAGRAPH.CENTER, after=4, spacing=1.0)
    c.paragraph_format.keep_with_next = True
    add_runs(c, f"**Table {tab_no[0]}:** {caption}", size=10.5)
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = doc.styles["Table Grid"]
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_cell_borders(t)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            fmt_body(p, align=WD_ALIGN_PARAGRAPH.LEFT, after=0, spacing=1.0)
            add_runs(p, str(val) if i else f"**{val}**", size=size)
            # short tables stay on one page
            p.paragraph_format.keep_with_next = len(rows) <= 16 and i < len(rows) - 1
            if widths:
                cell.width = Inches(widths[j])
            if i == 0:
                sh = OxmlElement("w:shd")
                sh.set(qn("w:val"), "clear")
                sh.set(qn("w:color"), "auto")
                sh.set(qn("w:fill"), header_fill)
                cell._tc.get_or_add_tcPr().append(sh)
    # the header row repeats on every page
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)
    sp = doc.add_paragraph()
    fmt_body(sp, after=6, spacing=1.0)
    return t


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def toc():
    """A TOC field whose stored result already lists the entries (read from Word's own layout of this document), so
    the report opens with a finished table of contents and Word can still refresh it (right-click, Update Field)."""
    import json
    from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER

    entries = json.load(open(D / "toc_entries.json")) if (D / "toc_entries.json").exists() else []

    def fld(r, kind):
        e = OxmlElement("w:fldChar")
        e.set(qn("w:fldCharType"), kind)
        r._r.append(e)

    if not entries:
        entries = [[1, "Right-click and choose Update Field to build the table of contents.", ""]]
    width = doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin
    for k, (lvl, text, page) in enumerate(entries):
        p = doc.add_paragraph()
        fmt_body(p, align=WD_ALIGN_PARAGRAPH.LEFT, after=3, spacing=1.15)
        p.paragraph_format.left_indent = Inches(0.3 * (lvl - 1))
        p.paragraph_format.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        if k == 0:
            r = p.add_run()
            fld(r, "begin")
            ins = OxmlElement("w:instrText")
            ins.set(qn("xml:space"), "preserve")
            ins.text = r'TOC \o "1-2" \h \z \u'
            r._r.append(ins)
            fld(r, "separate")
        add_runs(p, f"{text}	{page}", size=12 if lvl == 1 else 11.5)
        if lvl == 1:
            for r_ in p.runs:
                r_.bold = False
        if k == len(entries) - 1:
            fld(p.add_run(), "end")


# ------------------------------------------------------------------ preliminary pages
heading("Abstract", 1)
para("Pressure inside blood vessels and around engineered structures decides whether a narrowing is dangerous or a "
     "design is safe, yet it is the quantity that imaging cannot measure: 4D flow magnetic resonance imaging (MRI) "
     "measures blood velocity, while pressure across an aortic coarctation is still measured invasively with a "
     "catheter or estimated with the simplified Bernoulli rule, which is often off by 30% or more. Physics-informed "
     "neural networks (PINNs) can combine measured velocity with the Navier-Stokes equations and infer the pressure "
     "that no instrument recorded, but their training is fragile and their accuracy on real three-dimensional scans "
     "validated against catheters is rarely reported.")
para("This project implements a complete PINN framework in JAX/Flax and studies it in two stages. First, seven "
     "standard incompressible-flow benchmarks (Taylor-Green vortices in 2D and 3D, the lid-driven cavity, flow past a "
     "cylinder, an inverse cylinder-wake problem and a parametric PI-DeepONet) are trained with Fourier features, "
     "hard constraints, causal training, adaptive loss weighting and residual-based adaptive sampling, and every "
     "failure is attributed to a component through ablation. Three benchmarks pass their gates (2D Taylor-Green "
     "4.05e-5 error, the inverse wake with the viscosity recovered to 0.065%, the DeepONet at 9.65% on an unseen "
     "Reynolds number); the cavity, cylinder and 3D Taylor-Green failures are explained. Second, the method is applied "
     "to blood-pressure estimation: on simulated stenoses it reaches 6.2% and 4.1% peak-pressure error against "
     "21-31% for Bernoulli, and on a public 3D-printed aortic coarctation phantom scanned by a real MRI it was "
     "diagnosed, corrected (a separate periodic pressure network, a Smagorinsky eddy viscosity and the measured "
     "acceleration as data) and raised from 0.3 to 8.74 mmHg against the catheter's 13.1 mmHg, a 33% error that beats "
     "Bernoulli (39%) on the development case. Frozen and applied to all eight cases, however, the networks reach a "
     "42.0% mean error against Bernoulli's 30.4% (pressure Poisson 45.7%); read at the catheter's own marker locations "
     "they reach 43.5%. They fail most on the severe narrowings, whose narrow jets the scan blurs. Stage II trains on "
     "the acceleration-encoded scan alone, weights the measured acceleration by its reliability and addresses the "
     "resolution limit of the severe jets.")
p = doc.add_paragraph()
fmt_body(p)
add_runs(p, "**Keywords:** physics-informed neural networks, Navier-Stokes equations, 4D flow MRI, aortic coarctation, "
            "pressure estimation, JAX, scientific machine learning")

heading("Table of Contents", 1)
toc()

# ------------------------------------------------------------------ 1. Introduction
heading("1. Introduction and Research Context", 1)
heading("1.1 Background of the domain", 2, False)
para("Fluid flow in engineering and in the human body obeys the incompressible Navier-Stokes equations, a statement "
     "of Newton's second law and mass conservation for a fluid. Classical computational fluid dynamics (CFD) solves "
     "them on a mesh; it is accurate but needs a geometry, boundary conditions and expert set-up, and it cannot easily "
     "absorb measurements. Physics-informed neural networks [1] take a different route: a neural network maps a point "
     "in space and time to the velocity and pressure there, and training penalises any violation of the equations, "
     "the boundary conditions and, when available, the measured data. Automatic differentiation provides the exact "
     "derivatives the equations need, so no mesh is required, and the same machinery solves forward problems "
     "(predicting a flow) and inverse problems (inferring unmeasured quantities from measurements) [1], [2].")
para("A standard ladder of benchmarks with known answers tests such methods: the 2D Taylor-Green vortex (exact "
     "solution), the lid-driven cavity (Ghia et al. [11]), flow past a cylinder in a channel (Schafer and Turek "
     "[12]) and the 3D Taylor-Green vortex at Re = 1600, which turns turbulent (DeBonis [13]). Recent work added "
     "training techniques for these problems: Fourier features against spectral bias [6], adaptive loss weights [3], "
     "causal training for time-dependent flows [4], residual-based adaptive sampling (RAD) [5], separable networks for "
     "3D [7] and physics-informed operator networks (DeepONets) that learn whole families of solutions [8], [9].")
heading("1.2 Motivation", 2, False)
para("An aortic coarctation is a congenital narrowing of the aorta. Its severity, and the decision to intervene, rests "
     "largely on the pressure drop across the narrowing. The gold standard is cardiac catheterisation, an invasive "
     "procedure with radiation exposure and procedural risk. The routine non-invasive estimate uses Doppler "
     "ultrasound and the simplified Bernoulli rule, pressure drop = 4 v^2 with v the jet's peak speed [24], which ignores "
     "friction, unsteadiness and the jet's shape. 4D flow MRI [16] measures the full three-dimensional velocity field "
     "through an averaged heartbeat, but never pressure. A method that infers pressure reliably from MRI velocity alone "
     "would remove the need for many catheter studies.")
heading("1.3 Current scenario and application context", 2, False)
bullets(["**Clinical practice:** catheter measurement for the decision, Doppler/Bernoulli for screening and follow-up; "
         "4D flow MRI is available in larger centres and used mainly for flow quantification.",
         "**Conventional MRI pressure estimation:** solving a pressure Poisson equation (PPE) on the measured velocity "
         "differentiates noisy voxel data; on the public phantom used here it underestimates the catheter (slope 0.52).",
         "**Research:** PINNs have inferred hidden pressure from flow visualisations [2] and arterial pressure "
         "waveforms from 4D flow MRI with reduced one-dimensional models [10]; a recently released phantom dataset with "
         "4D flow MRI, acceleration-encoded MRI and simultaneous catheter pressure [19] now allows a validated test.",
         "**SDG relevance:** SDG 3 (non-invasive diagnosis) and SDG 9 (open, reproducible scientific software)."])
heading("1.4 Need for the proposed work", 2, False)
para("On the 8 cases of the public phantom, the clinical Bernoulli rule has a mean peak-pressure error of 30.4% and is "
     "always low; the conventional PPE method has 45.7%. PINN training on real 3D data is known to be fragile [3], [17], "
     "and published PINN pressure estimates from 4D flow MRI either use reduced-order models or lack catheter "
     "validation. A careful, reproducible study that first establishes what PINNs can and cannot do on the standard "
     "benchmarks and then tests them against catheter pressure is needed before such methods can be trusted.")
heading("1.5 Project scope", 2, False)
label("Included:")
bullets(["implementation of the full PINN pipeline (physics residuals, architectures, constraints, losses, training, "
         "metrics, visualisation) with a 60-test suite;",
         "training and evaluation of the seven benchmarks against their published gates, with an ablation study;",
         "an inverse PINN for pressure from MRI, tested on simulated stenoses and on the 8-case Stanford phantom "
         "against catheter pressure, with Bernoulli and PPE baselines;",
         "a live interactive viewer that runs every trained network on the GPU."])
label("Excluded:")
bullets(["real patient data and any clinical use; the phantom is 3D-printed and the study is research only;",
         "turbulence-resolving simulation (DNS/LES) of the phantom, and a full mesh-based CFD reference for it;",
         "optimisers not available in the chosen software stack (SOAP), and OpenFOAM/FEniCSx reference solutions."])

# ------------------------------------------------------------------ 2. Problem
heading("2. Problem Identification and Motivation", 1)
para("Two problems were identified. First, PINN results in the literature are reported on selected cases with "
     "specific tricks; it is unclear which component makes a PINN succeed or fail on the standard benchmark ladder. "
     "Second, the clinically important quantity, the pressure drop across a narrowing, cannot be measured by imaging, "
     "and the non-invasive estimates in use are biased low.")
label("Problem Statement:")
para("To determine whether a physics-informed neural network, trained only on the velocity (and, where available, "
     "acceleration) measured by 4D flow MRI, can infer the trans-coarctation pressure drop more accurately than the "
     "simplified Bernoulli rule when validated against simultaneous catheter pressure, and to establish, on the "
     "standard incompressible-flow benchmarks, which training components make PINNs succeed or fail.")
label("Research questions:")
numbered(["RQ1: Which PINN components (Fourier features, network architecture, hard constraints, causal training, "
          "adaptive weighting, adaptive sampling, separable networks) decide success or failure on the benchmark ladder?",
          "RQ2: Can an inverse PINN recover the pressure drop across a simulated stenosis within 10% of a CFD truth from a "
          "realistic synthetic MRI scan (noise, voxel and phase averaging)?",
          "RQ3: On real 4D flow MRI of a coarctation phantom, can the PINN beat Bernoulli's 30.4% mean error against "
          "the catheter, and what limits its accuracy?"])

# ------------------------------------------------------------------ 3. Literature
heading("3. Related Research / Critical Literature Review", 1)
para("Sources were identified from the project handbook's reference list, citation chains of the core PINN papers "
     "(forward and backward), and targeted searches for PINN pressure estimation from MRI (journals: J. Comput. Phys., "
     "CMAME, SIAM J. Sci. Comput., Science, NeurIPS; databases: Google Scholar, arXiv). Table {tab:lit} compares the most "
     "relevant works by problem, data, method, findings and limitation.")
lit = [["Sr.", "Author / Year", "Problem addressed", "Dataset / source", "Method / algorithm", "Key findings",
        "Limitation / gap", "Relevance to this work"],
       ["1", "Raissi et al., 2019 [1]", "Forward and inverse PDE problems", "Burgers, Navier-Stokes cylinder wake (Nektar)",
        "PINN: MLP, AD residuals in the loss", "Recovers PDE coefficients and hidden pressure from sparse data",
        "Fixed loss weights; hard for unsteady, high-Re flows", "Basis of the loss design; our inverse-wake benchmark"],
       ["2", "Raissi et al., 2020 [2]", "Velocity and pressure from flow visualisation", "Synthetic and experimental flows",
        "PINN with passive-scalar transport", "Hidden velocity and pressure inferred from concentration fields",
        "Needs dense scalar data; no catheter validation", "Motivates pressure from imaging data"],
       ["3", "Wang, Teng, Perdikaris, 2021 [3]", "Gradient pathologies in PINN training", "Helmholtz, flow benchmarks",
        "Learning-rate annealing (grad-norm weights), Modified MLP", "Large error reductions over fixed weights",
        "Weights can explode for near-zero terms", "Used (rows C, F); we document weight explosions"],
       ["4", "Wang, Sankaran, Perdikaris, 2024 [4]", "Violation of causality in time-dependent PINNs", "Chaotic and fluid systems",
        "Causal residual weights with an annealed tolerance", "Fixes failure to propagate the initial condition",
        "Extra hyper-parameters; slow annealing", "Used for the cylinder and Taylor-Green 3D"],
       ["5", "Wu et al., 2023 [5]", "Choice of collocation points", "Several PDE benchmarks",
        "Residual-based adaptive distribution (RAD)", "Better accuracy at the same budget",
        "Only sees large pointwise residuals", "Used; extended to data misfit; we show its blind spot"],
       ["6", "Tancik et al., 2020 [6]", "Spectral bias of MLPs", "Images, 3D shapes",
        "Random Fourier feature embedding", "Networks learn high-frequency detail",
        "Scale sigma must be tuned per problem", "Used in every network here"],
       ["7", "Cho et al., 2023 [7]", "Cost of 3D + time PINNs", "3D Navier-Stokes, (2+1)D problems",
        "Separable PINN (SPINN), forward-mode AD on grids", "Orders-of-magnitude faster training",
        "Low-rank structure limits capacity", "Used for the 3D Taylor-Green vortex"],
       ["8", "Wang, Wang, Perdikaris, 2021 [9]", "Solution operators of parametric PDEs", "Several parametric PDEs",
        "Physics-informed DeepONet", "Fast predictions for unseen parameters",
        "Unstable at high Reynolds number", "Used for the cavity at any lid speed"],
       ["9", "Kissas et al., 2020 [10]", "Arterial blood pressure from 4D flow MRI", "In-vitro and patient 4D flow MRI",
        "PINN with a 1D reduced-order blood-flow model", "Pressure waveforms from velocity and area",
        "1D model: no 3D jet or turbulent loss", "Closest prior work; we use full 3D Navier-Stokes"],
       ["10", "Nair et al., dataset [19]", "Validated pressure estimation in coarctation", "3D-printed phantom, 8 cases, catheter",
        "Conventional PPE from velocity; acceleration-encoded 4D-FlowP", "Acceleration encoding improves agreement (slope 0.63 to 0.95)",
        "Phantom only; no learning-based method", "Our validation data; our baselines"]]
table(lit, "Critical comparison of related research", key="lit", widths=[0.35, 0.9, 0.9, 0.85, 0.95, 0.95, 0.9, 0.95], size=8.5)
label("Literature synthesis:")
para("Three patterns emerge. (i) PINN accuracy depends heavily on training mechanics: weighting [3], sampling [5], "
     "causality [4] and architecture [6], [7]; most papers show improvements on chosen problems, but few attribute "
     "failures on a full benchmark ladder to individual components. (ii) Pressure-from-imaging PINNs either rely on "
     "dense visual data [2] or on reduced 1D models [10]; full 3D Navier-Stokes inference on clinical-resolution "
     "4D flow MRI with catheter validation is missing. (iii) The newest MRI sequences encode the fluid's acceleration "
     "[19], which enters the momentum equation directly, yet no physics-informed network uses it as data (abstract-level "
     "search, October 2026). These patterns define the research gap.")
label("Minimum expectation:")
para("Ten core works are compared in Table {tab:lit} and twenty-four sources are cited overall; the guide may specify a "
     "different number or additional sources for Stage II.")

# ------------------------------------------------------------------ 4. Gap
heading("4. Research Gap", 1)
label("Research Gap:")
bullets(["No reproducible study attributes PINN success and failure on the full standard benchmark ladder to "
         "individual components with ablation and independent references.",
         "PINN pressure inference from real 3D 4D flow MRI, validated against simultaneous catheter pressure, is rarely "
         "reported; existing studies use synthetic data or 1D reduced models.",
         "A failure mode in which a jointly trained PINN leaves the pressure flat and absorbs the flow's acceleration "
         "into a small but coherent momentum residual has not been documented, nor has a fix.",
         "Acceleration-encoded MRI has not been used as training data for a physics-informed network."])
label("Novelty / proposed contribution:")
bullets(["A benchmark study with gates, ablation rows A-H and independent finite-difference references, including the "
         "finding that the published 2% cavity gate is tighter than the accuracy of its reference data.",
         "Diagnostics for PINN pressure: a momentum-balance path integral along the vessel and a stand-alone pressure "
         "fit to a frozen velocity, which separate data limits from training failures.",
         "A separate, exactly periodic pressure network that raised the phantom's pressure drop from 0.3 to 5.3 mmHg at "
         "the same budget; a Smagorinsky eddy viscosity computed from the resolved flow; the measured acceleration as a "
         "data term (8.74 mmHg, 33% error, ahead of Bernoulli on the development case; 42.0% mean error on all eight "
         "cases, behind Bernoulli's 30.4%).",
         "Data-preparation steps for the public phantom (eddy-current correction, velocity sign recovery by minimum "
         "divergence, catheter clock alignment by flow-meter cross-correlation) and an interactive GPU viewer."])

# ------------------------------------------------------------------ 5. Objectives
heading("5. Research Objectives, Scope and Expected Contribution", 1)
label("Objectives:")
numbered(["Implement the PINN framework (physics, networks, constraints, losses, training, metrics) in JAX/Flax with "
          "tests that verify the residuals against exact solutions to 1e-10.",
          "Train the seven benchmarks against their published gates and attribute every failure through ablation.",
          "Build an inverse PINN for pressure from MRI and validate it on simulated stenoses within 10% of the CFD truth.",
          "Apply it to the 8-case coarctation phantom and beat Bernoulli's 30.4% mean error against the catheter.",
          "Provide a live viewer and reproducible documentation of every run, result and decision."])
label("Scope:")
para("Incompressible, Newtonian, constant-property flows at Reynolds numbers 100-2600; forward and inverse problems; "
     "synthetic and phantom MRI data. Clinical validation on patients is outside the scope of the project.")
label("Expected contribution:")
para("A tested open-source framework and model set, a benchmark analysis with failure attribution, a method and "
     "diagnostics for pressure inference from 4D flow MRI validated against catheter pressure, a prepared phantom "
     "dataset pipeline, an interactive viewer, and a research paper (draft in Annexure A).")

# ------------------------------------------------------------------ 6. Methodology
heading("6. Proposed Methodology / Module Design", 1)
para("Figure {fig:method} shows the end-to-end methodology. The same pipeline serves the benchmarks (physics only, graded against "
     "exact, CFD or DNS references) and the MRI problems (measured data plus physics, graded against catheter pressure).")
figure("fig1_methodology.png", "End-to-end methodology and modules", key="method")
label("Modules:")
bullets(["**Data preparation:** for the phantom, eddy-current background correction (2nd-order polynomial on static "
         "tissue), velocity component signs chosen by minimum divergence, lumen segmentation from speed, catheter "
         "planes checked against the flow meter, and alignment of the catheter and scan clocks by cross-correlating "
         "the flow-meter waveform with the scan's plane flux.",
         "**Networks:** a point-wise network maps (t, x, y[, z]) to (u, v[, w], p). Inputs pass through periodic "
         "embeddings and Fourier features [6]; the trunk is a Modified MLP [3]. For the phantom the pressure has its "
         "own network with 12 heartbeat harmonics (Figure {fig:networks}). Separable networks (SPINN) [7] handle 3D grids and a "
         "DeepONet [9] handles a parameter (the Reynolds number).",
         "**Automatic differentiation:** forward-mode directional derivatives (jvp) give du/dt, grad u, the Laplacian "
         "and grad p without forming full Jacobian tensors (which needed 63 GB).",
         "**Loss:** data misfit (voxel- and window-averaged as the scanner averages), momentum and continuity residuals, "
         "wall and boundary terms, and optional acceleration and eddy-viscosity terms.",
         "**Training:** Adam with warm-up and exponential decay; grad-norm adaptive weights; causal weights; RAD; "
         "Reynolds-number curricula and time windows.",
         "**Evaluation and diagnostics:** gate metrics; Bernoulli and PPE baselines; momentum path integrals and "
         "stand-alone pressure fits.",
         "**Live viewer:** a FastAPI inference server on the GPU and an interactive desktop viewer (Section 9.2)."])
figure("fig2_networks.png", "Velocity network and the separate periodic pressure network used for the phantom", key="networks")
label("Methodology justification:")
para("A mesh-free PINN can assimilate noisy, voxel-averaged velocity directly and enforce the full 3D momentum "
     "balance, which Bernoulli (a 1D, inviscid, steady relation [24]) and reduced 1D models [10] cannot. Classical "
     "PPE differentiates noisy data; the PINN instead fits a smooth field whose derivatives are exact. The benchmark "
     "ladder is the standard way to establish trust in a PDE solver before it is used on unknown data.")
label("Software Requirement Specifications:")
table([["Type", "Requirement"],
       ["Functional", "Train any benchmark from the command line with an ablation row and overrides"],
       ["Functional", "Evaluate a trained run against its reference and write final_eval.json"],
       ["Functional", "Prepare the phantom data, compute baselines and a comparison report"],
       ["Functional", "Serve trained networks to an interactive viewer (fields, particles, curves)"],
       ["Non-functional", "Reproducible: fixed seed 42, configs saved with every run, float32 at highest precision"],
       ["Non-functional", "Peak GPU memory set by a chunk size, not the batch (fits 6-16 GB GPUs)"],
       ["Non-functional", "Physics verified by tests before training (residuals < 1e-10 in float64)"],
       ["Non-functional", "Linux/WSL2 with CUDA for training; Windows CPU for tests and the viewer"]],
      "Key software requirements", widths=[1.4, 5.3])

# ------------------------------------------------------------------ 7. Mathematical model
heading("7. Mathematical Model and Algorithm Design", 1)
label("Variables / parameters:")
bullets(["u = (u, v, w): velocity; p: pressure; t: time; x = (x, y, z): position; rho: density; mu: dynamic viscosity; "
         "nu = mu/rho: kinematic viscosity; Re = U L / nu: Reynolds number.",
         "theta: network weights; lambda_k: loss weights; N_r, N_d: numbers of collocation and data points.",
         "Phantom scales: L = 10 mm, U = 1 m/s, T = 10 ms, rho = 1100 kg/m^3, mu = 4.2 mPa s, giving Re = 2619 and a "
         "pressure unit rho U^2 = 1100 Pa = 8.25 mmHg."])
label("Assumptions:")
bullets(["Incompressible, Newtonian fluid with constant density and viscosity.",
         "Rigid walls (no slip) and, for the MRI problems, an exactly periodic heartbeat.",
         "MRI velocity is a voxel- and acquisition-window average of the true velocity, with Gaussian noise.",
         "Unresolved turbulence acts as an extra (eddy) viscosity."])
label("Mathematical formulation:")
para("Non-dimensional incompressible Navier-Stokes equations:")
equation("∂u/∂t + (u·∇)u = −∇p + (1/Re) ∇²u ,        ∇·u = 0")
para("For a network N_θ(t, x) = (u, p) the residuals at a collocation point are")
equation("r_mom = ∂u/∂t + (u·∇)u + ∇p − (1/Re) ∇²u ,        r_c = ∇·u")
para("and the training loss is a weighted sum of mean squared terms:")
equation("L(θ) = λ_d·mean|ū_θ − u_MRI|² + λ_r·mean|r_mom|² + λ_c·mean|r_c|² + λ_w·mean|u_wall|² + λ_a·mean|ā_θ − a_MRI|²")
para("where ū_θ averages the network over each voxel and acquisition window (4 sub-samples), and ā_θ is the material "
     "acceleration Du/Dt = ∂u/∂t + (u·∇)u averaged the same way when acceleration-encoded MRI is available. Hard "
     "constraints remove terms exactly: a stream function ψ with u = ∂ψ/∂y, v = −∂ψ/∂x makes ∇·u = 0 by construction; "
     "for the synthetic stenosis the walls are built in as ψ = Q(t)(3η − η³)/4 + (1 − η²)² N_θ with η = y/h(x). "
     "Unresolved turbulence enters through the Smagorinsky model [15]:")
equation("ν → ν + ν_t ,     ν_t = (C_s Δ)² |S| ,     |S| = √(2 S_ij S_ij) ,     C_s = 0.17,  Δ = 1.8 mm (voxel)")
para("Adaptive weights follow the gradient norms [3], updated every 1000 steps with momentum 0.9:")
equation("λ̂_k = Σ_j ‖∇_θ L_j‖ / ‖∇_θ L_k‖ ,        λ_k ← 0.9 λ_k + 0.1 λ̂_k")
para("RAD [5] redraws collocation points with probability p(x) ∝ ε(x)^k / mean(ε^k) + c (k = 2, c = 0 for the "
     "phantom, 20% kept uniform). The reported quantity is the section-averaged pressure drop between the catheter "
     "planes, ΔP(t) = p̄_AAo(t) − p̄_DAo(t), and its peak over the beat; the gate is the relative error of the peak "
     "against the catheter.")
label("Algorithm / pseudocode:")
code(["Algorithm 1: training a physics-informed network",
      "input : config (architecture, loss weights, schedule), data D (optional), steps S",
      "init  : theta <- network init (seed 42); weights lambda; Adam state",
      "for s = 1..S:",
      "    draw collocation points X_r (uniform, or from the RAD pool) and data batch X_d",
      "    compute u, p and their derivatives at X_r by forward-mode AD (chunked, rematerialised)",
      "    L <- sum_k lambda_k * mean squared term_k        # data, momentum, continuity, walls, ...",
      "    theta <- Adam(theta, grad L), learning rate with warm-up and decay",
      "    every 1000 steps: update lambda by gradient norms; resample RAD pool; anneal causal tolerance",
      "    every 5000 steps: evaluate (gate metrics), log, checkpoint",
      "output: theta, metrics.csv, final_eval.json"])
label("Complexity or computational considerations:")
para("Each step costs one forward pass plus derivative passes per collocation point; third-order derivatives (stream "
     "function) cost about three times second-order ones. Memory would grow with the batch, so residuals are evaluated "
     "in rematerialised chunks (lax.map plus jax.checkpoint): peak memory scales with the chunk size. SPINN evaluates "
     "32^4 (about one million) space-time points per step at 3.1 GB. Measured step times on an RTX 5070 Ti range from "
     "0.013 s (DeepONet) to 0.36 s (stream-function cylinder); the whole benchmark study took 30.5 GPU hours.")
label("Alternative algorithms considered and reason for selection:")
table([["Alternative", "Why not chosen (or how used)"],
       ["Simplified Bernoulli 4v^2", "Clinical standard; 1D, inviscid, steady; used as the baseline to beat"],
       ["Pressure Poisson equation on the scan", "Differentiates noisy voxel data; used as the second baseline"],
       ["Mesh-based CFD of the phantom", "Needs exact geometry and boundary conditions; cannot assimilate data"],
       ["Purely data-driven network", "No physics: cannot infer pressure, which is never measured"],
       ["L-BFGS second stage", "Implemented; overfits the fixed batch (loss 1000x lower, error 3x higher); dropped"],
       ["Free learned eddy-viscosity field", "Implemented; collapses to zero in joint training; replaced by Smagorinsky"]],
      "Alternatives considered", widths=[2.2, 4.5])

# ------------------------------------------------------------------ 8. WBS, plan, risk
heading("8. Work Breakdown Structure, Project Plan and Risk Analysis", 1)
table([["WBS", "Work package", "Main outputs", "Lead"],
       ["1", "Environment and physics core", "WSL2/CUDA set-up, residuals, 60 tests", "Mehul Rawat"],
       ["2", "Taylor-Green 2D and 3D", "tgv2d_G, tgv3d_H, ablation on Benchmark A", "Atharva Palande"],
       ["3", "Cavity and PI-DeepONet", "cavity rows C/D/F, FD reference solver, DeepONet runs", "Ishwari Shendarkar"],
       ["4", "Cylinder forward and inverse", "time-windowed runs, inverse wake", "Mantavya Agarwal"],
       ["5", "Synthetic stenosis from MRI", "CFD solver, synthetic scan, stenosis_mri runs", "Mehul Rawat"],
       ["6", "Coarctation phantom", "data pipeline, baselines, networks, diagnostics", "Mehul Rawat"],
       ["7", "Live viewer", "GPU inference server, interactive viewer", "Mehul Rawat"],
       ["8", "Documentation and paper", "STATUS log, README, paper draft, this report", "All members"]],
      "Work breakdown structure", widths=[0.5, 1.9, 2.7, 1.6])
figure("fig3_gantt.png", "Project plan: Stage I (done) and Stage II (planned)", key="gantt")
table([["Risk", "Likelihood / impact", "Mitigation"],
       ["GPU budget too small for the handbook's budgets (~150 GPU h)", "High / high",
        "Reduced budgets stated per run; desktop RTX 5070 Ti; separable networks"],
       ["Desktop instability (unclean restarts during training)", "Medium / medium",
        "Resumable training driver; GPU power cap; checkpoints every 10k steps"],
       ["Training pathologies (weight explosion, flat pressure)", "High / high",
        "Ablation rows; diagnostics (path integral, stand-alone fit); design fixes"],
       ["Data errors in the public phantom export", "Medium / high",
        "Flow-meter checks, sign recovery, QA images per case"],
       ["Overfitting the scan's noise", "Medium / medium", "No L-BFGS on MRI problems; validation against catheter"],
       ["Tuning on the test cases", "Medium / high",
        "Development case P2 baseline only; frozen setup on 7 held-out cases"]],
      "Risk analysis", widths=[2.6, 1.3, 2.8])

# ------------------------------------------------------------------ 9. Preliminary implementation
heading("9. Preliminary Implementation", 1)
para("All modules of Figure {fig:method} are implemented and in use. This section describes the code base and tools, "
     "the live viewer, the results of the data preparation, and the barriers met during the work and how each was "
     "overcome.")
heading("9.1 Code base and tools", 2, False)
para("The code base (Python, about 13,500 lines) is organised as a package with one module per step of the method, one "
     "class per benchmark, command-line scripts and a viewer:")
table([["Part", "Contents"],
       ["pinnflow/physics.py", "Navier-Stokes residuals (velocity-pressure, stream function, vector potential), "
                               "forward-mode derivatives, vorticity and Q-criterion"],
       ["pinnflow/archs.py", "Fourier features, Modified MLP, PirateNet, SPINN, DeepONet"],
       ["pinnflow/constraints.py, losses.py", "Hard boundary conditions; loss terms, grad-norm, causal weights, RAD"],
       ["pinnflow/training.py, configs.py", "Trainer, Adam/L-BFGS, curricula, time windows; configs and ablation rows A-H"],
       ["pinnflow/problems/", "One class per benchmark, including the stenosis and phantom inverse problems"],
       ["pinnflow/coa_phantom.py, vessel.py", "Phantom data preparation and baselines; stenosis CFD and synthetic MRI"],
       ["scripts/", "train.py, evaluate.py, serve.py; data, results and machine scripts"],
       ["viewer/", "Interactive viewer (PySide6, PyVista) fed by the GPU inference server"],
       ["tests/", "60 tests, including the physics gate (residuals < 1e-10 on exact solutions)"]],
      "Implemented code base", widths=[2.2, 4.5], key="code")
para("Technologies: JAX and Flax (networks and automatic differentiation), Optax (optimisers), ml_collections "
     "(configuration), NumPy/SciPy (references and data), FastAPI (inference server), PySide6 and PyVista (viewer), "
     "pytest. Training runs inside WSL2 with CUDA on an RTX 5070 Ti (16 GB); development uses an RTX 3060 laptop GPU. "
     "Trained models travel from the desktop to the laptop through a private model repository and a watcher that "
     "fetches each new model and restarts the live server, so results appear in the viewer without manual steps.")
heading("9.2 Live viewer", 2, False)
para("Every trained network can be inspected live: the inference server evaluates the network on the GPU for every "
     "frame, pigment particles are carried by the network's own velocity, and each scene shows its verdict, its "
     "reference and the evidence for it (Figures {fig:viewer_phantom} and {fig:viewer_cyl}).")
figure("fig7_viewer_phantom.png", "Live viewer: the real MRI phantom (P2 baseline), laminar and acceleration-trained "
                                  "networks side by side, pressure over the heartbeat against the catheter", 6.4,
       key="viewer_phantom")
figure("fig8_viewer_cylinder.png", "Live viewer: the forward cylinder benchmark, pigment carried by the network and the "
                                   "flow-rate evidence of its failure", 6.4, key="viewer_cyl")
heading("9.3 Data preparation results", 2, False)
para("Figure {fig:phantoms} shows the eight phantom aortas as the scanner measured them, after the corrections of "
     "Section 6: the maximum speed at peak flow, darker meaning faster. The jet below each coarctation is visible, "
     "stronger in the severe and normal-flow cases. Table {tab:qa} summarises the quality checks of the prepared data.")
figure("fig10_phantoms.png", "The eight phantom aortas from 4D flow MRI (speed at peak flow; darker is faster). "
                             "P2 baseline shows the upstream (AAo) and downstream (DAo) catheter planes", 6.4,
       key="phantoms")
table([["Case", "Lumen volume (mL)", "Upstream flux / flow meter", "Catheter clock shift (ms)", "Shift correlation"],
       ["P1 baseline", "124", "73.3 / 67.0 (1.09)", "-124", "0.99"],
       ["P1 severe", "121", "67.8 / 66.9 (1.01)", "-130", "0.99"],
       ["P2 baseline", "198", "62.0 / 67.7 (0.92)", "-147", "0.97"],
       ["P2 severe", "170", "63.1 / 68.1 (0.93)", "-135", "0.98"],
       ["P1 baseline, low flow", "118", "40.9 / 37.5 (1.09)", "-136", "1.00"],
       ["P1 severe, low flow", "108", "39.4 / 36.8 (1.07)", "-136", "0.99"],
       ["P2 baseline, low flow", "189", "33.0 / 36.0 (0.92)", "-123", "0.99"],
       ["P2 severe, low flow", "150", "37.8 / 35.7 (1.06)", "-121", "0.97"]],
      "Quality checks of the prepared phantom data (flows in mL/s)", widths=[1.6, 1.1, 1.6, 1.3, 1.1], size=9.5, key="qa")
para("Every case passes the flow-meter check (the scan's flux through the upstream plane is within 8-9% of the meter). "
     "The catheter recording and the scan count the heartbeat from different events (the meter's flow upstroke and the "
     "MRI trigger); matching the two flow waveforms finds the offset, 121-147 ms with correlation 0.97-1.00 "
     "(Figure {fig:clock}). Without it, the catheter and network pressure curves would be compared at the wrong times.")
figure("fig12_clock.png", "P2 baseline: the flow meter and the scan's flux before and after aligning the two clocks",
       key="clock")
para("For the synthetic stenosis, Figure {fig:stenosis_data} shows what the network sees: the CFD truth is resolved with "
     "50 cells per diameter, while the simulated scan averages it over 0.6 mm voxels and 24 phases and adds noise "
     "(SNR 20). The jet's sharp edges are blurred and its peak is lowered, which is why the network must rely on the "
     "physics to recover the pressure.")
figure("fig11_stenosis_data.png", "Synthetic stenosis at peak flow: the CFD truth against the simulated 4D flow scan "
                                  "the network trains on", key="stenosis_data")
heading("9.4 Barriers encountered and how we overcame them", 2, False)
para("Table {tab:barriers} lists the main barriers met during the project, the evidence that revealed each one, how it "
     "was overcome and the result. Several were found only by diagnosing a failure rather than by tuning, and the "
     "fixes changed the problem set-up or the model design rather than the optimiser settings.")
table([["Area", "Barrier and evidence", "How we overcame it", "Outcome"],
       ["Computing", "No CUDA JAX for Windows; pip silently installed a CPU-only JAX", "WSL2 Ubuntu with CUDA JAX for "
        "training; the Windows environment only for tests and the viewer", "GPU training on both machines"],
       ["Computing", "Full Jacobian/Hessian tensors of the residual needed 63 GB", "Forward-mode directional derivatives "
        "(jvp) and residuals evaluated in rematerialised chunks", "1.6-3.1 GB peak, set by the chunk size"],
       ["Computing", "SPINN at the handbook's 64^4 grid needed one 11.1 GB buffer", "A 32^4 grid (about one million "
        "space-time points per step)", "3D run trained at 3.1 GB"],
       ["Computing", "Compute budget: the handbook's plan needs about 150-160 GPU hours", "Reduced budgets stated per "
        "run; desktop RTX 5070 Ti; fewer, longer time windows", "Whole study in 30.5 GPU hours"],
       ["Computing", "WSL shut down idle sessions and killed background runs", "Idle timeouts disabled, detached "
        "processes and a keep-alive session", "Long runs finish unattended"],
       ["Computing", "Desktop restarted uncleanly five times in three days, four during training", "Resumable driver, "
        "checkpoints every 10k steps, GPU power cap", "No results lost"],
       ["Computing", "Training on the desktop, demonstrating on the laptop", "Private model repository and a watcher "
        "that fetches new models and restarts the server", "Models appear in the viewer automatically"],
       ["Training", "Cavity: exact lid and no-slip walls contradict mass conservation in two corners (continuity floor "
        "0.04, vortex 20% weak)", "Soft lid velocity, exact no-slip elsewhere", "0.27% from a 512^2 reference"],
       ["Training", "Adaptive weights exploded: 465 (lid), 2.4e5 (outflow), 6e9 (3D w)", "Fixed weights where it "
        "hurt; zero-target terms excluded from weighting", "Cavity soft lid 1.9% instead of 21%"],
       ["Training", "L-BFGS on a fixed batch: loss 1000x lower, error 3x higher", "Dropped for data problems; "
        "results reported before and after", "Stenosis served from the end of Adam (6.2%)"],
       ["Training", "Cylinder: impulsive start is incompatible with incompressibility", "Smooth sin^2 inflow ramp "
        "over the first second", "Residuals 20-80x lower"],
       ["Training", "Cylinder: only 2% of the inflow reached the outlet (small divergence everywhere)", "Stream "
        "function, divergence-free by construction", "75-77% reaches the outlet"],
       ["Training", "PI-DeepONet diverged (loss spike, 70-87% error)", "Reynolds curriculum, learning rate 3e-4, "
        "gradient clipping, larger batches", "9.65% at the unseen Re 400"],
       ["Reference", "An exact solution misses the cavity's 2% Ghia gate (Ghia's v table is off near the wall)",
        "Own finite-difference solver to 512^2, Richardson extrapolation", "Independent grading of every run"],
       ["Data", "Phantom archive is 18 GB", "Read the zip index and fetch only the needed members", "1.4 GB downloaded"],
       ["Data", "Eddy-current offsets of 1-2 cm/s; one case segmented the gel (plane flux 2% of the meter)",
        "2nd-order background correction on static tissue; speed-based segmentation", "Flux within 8-9% of the meter"],
       ["Data", "Two velocity components had wrong signs; the scan's own streamlines left the vessel",
        "Signs chosen by minimum divergence of the measured flow", "Consistent flow in all 8 cases"],
       ["Data", "Catheter and scan clocks offset by 121-147 ms", "Cross-correlate the meter flow with the scan's flux",
        "Correlation 0.97-1.00"],
       ["Model", "Synthetic stenosis with soft walls learned almost no pressure (7e-4 mmHg)", "Walls built into a "
        "stream function, pressure output scaled", "6.2% and 4.1% errors"],
       ["Model", "RAD was silently inactive for the phantom", "Residual and data-misfit RAD (k = 2, c = 0)",
        "Systolic jet share of the data 6% to 18%"],
       ["Model", "Phantom pressure flat (0.3 mmHg) while its own flow needs about 6 mmHg", "Momentum path integral and "
        "stand-alone fit to diagnose; separate periodic pressure network", "5.3 mmHg at 20k steps, still rising"],
       ["Model", "Learned eddy viscosity collapses to zero (2.5 to 0.002 times the fluid's)", "Smagorinsky eddy "
        "viscosity computed from the flow", "Stable, about 1.2 times the fluid's"],
       ["Model", "Velocity-only data caps every method at 7-8 mmHg", "Acceleration-encoded MRI registered onto the "
        "scan and used as data", "8.74 mmHg, ahead of Bernoulli on P2 baseline"],
       ["Evaluation", "Networks read at the 2D-PC outlet plane, not where the catheter measured; on P1 the catheter's "
        "drop changes 2x between its 5 positions", "Gel markers located in the 1 mm anatomy scans (checked by eye); "
        "networks re-read at the AAo and DAo location-3 markers", "P1 baseline 36% to 7%; mean 43.5%: the gap to "
        "Bernoulli is real"]],
      "Barriers encountered and how they were overcome", widths=[0.8, 2.25, 2.25, 1.6], size=8.5, key="barriers")
label("Lessons learned:")
bullets(["Constraints that contradict each other (the exact cavity lid, the impulsive cylinder start) leave a residual "
         "that no amount of training removes; the fix is a consistent problem set-up.",
         "A small error that points the same way everywhere adds up to a large global error (lost mass in the channel, "
         "lost pressure along the aorta), and a mean-squared loss cannot see it. Integral checks find it.",
         "Adaptive weighting and second-order optimisers need care: both can make the loss look better while the "
         "solution gets worse, so every result is graded against an independent reference.",
         "Real data needs its own quality checks (flow meter, divergence, clock alignment) before any model is trained.",
         "Diagnosis before tuning saved compute: the stand-alone pressure fit showed in 28 minutes on a CPU that the "
         "information was in the flow, which pointed to the separate pressure network."])

# ------------------------------------------------------------------ 10. Results
heading("10. Experimental Setup, Results and Discussion", 1)
heading("10.1 Experimental setup", 2, False)
bullets(["Hardware: RTX 5070 Ti 16 GB (training), RTX 3060 6 GB (development); WSL2 Ubuntu with CUDA JAX.",
         "Precision: float32 with highest matmul precision; float64 for the physics tests; seed 42 for every run.",
         "Budgets: the handbook's budgets need about 150-160 GPU hours; within a 12-hour window several runs used "
         "reduced budgets, stated per run. The benchmark study used 30.5 GPU hours.",
         "References: exact solutions (Taylor-Green 2D), Ghia et al. [11] and our own finite-difference solver "
         "(cavity), FEATFLOW series [12] (cylinder), DeBonis DNS [13] (Taylor-Green 3D), Nektar wake data [1], CFD "
         "truth (synthetic stenosis), catheter pressure [19] (phantom)."])
heading("10.2 Benchmark results", 2, False)
table([["Benchmark", "Gate", "Result", "Verdict"],
       ["A: Taylor-Green 2D", "velocity rel. L2 < 1e-4", "4.05e-5 (pressure 2.59e-4)", "passed"],
       ["B: cavity, soft lid (row C)", "Ghia centreline < 2% at Re 1000", "u 0.63%, v 2.05%; 0.27% from our 512^2 solution", "gate missed"],
       ["B: cavity, hard lid (row F)", "same", "u 19.8%, v 20.2%", "failed"],
       ["C: DFG cylinder (forward)", "St within 3%, Cd_max within 5%", "no vortex shedding; best Cd_max 1.54 vs 3.22", "failed"],
       ["D: Taylor-Green 3D (SPINN)", "dissipation peak within 5%", "peak at t = 0, not t = 9; no transition", "failed"],
       ["Inverse wake", "recover coefficients and hidden pressure", "lambda_2 error 0.065%, pressure 2.44%", "passed"],
       ["PI-DeepONet cavity", "zero-shot error < 10% at Re 400", "9.65% (Re 100: 2.1%, Re 1000: 30%)", "passed"]],
      "Benchmark results against their gates", widths=[1.8, 1.7, 2.3, 0.9], size=9.5, key="bench")
para("Table {tab:cavity} compares the cavity runs at Re = 1000. The lid treatment decides the outcome; with the soft "
     "lid, fixed weights succeed where grad-norm weighting (row F) undoes the fix by raising the lid weight to about 465.")
table([["Run", "Lid", "Weights", "Ghia u / v error", "Error vs 512^2 solution"],
       ["row C", "soft", "fixed", "0.63% / 2.05%", "0.27%"],
       ["row D, soft lid", "soft (walls exact)", "fixed", "1.04% / 2.20%", "1.9%"],
       ["row F, soft lid", "soft (walls exact)", "grad-norm", "16.0% / 15.6%", "21.0%"],
       ["row D", "hard", "fixed", "21.3% / 19.8%", "28.3%"],
       ["row F", "hard", "grad-norm", "19.8% / 20.2%", "26.1%"]],
      "Lid-driven cavity at Re = 1000: the lid treatment and the weighting", widths=[1.3, 1.4, 1.0, 1.5, 1.5], size=9.5,
      key="cavity")
label("Discussion of the benchmarks:")
bullets(["**Cavity:** a converged solution of the same regularised problem scores u 0.41% / v 2.29% against Ghia's "
         "tables; the 2% v gate is tighter than the reference data (our grid sequence extrapolates to the spectral "
         "value of Botella and Peyret [14]). Row C is essentially exact (0.27% from the 512^2 solution). The hard lid "
         "fails because exact lid velocity and no slip on the side walls make mass conservation impossible in the two "
         "top corners; fixed weights fail the same way, so the constraint, not the weighting, is the cause.",
         "**Cylinder:** a chain of diagnoses: too few steps per time window; an impulsive start that is incompatible "
         "with incompressibility (fixed by ramping the inflow); and mass lost through a small divergence at every point "
         "that adds up over the 22-diameter channel (2% of the inflow reached the outlet). An exactly divergence-free "
         "stream function raised it to 75-77% (Figure {fig:cyl_mass}), but shedding did not appear within the budget.",
         "**Taylor-Green 3D:** the network keeps the flow two-dimensional (w stays near zero), so vortex stretching and "
         "the turbulent transition never occur; grad-norm weighting raised the weight of an identically-zero term to 6e9.",
         "**Ablation (Figure {fig:ablation}):** at short budgets the plain network (row A) learns fastest; the full stack "
         "pays off with long training (row G reached 4.05e-5 after 50k steps). Rows A-C, without the stream function, "
         "leave a divergence of 1.4e-2 to 7.7e-2; rows D-G have exactly zero. Row G costs about 9 times more per step.",
         "**PI-DeepONet (Figure {fig:deeponet}):** the first run diverged; a Reynolds curriculum, a lower learning rate "
         "with gradient clipping and finally a four-times larger batch brought the unseen Re = 400 below the gate.",
         "**Inverse versus forward:** the same cylinder flow fails as a forward solve and succeeds as an inverse problem, "
         "because the data pins the flow down and the physics only has to supply a few unknowns."])
figure("fig13_ablation_tgv.png", "Taylor-Green 2D ablation at 3,000 steps (rows A-G)", 5.6, key="ablation")
figure("fig15_cylinder_mass.png", "Cylinder: share of the inflow reaching stations down the channel", 5.2, key="cyl_mass")
figure("fig14_deeponet.png", "PI-DeepONet: error at the unseen Re = 400 over the training legs", 6.0, key="deeponet")
heading("10.3 Synthetic stenosis (pressure from simulated MRI)", 2, False)
table([["Case", "PINN peak-drop error", "Bernoulli error"],
       ["60% stenosis, Re 150", "6.2% (0.557 vs 0.594 mmHg)", "+31%"],
       ["70% stenosis, Re 300 (fast jet)", "4.1% (3.96 vs 4.13 mmHg)", "+21%"]],
      "Synthetic stenosis: PINN against the CFD truth", widths=[2.4, 2.4, 1.9], key="stenosis")
para("Exact walls (stream function) were essential: with walls as soft loss terms the network learned almost no "
     "pressure (7e-4 mmHg). L-BFGS overfitted the scan's noise and was dropped. Figure {fig:sten_training} shows the "
     "error during training: both networks pass the 10% gate by 15,000 steps and stay well below the Bernoulli rule "
     "applied to the same scan; the hidden pressure field was recovered to 3.9% (stenosis) and 9.0% (fast jet).")
figure("fig19_stenosis_training.png", "Synthetic stenosis: peak pressure-drop error during training", 5.6,
       key="sten_training")
heading("10.4 Real coarctation phantom (pressure from 4D flow MRI against the catheter)", 2, False)
para("The Stanford 4D-FlowP phantom [19] has 8 cases (2 geometries, baseline and severe narrowing, normal and low flow) "
     "with 4D flow MRI at 1.8 mm and 60 ms and catheter pressure at 1 kHz. Figure {fig:baselines} and Table "
     "{tab:baselines} show the baselines computed from the same scans: Bernoulli has slope 0.77 against the catheter "
     "and a mean error of 30.4%, always low; PPE has slope 0.52 (the source publication reports 0.63 for conventional "
     "4D flow) and 45.7%.")
figure("fig9_baselines.png", "Conventional estimates against catheter pressure on all 8 phantom cases", 4.2,
       key="baselines")
table([["Case", "Catheter", "PPE", "Bernoulli"],
       ["P1 baseline", "16.2", "6.5", "10.3"], ["P1 severe", "24.7", "10.2", "22.0"], ["P2 baseline", "13.1", "7.9", "8.0"],
       ["P2 severe", "20.8", "14.5", "16.7"], ["P1 baseline, low flow", "8.9", "5.3", "5.1"],
       ["P1 severe, low flow", "13.4", "6.7", "12.0"], ["P2 baseline, low flow", "7.2", "4.9", "3.9"],
       ["P2 severe, low flow", "12.0", "5.5", "7.4"]],
      "Peak pressure drop (mmHg) from the same scans", widths=[2.4, 1.4, 1.4, 1.5], size=10, key="baselines")
para("On the development case (P2 baseline, catheter 13.1 mmHg) the first network fitted the flow well (jet 9% slow) "
     "but predicted only 0.30 mmHg. Integrating the momentum equation along a 259 mm path through the vessel "
     "(Table {tab:momentum}, Figure {fig:momentum}) showed that the network's own flow needs about 6 mmHg, but the "
     "network left the flow's systolic acceleration in a small, coherent residual (0.02 per unit length, invisible in a "
     "mean-squared loss) instead of the pressure.")
table([["Time in the beat", "Acceleration term", "Convection term", "Viscous term", "Required p(DAo) - p(AAo)",
        "Network's p(DAo) - p(AAo)"],
       ["125 ms", "+4.55", "-0.13", "-0.11", "-4.53", "+0.05"],
       ["185 ms", "+3.73", "+2.31", "-0.17", "-6.21", "-0.12"],
       ["245 ms (peak flow)", "-0.51", "+5.89", "-0.15", "-5.52", "-0.34"],
       ["490 ms", "-3.72", "-0.89", "-0.17", "+4.44", "+0.10"]],
      "Momentum balance along the vessel for the first network (path integrals, mmHg)",
      widths=[1.3, 1.05, 1.05, 0.9, 1.2, 1.2], size=9, key="momentum")
figure("fig16_momentum.png", "The pressure drop the network's own flow requires against the drop in its pressure", 5.6,
       key="momentum")
para("A pressure fitted alone to the frozen velocity gave 7.26 mmHg with a third less residual and followed the "
     "catheter waveform with correlation 0.96: the information was in the flow, joint training failed to extract it. "
     "The fixes and their effect are in Figure {fig:journey}; Figure {fig:heartbeat} compares the waveforms and Figure "
     "{fig:phantom_training} shows how each design behaves during training.")
figure("fig6_journey.png", "P2 baseline: peak pressure drop at each step of the diagnosis and design", key="journey")
figure("fig4_heartbeat.png", "P2 baseline: pressure drop over one heartbeat, catheter against the networks",
       key="heartbeat")
figure("fig17_phantom_training.png", "P2 baseline: peak pressure drop during training for each design", 6.0,
       key="phantom_training")
para("Table {tab:variants} compares the designs at the same budget. A learned eddy-viscosity field does not help: "
     "joint training drives it towards zero within 20,000 steps whatever the penalty (Figure {fig:eddy}), because the "
     "data can be fitted without it. The Smagorinsky viscosity, computed from the resolved flow, stays at about 1.2 "
     "times the fluid's viscosity and adds a little loss.")
table([["Design (P2 baseline)", "Steps", "Peak drop (mmHg)", "Mean drop (mmHg)"],
       ["shared pressure output (first network)", "60k", "0.30", "0.00"],
       ["separate pressure network, laminar", "20k", "5.28", "0.86"],
       ["+ learned eddy viscosity, penalty 0.01", "20k", "5.48", "0.86"],
       ["+ learned eddy viscosity, penalty 0.001", "20k", "5.63", "0.95"],
       ["+ Smagorinsky (C_s 0.17), 4 sub-samples per voxel", "20k", "6.05", "0.92"],
       ["+ Smagorinsky (C_s 0.17), 16 sub-samples per voxel", "20k", "5.30", "1.04"],
       ["+ Smagorinsky + measured acceleration as data", "60k", "8.74", "1.61"],
       ["catheter (truth)", "", "13.10", "3.68"]],
      "Network designs on the development case", widths=[3.4, 0.7, 1.3, 1.3], size=9.5, key="variants")
figure("fig18_eddy.png", "Eddy viscosity during training: learned fields collapse, the Smagorinsky term does not", 5.6,
       key="eddy")
figure("fig5_methods.png", "All methods on the development case and the first held-out case", key="methods")
label("Held-out test on all eight cases:")
para("The design was frozen after the development case (P2 baseline) and run unchanged on the other seven: for each case "
     "a laminar pressure-network run and a run with the Smagorinsky viscosity and the measured acceleration, 60,000 "
     "steps each (16 runs, finished on 27 October 2026). Table {tab:eight} and Figure {fig:eight} give the result.")
table([["Case", "Catheter", "Bernoulli", "Pressure Poisson", "PINN, laminar", "PINN + acceleration"],
       ["P1 baseline", "16.2", "10.3 (36%)", "6.5 (60%)", "20.3 (25%)", "22.1 (36%)"],
       ["P1 severe", "24.7", "22.0 (11%)", "10.2 (59%)", "16.4 (33%)", "16.3 (34%)"],
       ["P2 baseline (development)", "13.1", "8.0 (39%)", "7.9 (40%)", "8.8 (33%)", "8.7 (33%)"],
       ["P2 severe", "20.8", "16.7 (20%)", "14.5 (30%)", "4.8 (77%)", "3.7 (82%)"],
       ["P1 baseline, low flow", "8.9", "5.1 (43%)", "5.3 (41%)", "14.8 (65%)", "14.9 (67%)"],
       ["P1 severe, low flow", "13.4", "12.0 (10%)", "6.7 (50%)", "9.8 (27%)", "11.8 (12%)"],
       ["P2 baseline, low flow", "7.2", "3.9 (46%)", "4.9 (32%)", "5.7 (21%)", "6.4 (11%)"],
       ["P2 severe, low flow", "12.0", "7.4 (38%)", "5.5 (54%)", "5.2 (57%)", "4.7 (61%)"],
       ["Mean error, 8 cases", "", "30.4%", "45.7%", "42.3%", "42.0%"]],
      "Peak pressure drop (mmHg) and error against the catheter on all eight phantom cases",
      widths=[1.7, 0.8, 1.0, 1.1, 1.05, 1.15], size=9, key="eight")
figure("fig21_phantom_cases.png", "Peak pressure drop of every method on the eight phantom cases", 6.4, key="eight")
bullets(["**The gate is missed:** 42.0% mean error against Bernoulli's 30.4%; the networks are ahead of pressure Poisson "
         "(45.7%) only. Over the seven held-out cases alone the errors are 43.6% (laminar) and 43.3% (with acceleration). "
         "The slope against the catheter is about 0.7 but R^2 is negative: the errors are not a consistent bias.",
         "**The measured acceleration does not help on average** (42.0% against 42.3%). It helps where it registers well "
         "onto the scan (P2 baseline low flow 21% to 11%, P1 severe low flow 27% to 12%) and hurts where it does not "
         "(P2 severe, Dice 0.65: 77% to 82%).",
         "**Two kinds of failure:** far too low on the severe P2 cases (3.7-5.2 against 12.0-20.8 mmHg) and far too high "
         "on the P1 baseline cases, with both networks, so the acceleration term is not the cause there."])
label("Reading the networks at the catheter's own locations:")
para("The catheter recorded the descending aorta at five positions, and on geometry P1 its peak drop changes by a "
     "factor of two between them (P1 baseline 11.7 to 23.2 mmHg; on P2 within 1.4 mmHg). The networks had been read at "
     "the outlet plane of the 2D phase-contrast series, not at a catheter position, and their drop rises steadily down "
     "the descending aorta (P1 baseline 13.7 to 30.8 mmHg). The gel markers beside the vessel are visible in each case's "
     "1 mm anatomy scan: five bars along the descending aorta and one beside the ascending aorta. They were located "
     "(automatic search, checked by eye on overlays) and the trained networks were re-read, without retraining, as "
     "cross-section means at the ascending marker and the middle (location 3) descending marker (Figure {fig:markers}).")
figure("fig22_markers.png", "Peak-drop error at the 2D-PC planes and at the catheter's own marker locations "
                            "(network with acceleration)", 6.4, key="markers")
bullets(["P1 baseline drops to 7% (acceleration) and 9% (laminar): its overshoot came from the reading position.",
         "Every other case gets worse, because the 2D-PC outlet plane lay near the bottom marker where the networks' drop "
         "is largest. The mean error becomes 43.5% (acceleration) and 45.5% (laminar), so the gap to Bernoulli is real.",
         "The severe narrowings fail most (2-4 mmHg against 12-21 mmHg): a narrow, fast jet is what the scan blurs, and "
         "the physics cannot restore what the data does not contain."])
label("Result discussion:")
bullets(["The separate periodic pressure network is the key design change: 0.3 to 5.3 mmHg at the same budget with an "
         "unchanged velocity fit.",
         "A free learned eddy viscosity collapses to zero in joint training; a Smagorinsky viscosity computed from the "
         "flow does not, but adds little loss at 1.8 mm resolution (6.05 mmHg).",
         "The measured acceleration as data gives 8.74 mmHg (33.3% error), above Bernoulli (8.0, 39%) and PPE (7.9, 40%) "
         "on this case. Velocity-only methods meet at 7-8 mmHg here: the scan's jet peaks at 1.42 m/s, while 13.1 mmHg "
         "would need about 1.8 m/s, so blur in space (1.8 mm) and time (60 ms) is a ceiling for every method.",
         "On the eight cases the frozen method misses the gate (42.0% against Bernoulli's 30.4%; 43.5% at the "
         "catheter markers). What worked on the development case does not generalise, and the severe jets are the "
         "main reason: the velocity data under-resolve them, and the pressure follows the velocity."])
label("Comparison with existing work:")
para("Bernoulli and PPE on the same scans are the direct comparison (Table {tab:baselines}, Figure {fig:methods}). The "
     "source publication reports that acceleration-encoded 4D-FlowP improves the conventional estimate from slope 0.63 "
     "to 0.95 [19]; our network is the first, to our knowledge, to use that acceleration as PINN training data. Unlike "
     "the 1D PINN of Kissas et al. [10], our model resolves the 3D jet and its losses, and unlike most PINN studies it "
     "is validated against simultaneous catheter pressure on every case of a public dataset, held-out cases "
     "included. On that test it does not yet beat the clinical rule.")
heading("10.5 Computational cost", 2, False)
para("Figure {fig:cost} and Table {tab:cost} give the cost of the main runs. Peak memory stays below 3.1 GB thanks to "
     "chunked, rematerialised residuals, so every run fits a consumer GPU; the step time is set mainly by the "
     "derivative order (third-order stream-function residuals cost about three times second-order ones).")
figure("fig20_cost.png", "Training wall-clock of the main runs", 5.6, key="cost")
table([["Run", "GPU", "Step time", "Peak memory", "Wall-clock"],
       ["tgv2d_G", "RTX 5070 Ti", "0.14 s", "< 6 GB", "about 2 h"],
       ["cavity_C_r2", "RTX 5070 Ti", "0.025 s", "1.59 GB", "1.6 h"],
       ["cavity_F_r2", "RTX 5070 Ti", "0.024 s", "1.58 GB", "2.3 h"],
       ["tgv3d_H (SPINN, 32^4)", "RTX 5070 Ti", "0.037 s", "3.07 GB", "1.2 h"],
       ["cylinder_inverse", "RTX 5070 Ti", "0.055 s", "1.63 GB", "0.5 h"],
       ["cylinder_sf_2x6k", "RTX 5070 Ti", "0.359 s", "1.87 GB", "1.0 h"],
       ["deeponet_cavity (last leg)", "RTX 5070 Ti", "0.025 s", "0.33 GB", "0.5 h (3.1 h in total)"],
       ["stenosis_mri_G", "RTX 3060 laptop", "-", "-", "1.6 h"],
       ["stenosis_mri_fast", "RTX 5070 Ti", "-", "-", "1.6 h"],
       ["coa P2 baseline, acceleration, 60k", "RTX 5070 Ti (250 W cap)", "-", "-", "about 3.9 h"],
       ["coa, 8 cases x 2 networks, 60k each", "RTX 5070 Ti, two at a time", "-", "-", "about 3.5 days"]],
      "Cost of the main runs", widths=[2.2, 1.6, 0.9, 1.0, 1.4], size=9.5, key="cost")

# ------------------------------------------------------------------ 11. Feasibility
heading("11. Feasibility, Limitations and Ethical Considerations", 1)
label("Feasibility:")
bullets(["Technical: all modules work end to end; a 60k-step phantom network trains in about 4 hours on one consumer GPU.",
         "Economic: open-source software and public data; a single consumer GPU suffices for Stage II.",
         "Schedule: Stage II tasks are planned with buffers (Figure {fig:gantt})."])
label("Limitations:")
bullets(["Three of seven benchmarks fail their gates within the reduced budgets (cylinder, Taylor-Green 3D, hard-lid cavity).",
         "The held-out test failed: 42.0% mean error against Bernoulli's 30.4% (43.5% at the catheter markers). Accuracy "
         "is limited by MRI resolution, most of all for the severe jets.",
         "The phantom is a rigid 3D print with a glycerol-water fluid: no wall compliance, no real blood rheology.",
         "One development case gave an optimistic picture (33% error, ahead of Bernoulli); the held-out protocol "
         "(frozen setup, seven cases) exposed it, which is why it was used.",
         "One seed per run; the pressure Poisson baseline is still read at the 2D-PC planes, not at the markers."])
label("Ethical considerations:")
bullets(["Only public, anonymised phantom data (CC BY-NC 4.0, cited) and simulated data are used; no patient data.",
         "The method is a research prototype, not a medical device; it must not be used for clinical decisions.",
         "Results, failures and reduced budgets are reported transparently; nothing is tuned on held-out cases."])

# ------------------------------------------------------------------ 12. Stage II
heading("12. Project Stage-II Plan and Research Milestones", 1)
para("Stage I ended with a clear negative result on real data: the method that beat Bernoulli on the development case "
     "does not generalise to the held-out cases. Stage II therefore targets the causes found, in order of expected gain.")
heading("12.1 Future research steps", 2, False)
numbered(["**Train on the acceleration-encoded scan alone.** The 4D-FlowP acquisition measures velocity and acceleration "
          "together, so the two agree in space and time by construction; the separate 4D flow scan, whose registration "
          "with it is weak on some cases (Dice 0.65 on P2 severe), is then needed only for the catheter planes.",
          "**Weight the acceleration by its reliability.** Down-weight wall voxels and voxels with low signal, where the "
          "acceleration encoding is noisiest, instead of trusting every voxel equally.",
          "**Read every method at the catheter markers.** Re-read the pressure Poisson baseline at the markers, and "
          "make a data-preparation version with marker planes so that the report, the viewer and the training all use "
          "the catheter's own positions.",
          "**Address the severe jets.** Test an integral (control-volume) form of the momentum loss across the "
          "narrowing, sub-voxel sampling with more points in the jet, and a super-resolution prior trained on CFD of "
          "coarctations, since the scan's jet peak (1.42 m/s on P2 baseline) is below what the catheter drop implies.",
          "**Calibrate, then freeze.** Choose the Smagorinsky constant and the loss weights on the development case "
          "only, freeze them, and rerun all eight cases; report several seeds per case.",
          "**Quantify uncertainty.** Ensembles of networks to put an error bar on each predicted pressure drop, so a "
          "clinician could see when the estimate is not trustworthy."])
table([["Milestone", "Target date", "Success criterion"],
       ["M1: held-out test on all 8 phantom cases (done)", "Oct 2026", "Reported: 42.0% against Bernoulli's 30.4%; gate failed"],
       ["M2: acceleration-encoded scan alone, reliability weighting, every method read at the markers",
        "Dec 2026", "Held-out mean error below 30.4%"],
       ["M3: severe-jet study (integral momentum loss, sub-voxel sampling, super-resolution prior); seeds and "
        "uncertainty", "Feb 2027", "Severe cases within 30%; error bars on every case"],
       ["M4: paper submission (journal or conference)", "Mar 2027", "Submitted manuscript"],
       ["M5: final report, demonstration, viva", "Apr 2027", "Report and live demo"]],
      "Stage II milestones", widths=[3.3, 1.1, 2.3])
label("Expected final outcome:")
para("A method for non-invasive pressure-drop estimation from 4D flow MRI with a documented comparison against "
     "catheter pressure on eight cases, an open framework with benchmark analysis, and a research paper.")
label("Contingency plan if the proposed approach does not meet the target:")
para("If the held-out mean error stays above Bernoulli's, the contribution is a negative but publishable result with "
     "a diagnosis: the benchmark study, the documented PINN failure mode (pressure absorbed into a coherent residual) "
     "and its fix, the effect of the reading position, and a quantified resolution limit of velocity MRI for severe "
     "jets. The paper draft is already written this way.")

# ------------------------------------------------------------------ 13. Deliverables
heading("13. Deliverables / Expected Outcomes", 1)
bullets(["Research paper: a draft exists (paper/main.pdf in the repository; Annexure A); target a journal or conference "
         "in scientific machine learning or computational biomechanics.",
         "Application: the PINN framework with eight trained model families, the data-preparation pipeline for the public "
         "phantom, and the live interactive viewer.",
         "Research proposal for Stage II: the future research steps of Section 12.1.",
         "Documentation: a dated results log of every run and decision, and this report."])

# ------------------------------------------------------------------ 14. Conclusion
heading("14. Conclusion", 1)
para("The project addressed two questions: what makes physics-informed neural networks succeed or fail on the standard "
     "incompressible-flow benchmarks, and whether they can infer blood pressure from MRI velocity better than the "
     "clinical Bernoulli rule. A complete, tested PINN framework was implemented; three benchmarks pass their gates and "
     "every failure is attributed to a specific cause (incompatible constraints, incompatible initial data, coherent "
     "residuals that integrate to large global errors, and adaptive-weighting pathologies). Inverse problems proved far "
     "more tractable than forward ones: the cylinder wake's viscosity was recovered to 0.065%, and simulated stenosis "
     "pressure to 4-6% against Bernoulli's 21-31%. On real MRI of a coarctation phantom, a diagnosed failure mode led to "
     "a separate periodic pressure network and to acceleration-encoded MRI as data, raising the predicted drop from "
     "0.3 to 8.74 mmHg (33% error) and beating both conventional methods on the development case. The frozen method "
     "then failed the held-out test on all eight cases: 42.0% mean error against Bernoulli's 30.4% (43.5% when read "
     "at the catheter's own positions), failing most on the severe narrowings whose jets the scan blurs. The answer to "
     "RQ3 is therefore no, not yet, and the reasons are identified: the resolution of the velocity data, the weak "
     "registration of the acceleration on some cases and the reading position. Stage II addresses them in order "
     "(Section 12.1) and prepares the paper, which reports the negative result with its diagnosis.")

# ------------------------------------------------------------------ references
heading("References", 1)
refs = [
    'M. Raissi, P. Perdikaris and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," J. Comput. Phys., vol. 378, pp. 686-707, 2019.',
    'M. Raissi, A. Yazdani and G. E. Karniadakis, "Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations," Science, vol. 367, no. 6481, pp. 1026-1030, 2020.',
    'S. Wang, Y. Teng and P. Perdikaris, "Understanding and mitigating gradient flow pathologies in physics-informed neural networks," SIAM J. Sci. Comput., vol. 43, no. 5, pp. A3055-A3081, 2021.',
    'S. Wang, S. Sankaran and P. Perdikaris, "Respecting causality for training physics-informed neural networks," Comput. Methods Appl. Mech. Eng., vol. 421, 116813, 2024.',
    'C. Wu, M. Zhu, Q. Tan, Y. Kartha and L. Lu, "A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks," Comput. Methods Appl. Mech. Eng., vol. 403, 115671, 2023.',
    'M. Tancik et al., "Fourier features let networks learn high frequency functions in low dimensional domains," in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020.',
    'J. Cho, S. Nam, H. Yang, S.-B. Yun, Y. Hong and E. Park, "Separable physics-informed neural networks," in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.',
    'L. Lu, P. Jin, G. Pang, Z. Zhang and G. E. Karniadakis, "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators," Nat. Mach. Intell., vol. 3, pp. 218-229, 2021.',
    'S. Wang, H. Wang and P. Perdikaris, "Learning the solution operator of parametric partial differential equations with physics-informed DeepONets," Sci. Adv., vol. 7, no. 40, eabi8605, 2021.',
    'G. Kissas, Y. Yang, E. Hwuang, W. R. Witschey, J. A. Detre and P. Perdikaris, "Machine learning in cardiovascular flows modeling: Predicting arterial blood pressure from non-invasive 4D flow MRI data using physics-informed neural networks," Comput. Methods Appl. Mech. Eng., vol. 358, 112623, 2020.',
    'U. Ghia, K. N. Ghia and C. T. Shin, "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method," J. Comput. Phys., vol. 48, no. 3, pp. 387-411, 1982.',
    'M. Schafer and S. Turek, "Benchmark computations of laminar flow around a cylinder," in Flow Simulation with High-Performance Computers II, Notes on Numerical Fluid Mechanics, vol. 52. Vieweg, 1996, pp. 547-566.',
    'J. R. DeBonis, "Solutions of the Taylor-Green vortex problem using high-resolution explicit finite difference methods," NASA Technical Memorandum NASA/TM-2013-217850, 2013.',
    'O. Botella and R. Peyret, "Benchmark spectral results on the lid-driven cavity flow," Comput. Fluids, vol. 27, no. 4, pp. 421-433, 1998.',
    'J. Smagorinsky, "General circulation experiments with the primitive equations: I. The basic experiment," Mon. Weather Rev., vol. 91, no. 3, pp. 99-164, 1963.',
    'M. Markl, A. Frydrychowicz, S. Kozerke, M. Hope and O. Wieben, "4D flow MRI," J. Magn. Reson. Imaging, vol. 36, no. 5, pp. 1015-1036, 2012.',
    'A. S. Krishnapriyan, A. Gholami, S. Zhe, R. M. Kirby and M. W. Mahoney, "Characterizing possible failure modes in physics-informed neural networks," in Advances in Neural Information Processing Systems (NeurIPS), vol. 34, 2021.',
    'S. Wang, S. Sankaran, H. Wang and P. Perdikaris, "An expert\'s guide to training physics-informed neural networks," arXiv:2308.08468, 2023.',
    'Nair et al., 4D-FlowP aortic coarctation phantom: 4D flow MRI, acceleration-encoded MRI and catheter pressure, Stanford Digital Repository, doi:10.25740/ms528wr9373, licence CC BY-NC 4.0.',
    'J. Bradbury et al., "JAX: composable transformations of Python+NumPy programs," 2018. [Online]. Available: https://github.com/google/jax',
    'J. Heek et al., "Flax: A neural network library and ecosystem for JAX," 2020. [Online]. Available: https://github.com/google/flax',
    'D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. Int. Conf. Learning Representations (ICLR), 2015.',
    'D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large scale optimization," Math. Program., vol. 45, pp. 503-528, 1989.',
    'H. Baumgartner et al., "Recommendations on the echocardiographic assessment of aortic valve stenosis: a focused update from the European Association of Cardiovascular Imaging and the American Society of Echocardiography," J. Am. Soc. Echocardiogr., vol. 30, no. 4, pp. 372-392, 2017.',
]
for k, r_ in enumerate(refs, 1):
    p = doc.add_paragraph()
    fmt_body(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=4, spacing=1.15)
    p.paragraph_format.left_indent = Inches(0.45)
    p.paragraph_format.first_line_indent = Inches(-0.45)
    add_runs(p, f"[{k}]\t{r_}", size=11)

# ------------------------------------------------------------------ annexures
heading("Annexures", 1)
heading("Annexure A: Proposal Draft / Research Paper", 2, False)
para("A paper draft covering the benchmark study (introduction, method, setup, results, discussion) is maintained in "
     "the project repository as paper/main.tex, with figures in paper/figures. It now also covers pressure from flow MRI: the "
     "synthetic stenosis and the eight-case phantom test against the catheter, with the failed gate, its diagnosis "
     "and the future research steps.")
heading("Annexure B: Additional data summaries / charts", 2, False)
table([["Run", "Benchmark", "Configuration", "Wall-clock"],
       ["tgv2d_G", "Taylor-Green 2D", "row G, 50k Adam + L-BFGS", "about 2 h"],
       ["cavity_C_r2 / cavity_F_r2", "cavity", "rows C / F, 200k Adam + 20k L-BFGS", "1.6 h / 2.3 h"],
       ["cylinder_sf_2x6k", "DFG cylinder", "stream function, 2 windows x 6k", "1.0 h"],
       ["tgv3d_H", "Taylor-Green 3D", "SPINN, 80k Adam + 1k L-BFGS, 32^4 grid", "1.2 h"],
       ["cylinder_inverse", "inverse wake", "25k Adam + 1k L-BFGS", "0.5 h"],
       ["deeponet_cavity_cont4", "PI-DeepONet", "660k Adam in total, larger batch last", "about 3 h"],
       ["stenosis_mri_G / _fast", "synthetic stenosis", "40k / 80k Adam", "1.6 h each"],
       ["coa_P2_baseline_accel_60k", "phantom", "pressure net + Smagorinsky + acceleration, 60k", "about 4 h"],
       ["coa_<case>_lam_60k / _accel_60k", "phantom, 8 cases", "laminar / Smagorinsky + acceleration, 60k each",
        "16 runs"]],
      "Main trained runs", widths=[2.0, 1.4, 2.4, 0.9], size=9.5)
heading("Annexure C: Code and repository", 2, False)
para("Repository: github.com/VoyagerToko/pinn (private; access on request). Main entry points: scripts/train.py "
     "(training), scripts/evaluate.py (evaluation), scripts/serve.py (inference server), python -m viewer (viewer); "
     "docs/STATUS.md records every run, result and decision with dates.")
heading("Annexure D: Prototype screenshots", 2, False)
para("Figures {fig:viewer_phantom} and {fig:viewer_cyl} (Section 9.2) show the live viewer: the real MRI phantom with both networks and the heartbeat "
     "comparison, and the forward cylinder benchmark.")
heading("Annexure E: Certificates of MOOC Course Completion", 2, False)
para("[To be attached by each student: Mehul Rawat, Mantavya Agarwal, Ishwari Shendarkar, Atharva Palande.]")

# fill in {fig:key} / {tab:key} references
import re as _re


def _fill(par):
    for r in par.runs:
        if "{fig:" in r.text or "{tab:" in r.text:
            r.text = _re.sub(r"\{((?:fig|tab):[a-z_0-9]+)\}", lambda m: str(LABELS[m.group(1)]), r.text)


for par in doc.paragraphs:
    _fill(par)
for t in doc.tables:
    for row in t.rows:
        for cell in row.cells:
            for par in cell.paragraphs:
                _fill(par)
doc.save(D / "report_draft.docx")
print("saved report_draft.docx;", fig_no[0], "figures,", tab_no[0], "tables")
