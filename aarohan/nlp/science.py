"""Scientific intelligence and fundamental physical law grounding for Aarohan 2.0.

Provides exact, rigorous mathematical formulations, physical interpretations,
variable definitions, and LaTeX representations for fundamental laws of physics,
chemistry, and thermodynamics. Prevents neural hallucination in small base models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScientificLaw:
    """Encapsulates a verified physical or mathematical law."""

    law_id: str
    canonical_title: str
    category: str
    summary: str
    formula_latex: str
    variables: dict[str, str]
    detailed_markdown: str


SCIENTIFIC_LAWS: dict[str, ScientificLaw] = {
    "third_law_thermodynamics": ScientificLaw(
        law_id="third_law_thermodynamics",
        canonical_title="Third Law of Thermodynamics",
        category="Thermodynamics",
        summary="As temperature approaches absolute zero (0 K), the entropy of a pure, perfectly crystalline substance approaches zero.",
        formula_latex=r"\lim_{T \to 0} S = 0 \quad \text{and} \quad S = k_{\text{B}} \ln \Omega",
        variables={
            "S": "Entropy of the system (in J·K⁻¹)",
            "T": "Absolute thermodynamic temperature (in Kelvin, K)",
            "k_B": "Boltzmann constant (≈ 1.380649 × 10⁻²³ J·K⁻¹)",
            r"\Omega": "Number of accessible quantum microstates (multiplicity)",
            r"\Delta S": "Isothermal change in entropy (approaches 0 as T → 0 K)",
        },
        detailed_markdown=(
            "### The Third Law of Thermodynamics\n\n"
            "The **Third Law of Thermodynamics** (historically established by Walther Nernst and Max Planck) "
            "states that **as the absolute temperature of a closed system approaches absolute zero ($0\\text{ K}$ or $-273.15^\\circ\\text{C}$), "
            "the entropy of a pure, perfectly crystalline substance approaches a constant minimum, defined as zero.**\n\n"
            "---\n\n"
            "### 1. Mathematical Formulations\n\n"
            "#### A. Thermodynamic Formulation (Nernst-Simon Statement)\n"
            "For any isothermal physical or chemical transformation involving pure condensed phases in internal equilibrium:\n"
            "$$\\lim_{T \\to 0} S = 0$$\n"
            "$$\\lim_{T \\to 0} \\Delta S = 0$$\n\n"
            "**Variables:**\n"
            "- $S$: Entropy of the system (measured in $\\text{J}\\cdot\\text{K}^{-1}$).\n"
            "- $T$: Absolute thermodynamic temperature in Kelvin ($\\text{K}$).\n"
            "- $\\Delta S$: Entropy change during an isothermal process as $T \\to 0$.\n\n"
            "---\n\n"
            "#### B. Statistical Mechanics Formulation (Boltzmann-Planck Relation)\n"
            "Ludwig Boltzmann formulated entropy in terms of statistical multiplicity:\n"
            "$$S = k_{\\text{B}} \\ln \\Omega$$\n\n"
            "**Variables:**\n"
            "- $k_{\\text{B}} = 1.380649 \\times 10^{-23}\\ \\text{J}\\cdot\\text{K}^{-1}$ (exact **Boltzmann constant**).\n"
            "- $\\Omega$: Number of accessible microscopic quantum states (degeneracy of the ground state).\n\n"
            "**At Absolute Zero ($T \\to 0\\text{ K}$):**\n"
            "A perfect crystalline system collapses into its non-degenerate ground state where $\\Omega = 1$:\n"
            "$$S = k_{\\text{B}} \\ln(1) = 0\\ \\text{J/K}$$\n\n"
            "---\n\n"
            "#### C. Heat Capacity Behaviour & Unattainability of Absolute Zero\n"
            "Entropy is related to heat capacity at constant pressure ($C_p$) by:\n"
            "$$S(T) - S(0) = \\int_0^T \\frac{C_p(T')}{T'}\\, dT'$$\n\n"
            "For this integral to converge as $T \\to 0$, the heat capacities must approach zero:\n"
            "$$\\lim_{T \\to 0} C_p = 0 \\quad \\text{and} \\quad \\lim_{T \\to 0} C_v = 0$$\n\n"
            "**Physical Implication (Principle of Unattainability):**\n"
            "Because an infinite number of adiabatic demagnetization or cooling cycles would be required, "
            "it is physically impossible to cool any real macroscopic system to **exact absolute zero ($0\\text{ K}$)** in a finite sequence of operations."
        ),
    ),
    "second_law_thermodynamics": ScientificLaw(
        law_id="second_law_thermodynamics",
        canonical_title="Second Law of Thermodynamics",
        category="Thermodynamics",
        summary="The total entropy of an isolated system always increases over time in spontaneous processes, and remains constant in reversible processes.",
        formula_latex=r"\Delta S_{\text{total}} \ge 0 \quad \text{and} \quad dS \ge \frac{\delta Q}{T}",
        variables={
            "S": "Entropy of the system (in J·K⁻¹)",
            "Q": "Heat exchanged (in Joules, J)",
            "T": "Thermodynamic temperature (in Kelvin, K)",
            r"\eta_{\text{Carnot}}": "Maximum theoretical Carnot efficiency: 1 - (T_C / T_H)",
        },
        detailed_markdown=(
            "### The Second Law of Thermodynamics\n\n"
            "The **Second Law of Thermodynamics** establishes the direction of spontaneous thermodynamic processes "
            "and asserts that the total entropy of an isolated system never decreases.\n\n"
            "---\n\n"
            "### 1. Mathematical Formulations\n\n"
            "#### A. Clausius Inequality\n"
            "For any thermodynamic cycle or process:\n"
            "$$\\oint \\frac{\\delta Q}{T} \\le 0$$\n\n"
            "For differential changes:\n"
            "$$dS \\ge \\frac{\\delta Q}{T}$$\n"
            "*(Equality holds for reversible processes; strict inequality holds for irreversible/spontaneous processes).*\n\n"
            "#### B. Isolated System Entropy\n"
            "$$\\Delta S_{\\text{universe}} = \\Delta S_{\\text{system}} + \\Delta S_{\\text{surroundings}} \\ge 0$$\n\n"
            "#### C. Carnot Engine Maximum Thermal Efficiency\n"
            "$$\\eta_{\\text{max}} = 1 - \\frac{T_{\\text{cold}}}{T_{\\text{hot}}}$$\n"
            "Where $T_{\\text{hot}}$ and $T_{\\text{cold}}$ are the temperatures of the heat reservoirs in Kelvin."
        ),
    ),
    "first_law_thermodynamics": ScientificLaw(
        law_id="first_law_thermodynamics",
        canonical_title="First Law of Thermodynamics",
        category="Thermodynamics",
        summary="Energy cannot be created or destroyed, only transformed from one form to another: internal energy change equals heat added minus work done.",
        formula_latex=r"\Delta U = Q - W \quad \text{or} \quad dU = \delta Q - \delta W",
        variables={
            r"\Delta U": "Change in internal energy (in Joules, J)",
            "Q": "Heat added to the system (in Joules, J)",
            "W": r"Work done by the system on surroundings (in Joules, J, where W = \int P dV)",
        },
        detailed_markdown=(
            "### The First Law of Thermodynamics\n\n"
            "The **First Law of Thermodynamics** is the principle of conservation of energy applied to thermodynamic systems.\n\n"
            "---\n\n"
            "### 1. Mathematical Formulation\n"
            "$$\\Delta U = Q - W$$\n\n"
            "In differential form:\n"
            "$$dU = \\delta Q - \\delta W$$\n\n"
            "**Variables:**\n"
            "- $\\Delta U$: Net change in the system's internal energy (Joules).\n"
            "- $Q$: Net heat transferred into the system from surroundings.\n"
            "- $W$: Work performed by the system against external forces (for pressure-volume work, $W = \\int P\\, dV$).\n\n"
            "*(Note: In IUPAC chemical convention where work is defined as done on the system, the formula is written $\\Delta U = Q + W$).*"
        ),
    ),
    "zeroth_law_thermodynamics": ScientificLaw(
        law_id="zeroth_law_thermodynamics",
        canonical_title="Zeroth Law of Thermodynamics",
        category="Thermodynamics",
        summary="If two systems are each in thermal equilibrium with a third system, they are in thermal equilibrium with each other.",
        formula_latex=r"T_A = T_C \quad \text{and} \quad T_B = T_C \implies T_A = T_B",
        variables={
            "T_A, T_B, T_C": "Thermodynamic temperatures of systems A, B, and C",
        },
        detailed_markdown=(
            "### The Zeroth Law of Thermodynamics\n\n"
            "The **Zeroth Law of Thermodynamics** establishes temperature as a transitive, well-defined state function and provides the foundational basis for thermometry.\n\n"
            "---\n\n"
            "### 1. Mathematical Formulation\n"
            "If system $A$ is in thermal equilibrium with system $C$, and system $B$ is in thermal equilibrium with system $C$:\n"
            "$$T_A = T_C \\quad \\text{and} \\quad T_B = T_C \\implies T_A = T_B$$\n\n"
            "**Significance:**\n"
            "Allows the calibration of thermometers (system $C$) to measure temperature without direct heat exchange between system $A$ and system $B$."
        ),
    ),
    "ideal_gas_law": ScientificLaw(
        law_id="ideal_gas_law",
        canonical_title="Ideal Gas Law",
        category="Thermodynamics / Chemistry",
        summary="The equation of state of a hypothetical ideal gas relating pressure, volume, temperature, and quantity.",
        formula_latex=r"PV = nRT = N k_{\text{B}} T",
        variables={
            "P": "Absolute pressure (Pascals, Pa)",
            "V": "Volume (m³)",
            "n": "Amount of substance (moles)",
            "R": "Universal gas constant (≈ 8.314462 J·mol⁻¹·K⁻¹)",
            "T": "Absolute temperature (Kelvin, K)",
            "N": "Total number of gas molecules",
            "k_B": "Boltzmann constant (≈ 1.380649 × 10⁻²³ J·K⁻¹)",
        },
        detailed_markdown=(
            "### The Ideal Gas Law\n\n"
            "$$\nPV = nRT = N k_{\\text{B}} T\n$$\n\n"
            "**Variables:**\n"
            "- $P$: Pressure in $\\text{Pa}$ (or $\\text{N/m}^2$).\n"
            "- $V$: Volume in $\\text{m}^3$.\n"
            "- $n$: Number of moles of gas.\n"
            "- $R = 8.314462\\ \\text{J}\\cdot\\text{mol}^{-1}\\cdot\\text{K}^{-1}$ (Universal Gas Constant).\n"
            "- $T$: Absolute temperature in Kelvin ($\\text{K}$).\n"
            "- $N$: Number of individual particles ($N = n \\times N_A$).\n"
            "- $k_{\\text{B}} = 1.380649 \\times 10^{-23}\\ \\text{J/K}$ (Boltzmann constant)."
        ),
    ),
    "newtons_second_law": ScientificLaw(
        law_id="newtons_second_law",
        canonical_title="Newton's Second Law of Motion",
        category="Classical Mechanics",
        summary="The net force on an object is equal to the time rate of change of its momentum, or mass times acceleration for constant mass.",
        formula_latex=r"\vec{F}_{\text{net}} = \frac{d\vec{p}}{dt} = m\vec{a}",
        variables={
            "F": "Net force vector (Newtons, N)",
            "p": "Linear momentum vector (p = m·v, kg·m/s)",
            "m": "Mass of the object (kg)",
            "a": "Acceleration vector (m/s²)",
        },
        detailed_markdown=(
            "### Newton's Second Law of Motion\n\n"
            "$$\\vec{F}_{\\text{net}} = \\frac{d\\vec{p}}{dt} = m\\vec{a}$$\n\n"
            "**Variables:**\n"
            "- $\\vec{F}_{\\text{net}}$: Vector sum of all external forces acting on the body (Newtons, $\\text{N}$).\n"
            "- $m$: Inertial mass of the body ($\\text{kg}$).\n"
            "- $\\vec{a}$: Acceleration vector ($\\text{m/s}^2$).\n"
            "- $\\vec{p} = m\\vec{v}$: Linear momentum ($\\text{kg}\\cdot\\text{m/s}$)."
        ),
    ),
    "newtons_gravitation": ScientificLaw(
        law_id="newtons_gravitation",
        canonical_title="Newton's Law of Universal Gravitation",
        category="Classical Mechanics",
        summary="Every particle attracts every other particle with a force proportional to the product of their masses and inversely proportional to the square of the distance between them.",
        formula_latex=r"F = G \frac{m_1 m_2}{r^2}",
        variables={
            "F": "Gravitational force magnitude (Newtons, N)",
            "G": "Gravitational constant (≈ 6.67430 × 10⁻¹¹ N·m²·kg⁻²)",
            "m_1, m_2": "Masses of the two interacting bodies (kg)",
            "r": "Distance between the centers of mass (meters, m)",
        },
        detailed_markdown=(
            "### Newton's Law of Universal Gravitation\n\n"
            "$$F = G \\frac{m_1 m_2}{r^2}$$\n\n"
            "**In vector form:**\n"
            "$$\\vec{F}_{12} = -G \\frac{m_1 m_2}{r^2} \\hat{r}_{12}$$\n\n"
            "**Variables:**\n"
            "- $G = 6.67430(15) \\times 10^{-11}\\ \\text{N}\\cdot\\text{m}^2\\cdot\\text{kg}^{-2}$ (Newtonian Gravitational Constant).\n"
            "- $m_1, m_2$: Point masses (in $\\text{kg}$).\n"
            "- $r$: Radial separation distance (in $\\text{m}$)."
        ),
    ),
    "einstein_mass_energy": ScientificLaw(
        law_id="einstein_mass_energy",
        canonical_title="Mass-Energy Equivalence (Einstein)",
        category="Relativity",
        summary="Mass and energy are equivalent and interchangeable.",
        formula_latex=r"E = mc^2 \quad \text{and} \quad E^2 = (pc)^2 + (m_0 c^2)^2",
        variables={
            "E": "Total energy (Joules, J)",
            "m": "Mass (kg)",
            "m_0": "Rest mass (kg)",
            "p": r"Relativistic momentum (p = \gamma m_0 v)",
            "c": "Speed of light in vacuum (299,792,458 m/s)",
        },
        detailed_markdown=(
            "### Einstein's Mass-Energy Equivalence\n\n"
            "**For a particle at rest:**\n"
            "$$E_0 = m_0 c^2$$\n\n"
            "**General relativistic energy-momentum relation:**\n"
            "$$E^2 = (pc)^2 + (m_0 c^2)^2$$\n\n"
            "**Variables:**\n"
            "- $E$: Total relativistic energy (Joules, $\\text{J}$).\n"
            "- $m_0$: Rest mass (kilograms, $\\text{kg}$).\n"
            "- $p$: Relativistic momentum ($p = \\gamma m_0 v$).\n"
            "- $c = 299{,}792{,}458\\ \\text{m/s}$ (speed of light in vacuum)."
        ),
    ),
    "ohms_law": ScientificLaw(
        law_id="ohms_law",
        canonical_title="Ohm's Law",
        category="Electromagnetism",
        summary="The current through a conductor between two points is directly proportional to the voltage across the two points.",
        formula_latex=r"V = IR \quad \text{and} \quad \vec{J} = \sigma \vec{E}",
        variables={
            "V": "Electric potential difference (Volts, V)",
            "I": "Electric current (Amperes, A)",
            "R": "Electrical resistance (Ohms, Ω)",
            "J": "Current density vector (A/m²)",
            r"\sigma": "Electrical conductivity (S/m)",
            "E": "Electric field vector (V/m)",
        },
        detailed_markdown=(
            "### Ohm's Law\n\n"
            "**Macroscopic Circuit Form:**\n"
            "$$V = IR$$\n\n"
            "**Microscopic Differential Form:**\n"
            "$$\\vec{J} = \\sigma \\vec{E}$$\n\n"
            "**Variables:**\n"
            "- $V$: Potential difference across the conductor (Volts, $\\text{V}$).\n"
            "- $I$: Current flow (Amperes, $\\text{A}$).\n"
            "- $R$: Resistance ($\\Omega$).\n"
            "- $\\vec{J}$: Current density ($\\text{A/m}^2$).\n"
            "- $\\sigma$: Electrical conductivity ($\\text{S/m}$).\n"
            "- $\\vec{E}$: Electric field vector ($\\text{V/m}$)."
        ),
    ),
    "coulombs_law": ScientificLaw(
        law_id="coulombs_law",
        canonical_title="Coulomb's Law",
        category="Electromagnetism",
        summary="The electrostatic force between two electric charges is directly proportional to the product of the charges and inversely proportional to the square of distance between them.",
        formula_latex=r"F = k_e \frac{|q_1 q_2|}{r^2} = \frac{1}{4\pi\varepsilon_0} \frac{|q_1 q_2|}{r^2}",
        variables={
            "F": "Electrostatic force magnitude (Newtons, N)",
            "k_e": "Coulomb's constant (≈ 8.98755 × 10⁹ N·m²·C⁻²)",
            "\varepsilon_0": "Vacuum permittivity (≈ 8.8541878 × 10⁻¹² F·m⁻¹)",
            "q_1, q_2": "Signed electric charges (Coulombs, C)",
            "r": "Distance between charges (meters, m)",
        },
        detailed_markdown=(
            "### Coulomb's Law\n\n"
            "$$F = \\frac{1}{4\\pi\\varepsilon_0} \\frac{|q_1 q_2|}{r^2}$$\n\n"
            "**Vector Form:**\n"
            "$$\\vec{F}_{12} = \\frac{1}{4\\pi\\varepsilon_0} \\frac{q_1 q_2}{r^2} \\hat{r}_{12}$$\n\n"
            "**Variables:**\n"
            "- $\\varepsilon_0 = 8.8541878128(13) \\times 10^{-12}\\ \\text{F/m}$ (Vacuum Permittivity).\n"
            "- $q_1, q_2$: Electric charges in Coulombs ($\\text{C}$).\n"
            "- $r$: Distance between charge centers ($\\text{m}$)."
        ),
    ),
    "photosynthesis": ScientificLaw(
        law_id="photosynthesis",
        canonical_title="Biochemical Mechanism of Photosynthesis",
        category="Biochemistry / Bioenergetics",
        summary="Photoautotrophic conversion of carbon dioxide and water into glucose and oxygen driven by solar photons, divided into thylakoid light-dependent reactions and stroma Calvin cycle carbon fixation.",
        formula_latex=r"6\text{CO}_2 + 6\text{H}_2\text{O} + h\nu \xrightarrow{\text{chlorophyll}} \text{C}_6\text{H}_{12}\text{O}_6 + 6\text{O}_2 \quad (\Delta G^\circ = +2870\text{ kJ/mol})",
        variables={
            r"\text{CO}_2": "Carbon dioxide from atmospheric intake via stomata",
            r"\text{H}_2\text{O}": "Water absorbed by roots and oxidized during photolysis",
            r"h\nu": "Light energy / solar photon flux (wavelengths 400–700 nm)",
            r"\text{C}_6\text{H}_{12}\text{O}_6": "D-Glucose synthesized as stored chemical energy",
            r"\text{O}_2": "Molecular oxygen byproduct released into the atmosphere",
        },
        detailed_markdown=(
            "### Photosynthesis: Complete Biochemical Mechanism & Chemical Reactions\n\n"
            "**Photosynthesis** is the central anabolic process by which green plants, algae, and cyanobacteria convert light energy into high-energy chemical bonds stored in hexose sugars (glucose).\n\n"
            "---\n\n"
            "### 1. Overall Net Chemical Reaction\n\n"
            "$$6\\text{CO}_2 + 6\\text{H}_2\\text{O} + \\text{photons} \\xrightarrow{\\text{chlorophyll}} \\text{C}_6\\text{H}_{12}\\text{O}_6 + 6\\text{O}_2$$\n\n"
            "$$\\Delta G^\\circ = +2870\\text{ kJ/mol} \\quad (\\text{Endothermic / Endergonic})$$\n\n"
            "---\n\n"
            "### 2. Stage 1: Light-Dependent Reactions (Thylakoid Membrane)\n\n"
            "Occurs across the **thylakoid membranes** of chloroplasts. Absorbed photons drive the non-cyclic photophosphorylation (Z-Scheme) and the photolysis of water.\n\n"
            "#### A. Photolysis of Water (Oxygen-Evolving Complex at PS II / P680):\n"
            "Photosystem II absorbs light at $\\lambda = 680\\text{ nm}$, exciting electrons that leave the reaction center. The manganese-calcium catalytic cluster ($\\ ext{Mn}_4\\text{CaO}_5$) replaces these electrons by splitting water:\n"
            "$$2\\text{H}_2\\text{O} \\xrightarrow{h\\nu} \\text{O}_2 + 4\\text{H}^+_{\\text{lumen}} + 4e^-$$\n\n"
            "#### B. Electron Transport Chain & Proton Pumping:\n"
            "Electrons flow sequentially: $\\text{P680} \\to \\text{Pheophytin} \\to \\text{Plastoquinone (PQ)} \\to \\text{Cytochrome } b_6f \\text{ complex} \\to \\text{Plastocyanin (PC)} \\to \\text{Photosystem I (P700)}$.\n"
            "The Cytochrome $b_6f$ complex pumps protons from the stroma into the thylakoid lumen, generating a steep electrochemical proton gradient ($\\Delta\\text{pH}$).\n\n"
            "#### C. NADPH Synthesis (PS I / Ferredoxin-NADP⁺ Reductase):\n"
            "Excited electrons from P700 transfer via ferredoxin to **FNR**:\n"
            "$$\\text{NADP}^+ + 2\\text{H}^+ + 2e^- \\xrightarrow{\\text{FNR}} \\text{NADPH} + \\text{H}^+$$\n\n"
            "#### D. Photophosphorylation (Chemiosmosis via ATP Synthase):\n"
            "Protons return down their concentration gradient into the stroma through $CF_0CF_1$ **ATP Synthase**:\n"
            "$$\\text{ADP} + \\text{P}_i + \\text{H}^+_{\\text{lumen}} \\xrightarrow{\\text{ATP Synthase}} \\text{ATP} + \\text{H}_2\\text{O}$$\n\n"
            "**Net Light-Dependent Reaction:**\n"
            "$$2\\text{H}_2\\text{O} + 2\\text{NADP}^+ + 3\\text{ADP} + 3\\text{P}_i \\xrightarrow{h\\nu} \\text{O}_2 + 2\\text{NADPH} + 2\\text{H}^+ + 3\\text{ATP}$$\n\n"
            "---\n\n"
            "### 3. Stage 2: Light-Independent Reactions / Calvin-Benson Cycle (Stroma)\n\n"
            "Occurs in the **chloroplast stroma**, utilizing the chemical energy ($\\text{ATP}$ and $\\text{NADPH}$) generated in the thylakoid to fix inorganic carbon.\n\n"
            "#### Phase 1: Carbon Fixation (Catalyzed by RuBisCO)\n"
            "Carbon dioxide reacts with the 5-carbon sugar Ribulose-1,5-bisphosphate (RuBP) via **RuBisCO** to produce an unstable 6-carbon intermediate that immediately cleaves into two molecules of 3-Phosphoglycerate (3-PGA):\n"
            "$$3\\text{RuBP (5C)} + 3\\text{CO}_2 + 3\\text{H}_2\\text{O} \\xrightarrow{\\text{RuBisCO}} 6\\text{ [3-Phosphoglycerate (3C)]}$$\n\n"
            "#### Phase 2: Reduction Phase\n"
            "3-PGA is phosphorylated by ATP and reduced by NADPH to Glyceraldehyde 3-Phosphate (G3P):\n"
            "$$6\\text{ (3-PGA)} + 6\\text{ATP} \\xrightarrow{\\text{Phosphoglycerate kinase}} 6\\text{ (1,3-Bisphosphoglycerate)} + 6\\text{ADP}$$\n"
            "$$6\\text{ (1,3-BPG)} + 6\\text{NADPH} + 6\\text{H}^+ \\xrightarrow{\\text{G3P Dehydrogenase}} 6\\text{ (G3P)} + 6\\text{NADP}^+ + 6\\text{P}_i$$\n\n"
            "**Sugar Yield:** For every 3 turns of the cycle (fixing $3\\text{CO}_2$), **1 net molecule of G3P** leaves the cycle to form glucose in the cytoplasm:\n"
            "$$2\\text{ G3P (3C)} \\xrightarrow{\\text{Gluconeogenesis / Aldolase}} \\text{C}_6\\text{H}_{12}\\text{O}_6\\text{ (Glucose)}$$\n\n"
            "#### Phase 3: Regeneration of RuBP\n"
            "The remaining 5 G3P molecules undergo a series of transketolase/aldolase reactions to regenerate 3 molecules of RuBP, consuming 3 additional ATP:\n"
            "$$5\\text{ G3P (3C)} + 3\\text{ATP} \\to 3\\text{ RuBP (5C)} + 3\\text{ADP} + 2\\text{P}_i$$\n\n"
            "---\n\n"
            "### 4. Summary Matrix of Stages\n\n"
            "| Feature | Light-Dependent Reactions | Calvin Cycle (Light-Independent) |\n"
            "| :--- | :--- | :--- |\n"
            "| **Location** | Thylakoid Membrane & Lumen | Chloroplast Stroma |\n"
            "| **Input Substrates** | $\\text{H}_2\\text{O}$, $\\text{NADP}^+$, $\\text{ADP}$, Photons | $\\text{CO}_2$, $\\text{ATP}$, $\\text{NADPH}$ |\n"
            "| **Primary Enzymes** | PS II (P680), PS I (P700), Cytochrome $b_6f$, ATP Synthase | RuBisCO, Phosphoglycerate kinase, G3P Dehydrogenase |\n"
            "| **Primary Products** | $\\text{O}_2$ (byproduct), $\\text{ATP}$, $\\text{NADPH}$ | $\\text{G3P} \\to \\text{Glucose} (\\text{C}_6\\text{H}_{12}\\text{O}_6)$, $\\text{ADP}$, $\\text{NADP}^+$ |\n"
            "| **Energy Source** | Solar radiation ($400-700\\text{ nm}$) | Chemical energy from ATP and NADPH |"
        ),
    ),
}

# Regex mapping user query patterns to canonical scientific laws
LAW_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\b(third|3rd)\s+(law|low)\s+of\s+thermodynamics\b|\bformula\s+of\s+(the\s+)?(third|3rd)\s+(law|low)\s+of\s+thermodynamics\b|\bthird\s+thermodynamic\s+law\b",
            re.IGNORECASE,
        ),
        "third_law_thermodynamics",
    ),
    (
        re.compile(
            r"\b(second|2nd)\s+(law|low)\s+of\s+thermodynamics\b|\bformula\s+of\s+(the\s+)?(second|2nd)\s+(law|low)\s+of\s+thermodynamics\b",
            re.IGNORECASE,
        ),
        "second_law_thermodynamics",
    ),
    (
        re.compile(
            r"\b(first|1st)\s+(law|low)\s+of\s+thermodynamics\b|\bformula\s+of\s+(the\s+)?(first|1st)\s+(law|low)\s+of\s+thermodynamics\b",
            re.IGNORECASE,
        ),
        "first_law_thermodynamics",
    ),
    (
        re.compile(
            r"\b(zeroth|0th)\s+(law|low)\s+of\s+thermodynamics\b|\bformula\s+of\s+(the\s+)?(zeroth|0th)\s+(law|low)\s+of\s+thermodynamics\b",
            re.IGNORECASE,
        ),
        "zeroth_law_thermodynamics",
    ),
    (
        re.compile(
            r"\bideal\s+gas\s+law\b|\bformula\s+of\s+(the\s+)?ideal\s+gas\b|\bpv\s*=\s*nrt\b",
            re.IGNORECASE,
        ),
        "ideal_gas_law",
    ),
    (
        re.compile(
            r"\bnewton'?s?\s+(second|2nd)\s+law\b|\bformula\s+of\s+newton'?s?\s+(second|2nd)\s+law\b|\bf\s*=\s*ma\b",
            re.IGNORECASE,
        ),
        "newtons_second_law",
    ),
    (
        re.compile(
            r"\b(law\s+of\s+universal\s+gravitation|newton'?s?\s+law\s+of\s+gravity|gravitational\s+force\s+formula)\b",
            re.IGNORECASE,
        ),
        "newtons_gravitation",
    ),
    (
        re.compile(
            r"\be\s*=\s*mc\^?2\b|\bmass\s+energy\s+equivalence\b|\beff?ect\s+of\s+mass\s+and\s+energy\b",
            re.IGNORECASE,
        ),
        "einstein_mass_energy",
    ),
    (
        re.compile(
            r"\bohm'?s?\s+law\b|\bv\s*=\s*ir\b|\bformula\s+of\s+ohm'?s?\s+law\b",
            re.IGNORECASE,
        ),
        "ohms_law",
    ),
    (
        re.compile(
            r"\bcoulomb'?s?\s+law\b|\belectrostatic\s+force\s+formula\b",
            re.IGNORECASE,
        ),
        "coulombs_law",
    ),
    (
        re.compile(
            r"\bphotosynthesis\b|\bprocess\s+of\s+photosynthesis\b|\bphotosynthetic\s+reaction\b",
            re.IGNORECASE,
        ),
        "photosynthesis",
    ),
]


def resolve_scientific_law(query: str) -> Optional[ScientificLaw]:
    """Deterministically resolves a scientific law query to an authoritative formulation."""
    normalized = query.lower().strip()
    for pattern, law_key in LAW_PATTERNS:
        if pattern.search(normalized):
            return SCIENTIFIC_LAWS.get(law_key)
    return None
