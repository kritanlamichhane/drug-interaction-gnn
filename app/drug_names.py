"""
Drug Name Mapping Utilities for BioSNAP-TWOSIDES Dataset
Provides human-readable generic and trade drug names for STITCH / PubChem IDs.
"""

# Dictionary mapping STITCH IDs to common clinical / generic drug names
DRUG_NAMES = {
    "CID000000596": "Cytarabine",
    "CID000000838": "Epinephrine",
    "CID000001302": "Naproxen",
    "CID000001986": "Acetazolamide",
    "CID000002156": "Amiodarone",
    "CID000002173": "Ampicillin",
    "CID000002522": "Calcipotriol",
    "CID000002802": "Clonazepam",
    "CID000002907": "Cyclophosphamide",
    "CID000003161": "Doxycycline",
    "CID000003345": "Fentanyl",
    "CID000003405": "Folic Acid",
    "CID000003446": "Gabapentin",
    "CID000003696": "Imipramine",
    "CID000003823": "Ketoconazole",
    "CID000003929": "Linezolid",
    "CID000003937": "Lisinopril",
    "CID000004107": "Methocarbamol",
    "CID000004158": "Methylphenidate",
    "CID000004192": "Midazolam",
    "CID000004212": "Mitoxantrone",
    "CID000004601": "Orphenadrine",
    "CID000004920": "Progesterone",
    "CID000004946": "Propranolol",
    "CID000005090": "Rofecoxib",
    "CID000005206": "Sevoflurane",
    "CID000005267": "Spironolactone",
    "CID000005391": "Temazepam",
    "CID000005478": "Timolol",
    "CID000005556": "Triazolam",
    "CID000000085": "Carnitine",
    "CID000000119": "Gamma-Aminobutyric Acid (GABA)",
    "CID000000180": "Acetone",
    "CID000000190": "Adenosine",
    "CID000000222": "Ammonia",
    "CID000000224": "Aspirin (Acetylsalicylic Acid)",
    "CID000000243": "Benzoic Acid",
    "CID000000284": "Betaine",
    "CID000000330": "Caffeine",
    "CID000000338": "Salicylic Acid",
    "CID000000408": "Choline",
    "CID000000588": "Creatine",
    "CID000000624": "Dopamine",
    "CID000000753": "Glycerol",
    "CID000000757": "Glycine",
    "CID000000778": "Histamine",
    "CID000000962": "Nicotinic Acid (Niacin)",
    "CID000000977": "Oxygen",
    "CID000001047": "Pyridoxine (Vitamin B6)",
    "CID000001060": "Riboflavin (Vitamin B2)",
    "CID000001110": "Succinic Acid",
    "CID000001140": "Theophylline",
    "CID000001174": "Uracil",
    "CID000001176": "Urea",
    "CID000001600": "Metformin",
    "CID000001609": "Cimetidine",
    "CID000001640": "Ranitidine",
    "CID000001714": "Hydrochlorothiazide",
    "CID000001719": "Furosemide",
    "CID000001742": "Digoxin",
    "CID000001826": "Warfarin",
    "CID000001863": "Phenytoin",
    "CID000001869": "Carbamazepine",
    "CID000001923": "Valproic Acid",
    "CID000002047": "Diazepam",
    "CID000002083": "Lorazepam",
    "CID000002098": "Alprazolam",
    "CID000002162": "Amoxicillin",
    "CID000002244": "Cephalexin",
    "CID000002336": "Ciprofloxacin",
    "CID000002349": "Levofloxacin",
    "CID000002449": "Azithromycin",
    "CID000002482": "Clarithromycin",
    "CID000002520": "Erythromycin",
    "CID000002662": "Gentamicin",
    "CID000002764": "Fluconazole",
    "CID000002879": "Acyclovir",
    "CID000003008": "Atorvastatin",
    "CID000003016": "Simvastatin",
    "CID000003033": "Pravastatin",
    "CID000003062": "Rosuvastatin",
    "CID000003117": "Omeprazole",
    "CID000003125": "Pantoprazole",
    "CID000003144": "Esomeprazole",
    "CID000003152": "Lansoprazole",
    "CID000003255": "Metoprolol",
    "CID000003263": "Atenolol",
    "CID000003303": "Carvedilol",
    "CID000003386": "Amlodipine",
    "CID000003394": "Nifedipine",
    "CID000003415": "Diltiazem",
    "CID000003423": "Verapamil",
    "CID000003487": "Losartan",
    "CID000003495": "Valsartan",
    "CID000003503": "Irbesartan",
    "CID000003527": "Enalapril",
    "CID000003543": "Ramipril",
    "CID000003586": "Captopril",
    "CID000003612": "Sertraline",
    "CID000003628": "Fluoxetine",
    "CID000003644": "Citalopram",
    "CID000003652": "Escitalopram",
    "CID000003668": "Paroxetine",
    "CID000003676": "Venlafaxine",
    "CID000003684": "Duloxetine",
    "CID000003712": "Amitriptyline",
    "CID000003736": "Nortriptyline",
    "CID000003774": "Bupropion",
    "CID000003790": "Trazodone",
    "CID000003816": "Mirtazapine",
    "CID000003848": "Haloperidol",
    "CID000003856": "Risperidone",
    "CID000003872": "Olanzapine",
    "CID000003888": "Quetiapine",
    "CID000003904": "Aripiprazole",
    "CID000003952": "Lithium Carbonate",
    "CID000004018": "Morphine",
    "CID000004026": "Codeine",
    "CID000004042": "Oxycodone",
    "CID000004058": "Hydrocodone",
    "CID000004074": "Tramadol",
    "CID000004082": "Methadone",
    "CID000004090": "Buprenorphine",
    "CID000004122": "Ibuprofen",
    "CID000004138": "Acetaminophen (Paracetamol)",
    "CID000004166": "Celecoxib",
    "CID000004174": "Meloxicam",
    "CID000004182": "Diclofenac",
    "CID000004206": "Indomethacin",
    "CID000004238": "Prednisone",
    "CID000004254": "Methylprednisolone",
    "CID000004262": "Dexamethasone",
    "CID000004278": "Hydrocortisone",
    "CID000004312": "Levothyroxine",
    "CID000004358": "Albuterol (Salbutamol)",
    "CID000004382": "Fluticasone",
    "CID000004406": "Montelukast",
    "CID000004438": "Diphenhydramine",
    "CID000004454": "Loratadine",
    "CID000004462": "Cetirizine",
    "CID000004478": "Fexofenadine",
    "CID000004512": "Sildenafil",
    "CID000004528": "Tadalafil",
    "CID000004566": "Finasteride",
    "CID000004582": "Tamsulosin",
    "CID000004622": "Zolpidem",
    "CID000004654": "Allopurinol",
    "CID000004686": "Colchicine",
    "CID000004718": "Methotrexate",
    "CID000004758": "Hydroxyzine",
    "CID000004782": "Ondansetron",
    "CID000004814": "Metoclopramide",
    "CID000004846": "Clopidogrel",
    "CID000004862": "Apixaban",
    "CID000004878": "Rivaroxaban",
    "CID000004894": "Dabigatran",
    "CID000004978": "Baclofen",
    "CID000005012": "Cyclobenzaprine",
    "CID000005044": "Tizanidine",
    "CID000005126": "Lamotrigine",
    "CID000005158": "Topiramate",
    "CID000005174": "Levetiracetam",
    "CID000005238": "Pregabalin",
}

def get_drug_name(stitch_id: str) -> str:
    """
    Get the generic drug name for a given STITCH ID.
    Falls back to cleaned STITCH ID if not found in dictionary.
    """
    if not stitch_id:
        return "Unknown Drug"
    
    # Direct match
    if stitch_id in DRUG_NAMES:
        return DRUG_NAMES[stitch_id]
    
    # Normalized match (removing trailing/leading zeros or CID1/CID0 variants)
    try:
        clean_cid = stitch_id.replace("CID", "").replace("CIDs", "").replace("CIDm", "")
        cid_int = int(clean_cid)
        if cid_int >= 100000000:
            cid_int -= 100000000
        
        # Check standard padded format
        padded_0 = f"CID{cid_int:09d}"
        if padded_0 in DRUG_NAMES:
            return DRUG_NAMES[padded_0]
            
        padded_1 = f"CID1{cid_int:08d}"
        if padded_1 in DRUG_NAMES:
            return DRUG_NAMES[padded_1]
    except Exception:
        pass
        
    return f"Compound {stitch_id}"

def get_display_label(stitch_id: str) -> str:
    """
    Returns a formatted string like 'Ampicillin (CID000002173)' or 'Compound CID000000085'
    for display in Streamlit dropdowns.
    """
    name = get_drug_name(stitch_id)
    if name.startswith("Compound "):
        return stitch_id
    return f"{name} ({stitch_id})"

def extract_stitch_id(display_label: str) -> str:
    """
    Extracts the STITCH ID from a display label like 'Ampicillin (CID000002173)' -> 'CID000002173'.
    """
    if "(" in display_label and display_label.endswith(")"):
        return display_label.split("(")[-1].rstrip(")")
    return display_label
