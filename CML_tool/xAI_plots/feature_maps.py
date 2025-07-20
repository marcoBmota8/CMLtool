from matplotlib.pyplot import cm, get_cmap

SLE_2021 = {
    'SIGNATURES_MAP' : {
        "S-650": "SLE w/ organ/system involvement",
        "S-1289": "High intensity SLE code",
        "S-1497": "Lupus nephritis", 
        "S-1588": "SLE w/ high DNA double strand Ab",
        "S-1683": "Toxic maculopathy coded in autoimmune patients",
        "S-1378": "Elevated anti-cardiolipin IgG \& IgM Ab", 
        "S-999" : r"Elevated anti-$\beta$2 \& anti-cardiolipin IgG Ab",
        "S-1581": "SLE treated w/ HCQ",
        "S-295" : "General injuries",
        "S-788" : "Rheumatoid arthritis",
        "S-976" : "Keratoconjunctivitis sicca in Sjögren's syndrome",
        "S-1475": "Elevated anti-cardiolipin IgA \& IgG Ab", 
        "S-063" : "Protean symptoms (allergy and pain)", 
        "S-1324" : "Coronary heart disease",
        "S-1756" : "Healthcare acquired vascular infection",
        "S-926" : "Anorectal benign tumor",
        "S-242" : "High-sensitivity C-reactive protein",
        "S-268" : "Non-tuberculosis mycobacterial pneumonia", # complication of immunosuppression
        'S-1364': 'Vascular insufficiency of the intestine',
        "S-1897" : "Autoimmune inflammation treated w/ Prednisone",
        "S-1539" : "Collagen disease, muscle disorder \& SLE", 
        "S-546" : "Aggressive \& chronic lupus nephritis", 
        "S-928" : "Autoimmune hepatitis",
        "S-835" : "Cutaneous lupus", # Lupus Erythematosus (SNOMED: 200936003 maps to L93, L93.2 & 695.4) -> L ICD10 family is diseases of the skin 
        # cutaneous lupus were considered negatives and still predictive as a positive source.
        "S-1544" : "Elevated SS-A/Ro Ab", 
        "S-1417" : "Dizziness \& vertigo",
        "S-031" : "Mixed connective tissue disease",
        "S-1135" : "Discoid lupus w/ glomerunephritis", 
        "S-080" : "Early CKD",
        "S-377" : "Aseptic necrosis of bone", # Corticosteroids cause this
        "S-385" : "Raynaud's",
        "S-353" : "Pregnancy complication in group B streptococcus",
        "S-1114" : "Head injury (contusion)",
        "S-373" : "Cervical radiculopathy",
        "S-563" : "Elevated IgG, IgA, and erythrocyte sedimentation rate",
        "S-1252" : "Tinea corporis", # Speculation is misDx of some cutaneous lupus
        "S-1192" : r"Elevated anti-$\beta$2 \& anti-cardiolipin IgM Ab",
        "S-717" : "Coded antiphospholipid syndrome",
        "S-1576" : "Systemic sclerosis",# SS was a Dx excluded from the definitins of SLE and still predictive
        "S-1300" : "Elevated specific gravity \& pH of Urine", # Is usually not looked at, unremarkable
        "S-766" : "Diazepam treatment",
        "S-100" : "Late-stage pregnancy complication",
        "S-1853" : "Benign prostatic hypertrophy w/ outflow obstruction tendency",
        "S-288" : "Specific visual field defect", # Complicated, might be from the eye exam 
        "S-1738" : "Alopecia, female tendency",
        "S-1717" : "Hematuria",
        "S-1924" : "Elevated antithrombin in renal transplant", 
        "S-657" : "Mycophenolate mofetil treatment",
        "S-1678" : "Undifferentiated connective tissue disease",
        "S-510" : "Age-related vision deterioration", 
        "S-813" : "Elevated CSF albumin and IgG",
        "S-658" : "Diplopia",
        "S-1238" : "Elevated immature granulocytes",
        "S-1494" : "Sjögren's syndrome",
        "S-539" : "Elevated nuclear anti-SCL-70 Ab", 
        "S-827" : "Elevated anti-Smith Ab", 
        "S-316" : "$\mathrm{FeSO_4}$ supplementation in anemia",
        "S-1348" : "Advanced (stage 3) CKD", # check (ask mike) 
        "S-1371" : "Non-ideopathic degenerative joint disease",
        "S-1413" : "Dermatomyositis", #-> Excluded in SLE definition so partial matches only 
        "S-1682" : "Elevated SS-A/Ro \& SS-B/La Ab", 
        "S-1062" : "Low back spondylosis",
        "S-735" : "Elevated creatinine in urine",
        "S-237" : "Pain in right knee", 
        "S-473" : "Coded high antibody titer w/ osteoarthritis", 
        "S-559" : "Elevated thyroglobulin", 
        "S-980" : "Gabapentin treatment", 
        "S-1936" : "Difficulty walking from lower body weakness", 
        "S-551" : "Elevated ANA titer lab result",
        "S-448" : "Contact dermatitis",
        "S-1279" : "Elevated erythrocyte sedimentation rate w/ female tendency",
        "S-1277" : "Elevated erythrocyte sedimentation rate w/ black race tendency",
        "S-770" : "Elevated metamyelocytes in CSF", # Indicate inflamation
        "S-1808": "Wrist joint pain",
        "S-069": "Sleep apnea",
        "S-384": "Skin sensation disturbance",
        "S-300": "Elevated QRS axis",
        "S-1744": "Asteatosis cutis \& keratosis",
        "S-1649": "Protein in urine",
        "S-1737": "Bee/wasp bite reactions",
        "S-1194": "Treatment w/ diphenhydramine",
        "S-517": "CABG to treat coronary atherosclerosis",
        "S-986" : "Benign prostatic hypertrophy w/o outflow obstruction tendency",
        "S-1498" : "Closed rib(s) fracture",
        "S-1793" : "Blood coagulation disorder w/ elevated lupus anticoagulants Ab",
        "S-1385" : "Degenerative joint disease in lower extremeties",
        "S-476" : "Male decreased libido from allergic rhinitis sleep deprivation",
        "S-001" : "Age-related macular degeneration",
        "S-1815" : "Citalopram treatment",
        "S-1599" : "Hydroxyzine treatment for inflammatory dermatosis",
        "S-1549" : "Inflammatory dermatosis",
        "S-1021" : "Coded proteinuria",
        "S-556" : "Intervertebral disk degeneration",
        "S-1167": "Myogenic ptosis",
        "S-771" : "Personality disorder",
        "S-1975" : "Pneumococcal pneumonia in multiple myeloma",
        "S-114" : "Mood disorders",
        "S-1172" : "Pain in right foot",
        "S-1226": "Transient cerebral ischemia from aneurysm",
        "S-1100": "Clonazepan treatment",
        "S-1592": "Idiopathic peripheral neuropathy",
        "S-324": "Subarachnoid hemorrhage",
        "S-331": "Diclofenac treatment", # treat migraines and headaches
        "S-944": "Podiatric conditions",
        "S-1115": "Elevated Q-T interval",
        "S-131": "Procedure complication other than bleeding",
        "S-1847": "Tachycardia",
        "S-1858": "Critical illness complications",
        "S-595": "Elevated lactate",
        "S-614": "Celecoxib treatment", # pain medication in osteoarthritis
        "S-1331": "GERD w/ Stricture of esophagus",
        "S-265": "Peripheral neuritis in lower back",
        "S-1950": "Head \& neck pain",
        "S-312": "Cyst of ovary",
        "S-1177": "Neurological disorder due to diabetes type 2",
        "S-1097": "Spasmodic torticollis",
        "S-1781": "Mononeuritis",
        "S-145": "Hypertensive heart failure",
        "S-266": "Acquired deformity of ankle-foot",
        "S-1648": "Low-grade squamous intraepithelial lesion",
        "S-544": "Rheumatoid arthritis treated with methotrexate",
        "S-1013": "Hypothyroidism",
        "S-1638": "Asthenia",
        "S-239": "Neoplasm of uncertain behavior of skin",
        "S-378": "Elevated ferritin",
        "S-1245":"Elevated likelihood of `Black` vs. `White` recorded race"
        },

    'CHANNELS_MAP' : { # Mapping to for better strigns for each feature
        "Systemic lupus erythematosus with organ/system involvement":"SLE w/ organ/system involvement",
        "Systemic lupus erythematosus":"SLE",
        "DNA double strand Ab":"ds-DNA Ab",
        "[Units/volume]" : "[Units/vol.]",
        "Oxygen saturation in Blood":"$\\mathrm{O_2}$ saturation in blood",
        "[#/volume]" : "[$\#$/vol.]",
        "[#/area]" : "[$\#$/area.]",
        "volume" : "vol.",
        "Ribonucleoprotein extractable nuclear IgG Ab":"snRNP IgG Ab",
        "SjÃ¶gren's" :"Sjöngren's",
        "Estimation of glomerular filtration rate" : "eGFR",
        'Antithrombin actual/normal in Platelet poor plasma by Chromogenic method' : "Antithrombin actual/normal in PPP (chromogenic method)"
    }
}


race_cohort_map = {
    'White': (
        '#1f6cb4',
        '#2cc4f8'
        ),
    'Black': (
        '#f5770a',
        '#f79f52'
        ),
    'Both': (
        '#0a090a',
        '#5e5b5e'
        ),
}