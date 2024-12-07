SLE_2021 = {

    'SIGNATURES_MAP' : {
        "S-650": "SLE w/ organ/system involvement",
        "S-1289": "High intensity SLE code",
        "S-1497": "Lupus nephritis w/ glomerunephritis", # Check kidney
        "S-1588": "SLE w/ high DNA double strand Ab",
        "S-1683": "Toxic maculopathy coded in autoimmune patients",
        "S-1378": "SLE derived cardiolipin IgG Ab syndrome", # Check APS
        "S-999" : "IgG expressing APS", # Check APS
        "S-1581": "SLE treated w/ HCQ",
        "S-295" : "Injuries",
        "S-788" : "Rheumatoid arthritis",
        "S-976" : "Sjögren's syndrome",
        "S-1475": "SLE derived cardiolipin IgA Ab syndrome", # Check APS
        "S-063" : "Protean symptoms (allergy and pain)",
        "S-1324" : "Coronary heart disease",
        "S-1756" : "Healthcare acquired vascular infection",
        "S-926" : "Anorectal benign tumor",
        "S-242" : "High-sensitivity C-reactive protein",
        "S-268" : "Lung infection",
        "S-1364" : "Vascular insufficiency of the intestine",
        "S-1897" : "Autoimmune inflammation treated w/ Prednisone",
        "S-1539" : "Collagen disease \& muscle disorder",
        "S-546" : "Early-onset aggressive lupus nephritis", # Check kidney
        "S-928" : "Autoimmune hepatitis", # Check kidney
        "S-835" : "Cutaneous lupus",
        "S-1544" : "Autoimmune disease w/ high SS-A/Ro Ab",
        "S-1417" : "Benign paroxysmal positional vertigo",
        "S-031" : "SLE-MCTD Overlap Syndrome",
        "S-1135" : "African-Ancestry SLE w/ cutaneous-renal predominance", # Check kidney
        "S-080" : "Early CKD", # Check Kidney
        "S-377" : "Aseptic necrosis of bone",
        "S-385" : "Raynaud's disease w/ high ANA",
        "S-353" : "Troublesome pregnancy",
        "S-1114" : "Head injury",
        "S-373" : "Cervical radiculopathy",
        "S-563" : "Non-white active autoimmune systemic inflammation labs",
        "S-1252" : "Tinea corporis",
        "S-1192" : "IgM expressing APS" # Check APS
    },

    'CHANNELS_MAP' : { # Mapping to for better strigns for each feature
        "Systemic lupus erythematosus with organ/system involvement":"SLE with organ/system involvement ",
        "Systemic lupus erythematosus":"SLE ",
        "DNA double strand Ab [Units/volume] in Serum":"ds-DNA Ab [Units/volume]in Serum",
        "Oxygen saturation in Blood":"$\\mathrm{O_2}$ saturation in Blood \\:",
        "Monocytes [#/volume] in Blood by Automated count":"Monocytes [\\#/volume] in Blood by Automated count ",
        "Ribonucleoprotein extractable nuclear IgG Ab [Units/volume] in Serum by Immunoassay":"RNP IgG Ab Units/volume] in Serum by Immunoassay",
        "SjÃ¶gren's syndrome" :"Sjöngren's syndrome "
    }
}


# %%
from CML_tool.Utils import look_up_description
from CML_tool.Utils import read_pickle


meta_df = read_pickle('/home/barbem4/projects/Data/Initial Data' , 'meta.pkl')
df = look_up_description(
    meta_df,
    description='BMI'
    )
# %%
