KNOWLEDGE_BASE = {
    "akiec": {
        "description": "Actinic keratoses are rough, scaly patches on the skin caused by years of sun exposure. They are considered pre-cancerous because a small percentage can develop into squamous cell carcinoma if left untreated.",
        "causes": "Prolonged UV exposure from sunlight or tanning beds. More common in fair-skinned individuals, people over 40, and those with a history of sunburns.",
        "appearance": "Rough, dry, scaly patches usually smaller than 2.5 cm. Color ranges from skin-toned to reddish-brown. Often found on sun-exposed areas like face, ears, scalp, neck, forearms, and backs of hands.",
        "risk_factors": "Fair skin, history of sunburn, age over 40, living in sunny climates, weakened immune system.",
        "when_to_worry": "If the lesion becomes tender, starts bleeding, grows rapidly, or becomes thicker than surrounding skin. About 5-10% of actinic keratoses may progress to squamous cell carcinoma.",
        "prevention": "Regular sunscreen use (SPF 30+), protective clothing, avoiding peak sun hours, regular skin checks.",
    },
    "bcc": {
        "description": "Basal cell carcinoma is the most common type of skin cancer. It grows slowly and rarely spreads to other parts of the body, but it can cause significant local tissue damage if untreated.",
        "causes": "Primarily caused by cumulative UV radiation exposure. Can also result from radiation therapy, chronic arsenic exposure, or genetic conditions like Gorlin syndrome.",
        "appearance": "Often appears as a pearly or waxy bump, a flat flesh-colored or brown scar-like lesion, or a bleeding or scabbing sore that heals and returns. May have visible blood vessels on the surface.",
        "risk_factors": "Chronic sun exposure, fair skin, history of sunburns, radiation therapy, age over 50, male sex, family history of skin cancer.",
        "when_to_worry": "Any new or changing bump or growth on the skin, especially on sun-exposed areas. A sore that doesn't heal within a few weeks should be evaluated.",
        "prevention": "Sun protection, avoid tanning beds, regular skin self-exams, annual dermatologist visits for high-risk individuals.",
    },
    "bkl": {
        "description": "Benign keratoses include seborrheic keratoses and solar lentigines. These are non-cancerous growths that are very common, especially in older adults. They do not become cancerous.",
        "causes": "Seborrheic keratoses are linked to genetics and aging. Solar lentigines (age spots) are caused by sun exposure over time.",
        "appearance": "Seborrheic keratoses appear as waxy, raised, brown or black growths with a stuck-on appearance. Solar lentigines are flat, brown spots. Both are usually painless.",
        "risk_factors": "Aging, family history, sun exposure for solar lentigines.",
        "when_to_worry": "Generally harmless, but should be evaluated if a growth changes rapidly, bleeds without injury, or looks different from your other spots. Sometimes melanoma can mimic seborrheic keratosis.",
        "prevention": "Sunscreen for solar lentigines. Seborrheic keratoses cannot be prevented.",
    },
    "df": {
        "description": "Dermatofibromas are common, harmless skin growths that form firm, small nodules under the skin. They are benign and do not become cancerous.",
        "causes": "The exact cause is unknown. They may develop as a reaction to a minor skin injury like an insect bite, thorn prick, or ingrown hair.",
        "appearance": "Small, firm, raised bumps usually 0.5-1 cm in diameter. Color ranges from pink to brown. Often found on the legs. They may feel like a small button under the skin and typically dimple inward when pinched.",
        "risk_factors": "More common in women, adults aged 20-50. May appear after minor trauma to the skin.",
        "when_to_worry": "Usually no concern needed. See a doctor if the growth is painful, changes size rapidly, or looks unusual compared to typical dermatofibromas.",
        "prevention": "No known prevention methods.",
    },
    "mel": {
        "description": "Melanoma is the most serious type of skin cancer. It develops in the cells that give skin its color (melanocytes). Early detection is critical as melanoma can spread to other organs if not caught early.",
        "causes": "UV radiation is the primary cause. Can also develop from existing moles. Genetic factors play a significant role.",
        "appearance": "Often identified using the ABCDE rule: Asymmetry (one half doesn't match the other), Border irregularity (ragged or blurred edges), Color variation (multiple shades of brown, black, red, white, or blue), Diameter greater than 6mm, and Evolving (changing in size, shape, or color).",
        "risk_factors": "History of severe sunburns, many moles (50+), fair skin, family history of melanoma, weakened immune system, previous melanoma.",
        "when_to_worry": "Any new mole or existing mole that changes in size, shape, color, or begins to itch, bleed, or crust. Any lesion matching ABCDE criteria needs immediate evaluation.",
        "prevention": "Rigorous sun protection, monthly skin self-exams, annual full-body skin checks with a dermatologist, avoid tanning beds.",
    },
    "nv": {
        "description": "Melanocytic nevi are common moles. They are benign growths of melanocytes (pigment-producing cells). Most adults have 10-40 moles. The vast majority of moles never become cancerous.",
        "causes": "Develop from clusters of melanocytes. Influenced by genetics and sun exposure during childhood and adolescence.",
        "appearance": "Usually round or oval, smaller than 6mm, with a uniform color (tan, brown, or flesh-toned). Can be flat or raised, smooth or rough. Some moles may have hair growing from them.",
        "risk_factors": "Fair skin, sun exposure during youth, genetic predisposition. Having many moles (50+) increases melanoma risk slightly.",
        "when_to_worry": "Monitor using ABCDE criteria. Any mole that changes significantly should be evaluated. New moles appearing after age 30 deserve attention.",
        "prevention": "Sun protection especially in childhood. Regular monitoring of existing moles.",
    },
    "vasc": {
        "description": "Vascular lesions include cherry angiomas, angiokeratomas, and pyogenic granulomas. These are benign growths of blood vessels in the skin. They are not cancerous and rarely cause health problems.",
        "causes": "Cherry angiomas are linked to aging and genetics. Angiokeratomas involve dilated blood vessels. Pyogenic granulomas often follow minor trauma.",
        "appearance": "Cherry angiomas are small, bright red dome-shaped spots. Angiokeratomas are dark red to black rough papules. Pyogenic granulomas are rapidly growing red nodules that bleed easily.",
        "risk_factors": "Aging, pregnancy (for certain types), minor skin injuries.",
        "when_to_worry": "Usually harmless. Seek evaluation if a vascular lesion bleeds frequently, grows rapidly, or looks unusual.",
        "prevention": "No known prevention for most vascular lesions.",
    },
}


def get_context(class_code: str) -> str:
    info = KNOWLEDGE_BASE.get(class_code)
    if not info:
        return ""

    context = f"""VERIFIED MEDICAL INFORMATION ABOUT THIS CONDITION:

Description: {info['description']}

Typical appearance: {info['appearance']}

Common causes: {info['causes']}

Risk factors: {info['risk_factors']}

When to be concerned: {info['when_to_worry']}

Prevention: {info['prevention']}"""

    return context