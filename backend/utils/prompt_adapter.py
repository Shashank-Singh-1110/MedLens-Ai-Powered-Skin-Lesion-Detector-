from config import CLASSES
from backend.utils.knowledge_base import get_context


SYSTEM_PROMPT = """You are MedLens AI, a dermatology screening assistant. Follow these rules strictly:

MUST DO:
- Always state this is an AI screening tool, NOT a medical diagnosis
- Always recommend consulting a dermatologist for any concerning result
- Explain conditions in simple, non-medical language
- Be calm and non-alarming in tone even for serious conditions
- Mention that image quality and lighting can affect results
- Use ONLY the verified medical information provided to you — do not add claims beyond it

MUST NOT:
- Never prescribe medication or treatment
- Never say "you have cancer" or make definitive diagnostic claims
- Never discourage the patient from seeing a real doctor
- Never provide percentage survival rates or mortality statistics
- Never claim the AI is more accurate than a dermatologist
- Never discuss biopsy procedures or surgical options in detail

RESPONSE FORMAT:
Write 3-4 short paragraphs. No bullet points, no headers. Keep it under 200 words."""


def build_prompt(prediction: dict) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for LLaMA 3."""

    code = prediction["class_code"]
    name = prediction["class_name"]
    confidence = prediction["confidence"]
    severity = prediction["severity"]
    urgency = prediction["urgency"]
    top_3 = prediction["all_predictions"][:3]

    differentials = ", ".join(
        f"{CLASSES[cls]['name']} ({prob*100:.1f}%)"
        for cls, prob in top_3
    )

    # Confidence + margin analysis
    top_conf = top_3[0][1]
    second_conf = top_3[1][1] if len(top_3) > 1 else 0
    margin = top_conf - second_conf

    if confidence >= 0.8 and margin >= 0.3:
        certainty = "The model is fairly confident in this classification."
    elif margin < 0.15:
        second_name = CLASSES[top_3[1][0]]['name']
        certainty = (f"The model is uncertain between {name} and {second_name}. "
                     f"These conditions can look similar. Professional evaluation is strongly recommended.")
    elif confidence >= 0.5:
        certainty = "The model shows moderate confidence. A professional evaluation is recommended."
    else:
        certainty = "The model has low confidence. Strongly recommend professional consultation and do not lean into the primary diagnosis."

    # RAG: retrieve verified medical knowledge
    medical_context = get_context(code)

    user_prompt = f"""A skin lesion image was analyzed. Here are the results:

- Primary: {name} ({code})
- Confidence: {confidence*100:.1f}%
- Severity: {severity}
- Urgency: {urgency}
- Top 3: {differentials}
- {certainty}

{medical_context}

Using ONLY the verified medical information above, explain this result to the patient following your rules."""

    return SYSTEM_PROMPT, user_prompt

