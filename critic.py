import re

PASS_SCORE = 80

POLITE_WORDS = ["مشتری گرامی", "گرامی", "عزیز", "محترم", "سپاس", "ممنون", "متشکر"]
EMPATHY_WORDS = ["درک می کنیم", "درک می کنم", "ناراحتی شما", "نگرانی شما", "حق دارید", "شرایط شما"]
APOLOGY_WORDS = ["عذرخواهی", "پوزش", "متاسف", "ببخشید"]
ACTION_WORDS = [
    "ارجاع", "بررسی", "رسیدگی", "ثبت شد", "ثبت کرد", "بازگردانده", "استرداد",
    "بازگشت وجه", "تیم فنی", "واحد مالی", "پشتیبانی فنی", "اصلاح", "فعال سازی",
]
FOLLOW_UP_WORDS = ["ساعت آینده", "روز کاری", "پیامک", "ایمیل", "تماس", "کد پیگیری", "شماره پیگیری"]
FOLLOW_UP_TIME = re.compile(r"[0-9۰-۹]+\s*(ساعت|روز|دقیقه)")
FORBIDDEN_PROMISES = [
    "تضمین", "قطعا", "صد در صد", "۱۰۰ درصد", "100 درصد", "۱۰۰٪", "100%",
    "همین الان", "فورا", "بلافاصله",
]

WEIGHTS = {
    "complaint": {"polite": 8, "empathy": 7, "apology": 20, "action": 25, "follow_up": 20, "no_promise": 10, "length": 10},
    "question": {"polite": 40, "no_promise": 30, "length": 30},
}

WORD_LIMITS = {
    "complaint": (40, 150),
    "question": (15, 120),
}

FEEDBACK = {
    "polite": "Address the customer respectfully (e.g. 'مشتری گرامی').",
    "empathy": "Show empathy: acknowledge the customer's frustration explicitly.",
    "apology": "Add a clear apology (e.g. 'عذرخواهی می‌کنیم').",
    "action": "State one specific next action the support team is taking (e.g. referring the case to the finance team).",
    "follow_up": "Give a concrete follow-up time or channel (e.g. within 48 hours via SMS).",
}


def normalize(text):
    text = text.replace("‌", " ").replace("ي", "ی").replace("ك", "ک")
    return re.sub(r"[ً-ٟ]", "", text)


def contains_any(text, words):
    return any(word in text for word in words)


def score_reply(reply, intent="complaint"):
    text = normalize(reply or "")
    if not text.strip():
        return 0, ["The reply is empty. Write a complete reply."]

    profile = "question" if intent == "question" else "complaint"
    min_words, max_words = WORD_LIMITS[profile]
    word_count = len(text.split())
    promises = [word for word in FORBIDDEN_PROMISES if word in text]

    checks = {
        "polite": contains_any(text, POLITE_WORDS),
        "empathy": contains_any(text, EMPATHY_WORDS),
        "apology": contains_any(text, APOLOGY_WORDS),
        "action": contains_any(text, ACTION_WORDS),
        "follow_up": bool(FOLLOW_UP_TIME.search(text)) or contains_any(text, FOLLOW_UP_WORDS),
        "no_promise": not promises,
        "length": min_words <= word_count <= max_words,
    }
    feedback_texts = {
        **FEEDBACK,
        "no_promise": f"Remove guaranteed/unauthorized promises: {', '.join(promises)}.",
        "length": f"Keep the reply between {min_words} and {max_words} words (currently {word_count}).",
    }

    score = 0
    feedback = []
    for name, weight in WEIGHTS[profile].items():
        if checks[name]:
            score += weight
        else:
            feedback.append(feedback_texts[name])

    return score, feedback
