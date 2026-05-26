BASE_SCORES = {
    "Normal": 15,
    "Suspicious": 40,
    "Fraud": 65,
}

RISK_SIGNALS = [
    {
        "category": "Guaranteed profit claim",
        "keywords": ["guaranteed profit", "guaranteed return", "guaranteed income"],
        "bonus": 8,
    },
    {
        "category": "No-risk claim",
        "keywords": ["zero risk", "risk-free", "no risk", "absolutely safe"],
        "bonus": 8,
    },
    {
        "category": "Unrealistic return",
        "keywords": ["double your money", "high return", "20%", "30%", "50%", "daily profit"],
        "bonus": 7,
    },
    {
        "category": "Crypto or digital asset payment",
        "keywords": ["crypto", "cryptocurrency", "bitcoin", "btc", "usdt", "wallet", "mining"],
        "bonus": 5,
    },
    {
        "category": "VIP or private group lure",
        "keywords": ["vip", "private group", "trading group", "investment group", "mentor"],
        "bonus": 4,
    },
    {
        "category": "Urgency or scarcity",
        "keywords": ["urgent", "join now", "limited quota", "today only", "last chance"],
        "bonus": 4,
    },
    {
        "category": "Advance fee or withdrawal obstacle",
        "keywords": ["withdrawal fee", "unlock fee", "tax payment", "pay tax", "deposit more"],
        "bonus": 10,
    },
    {
        "category": "Fake platform claim",
        "keywords": ["trading platform", "mining platform", "investment website", "proprietary system"],
        "bonus": 5,
    },
    {
        "category": "Relationship investment pattern",
        "keywords": ["wrong number", "old friend", "dating", "relationship", "trust me"],
        "bonus": 5,
    },
    {
        "category": "High-risk payment method",
        "keywords": ["bank transfer", "wire transfer", "crypto payment"],
        "bonus": 7,
    },
]

RISK_EXPLANATIONS = {
    "Guaranteed profit claim": "The message promises guaranteed profit, which is a common red flag because real investments cannot guarantee returns.",
    "No-risk claim": "The message claims there is little or no risk, but legitimate investments always carry some level of risk.",
    "Unrealistic return": "The message suggests unusually high or fast returns, which is often used to lure victims into investment scams.",
    "Crypto or digital asset payment": "The message mentions cryptocurrency or digital wallets, which are frequently used in current investment fraud because transactions are difficult to reverse.",
    "VIP or private group lure": "The message tries to move the user into a VIP, private, or trading group, which is a common tactic for building trust and pressure.",
    "Urgency or scarcity": "The message creates urgency or limited availability, which can pressure users to act before verifying the investment.",
    "Advance fee or withdrawal obstacle": "The message asks for fees, taxes, or extra deposits before withdrawal, a strong warning sign of advance-fee investment fraud.",
    "Fake platform claim": "The message refers to a trading, mining, or investment platform, which may be used to make fake profits look legitimate.",
    "Relationship investment pattern": "The message resembles relationship-based investment fraud, where scammers build trust before introducing an investment.",
    "High-risk payment method": "The message mentions bank transfer, wire transfer, or crypto payment, which are common payment channels in reported scam losses.",
}


def detect_suspicious_keywords(text):
    normalized = str(text).lower()
    detected = []
    for signal in RISK_SIGNALS:
        for keyword in signal["keywords"]:
            if keyword in normalized:
                detected.append(keyword)
    return detected


def detect_risk_signals(text):
    normalized = str(text).lower()
    detected = []
    for signal in RISK_SIGNALS:
        matched_keywords = [keyword for keyword in signal["keywords"] if keyword in normalized]
        if matched_keywords:
            detected.append(
                {
                    "Category": signal["category"],
                    "Matched Keywords": ", ".join(matched_keywords),
                    "Bonus": signal["bonus"],
                }
            )
    return detected


def calculate_combination_bonus(detected_signals):
    categories = {signal["Category"] for signal in detected_signals}
    bonus = 0
    if "Advance fee or withdrawal obstacle" in categories and (
        "Crypto or digital asset payment" in categories or "High-risk payment method" in categories
    ):
        bonus += 8
    if "Guaranteed profit claim" in categories and "No-risk claim" in categories:
        bonus += 5
    if "Relationship investment pattern" in categories and "Crypto or digital asset payment" in categories:
        bonus += 5
    return bonus


def calculate_risk_score(predicted_label, text):
    score = BASE_SCORES.get(predicted_label, 50)
    detected_signals = detect_risk_signals(text)
    detected_keywords = detect_suspicious_keywords(text)
    score += sum(signal["Bonus"] for signal in detected_signals)
    score += calculate_combination_bonus(detected_signals)
    return min(score, 100), detected_keywords


def calculate_risk_breakdown(predicted_label, text):
    base_score = BASE_SCORES.get(predicted_label, 50)
    keyword_bonus = detect_risk_signals(text)
    combination_bonus = calculate_combination_bonus(keyword_bonus)
    raw_score = base_score + sum(item["Bonus"] for item in keyword_bonus) + combination_bonus
    final_score = min(raw_score, 100)
    return {
        "base_score": base_score,
        "keyword_bonus": keyword_bonus,
        "raw_score": raw_score,
        "final_score": final_score,
        "score_cap_applied": raw_score > 100,
        "combination_bonus": combination_bonus,
    }


def generate_risk_explanation(predicted_label, score, detected_signals):
    signal_phrases = {
        "Guaranteed profit claim": "it promises guaranteed returns",
        "No-risk claim": "it claims there is little or no risk",
        "Unrealistic return": "it suggests unusually high returns",
        "Crypto or digital asset payment": "it involves cryptocurrency or digital wallets",
        "VIP or private group lure": "it invites the user into a private investment group",
        "Urgency or scarcity": "it pressures the user to act quickly",
        "Advance fee or withdrawal obstacle": "it asks for extra fees before withdrawal",
        "Fake platform claim": "it refers to a trading or investment platform",
        "Relationship investment pattern": "it uses a trust-building relationship pattern",
        "High-risk payment method": "it mentions hard-to-reverse payment methods",
    }

    phrases = [
        signal_phrases[signal["Category"]]
        for signal in detected_signals
        if signal["Category"] in signal_phrases
    ]

    if phrases:
        selected_phrases = phrases[:3]
        reason_text = ", ".join(selected_phrases)
        if len(phrases) > 3:
            reason_text += ", and other warning signs"
    else:
        reason_text = "the model found limited scam-related wording"

    if predicted_label == "Fraud":
        model_text = "the model classifies it as fraud"
    elif predicted_label == "Suspicious":
        model_text = "the model finds it suspicious"
    else:
        model_text = "the model classifies it as normal"

    if score >= 81:
        level_text = "very high risk"
    elif score >= 61:
        level_text = "high risk"
    elif score >= 41:
        level_text = "medium risk"
    elif score >= 26:
        level_text = "low risk"
    else:
        level_text = "very low risk"

    return f"This message is {level_text} because {model_text} and {reason_text}."


def get_risk_level(score):
    if score <= 25:
        return "Very Low Risk"
    if score <= 35:
        return "Low Risk"
    if score <= 60:
        return "Medium Risk"
    if score <= 80:
        return "High Risk"
    return "Very High Risk"


if __name__ == "__main__":
    sample = "Guaranteed profit with zero risk. Join our VIP crypto group now!"
    risk_score, keywords = calculate_risk_score("Fraud", sample)
    print(f"Risk Score: {risk_score}")
    print(f"Risk Level: {get_risk_level(risk_score)}")
    print(f"Detected Keywords: {', '.join(keywords) if keywords else 'None'}")
