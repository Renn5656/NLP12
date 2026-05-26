HIGH_RISK_KEYWORDS = {
    "guaranteed profit": 10,
    "zero risk": 10,
    "double your money": 10,
    "join now": 5,
    "limited quota": 5,
    "vip": 5,
    "risk-free": 10,
    "urgent": 5,
}

BASE_SCORES = {
    "Normal": 20,
    "Suspicious": 55,
    "Fraud": 85,
}


def detect_suspicious_keywords(text):
    normalized = str(text).lower()
    return [keyword for keyword in HIGH_RISK_KEYWORDS if keyword in normalized]


def calculate_risk_score(predicted_label, text):
    score = BASE_SCORES.get(predicted_label, 55)
    detected_keywords = detect_suspicious_keywords(text)
    score += sum(HIGH_RISK_KEYWORDS[keyword] for keyword in detected_keywords)
    return min(score, 100), detected_keywords


def calculate_risk_breakdown(predicted_label, text):
    base_score = BASE_SCORES.get(predicted_label, 55)
    detected_keywords = detect_suspicious_keywords(text)
    keyword_bonus = [
        {"Keyword": keyword, "Bonus": HIGH_RISK_KEYWORDS[keyword]}
        for keyword in detected_keywords
    ]
    raw_score = base_score + sum(item["Bonus"] for item in keyword_bonus)
    final_score = min(raw_score, 100)
    return {
        "base_score": base_score,
        "keyword_bonus": keyword_bonus,
        "raw_score": raw_score,
        "final_score": final_score,
        "score_cap_applied": raw_score > 100,
    }


def get_risk_level(score):
    if score <= 35:
        return "Low Risk"
    if score <= 70:
        return "Medium Risk"
    return "High Risk"


if __name__ == "__main__":
    sample = "Guaranteed profit with zero risk. Join our VIP crypto group now!"
    risk_score, keywords = calculate_risk_score("Fraud", sample)
    print(f"Risk Score: {risk_score}")
    print(f"Risk Level: {get_risk_level(risk_score)}")
    print(f"Detected Keywords: {', '.join(keywords) if keywords else 'None'}")
