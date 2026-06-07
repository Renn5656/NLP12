"""Build a reproducible investment-fraud classification dataset.

The generated rows are synthetic and explicitly marked as such. Long examples
are added to every class so message length is not a shortcut for the label.
"""

from pathlib import Path
import random

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "messages_raw.csv"
OUTPUT_PATH = BASE_DIR / "data" / "messages.csv"
SEED = 42

INVESTMENT_TERMS = (
    r"invest|trading|broker|stock|crypto|bitcoin|usdt|return|profit|platform|"
    r"wallet|forex|market|portfolio|fund|asset|share|dividend|withdraw"
)
TEMPLATE_LEAKAGE_TERMS = (
    r"\bsuspicious\b|\bfraud note\b|\bnormal note\b|\bframed\b|"
    r"\bcontext\b|\bexamples\b"
)


OPENERS = [
    "I am contacting you with an opportunity that has recently opened for selected investors.",
    "Our private advisory desk is inviting a limited number of participants to review a new investment program.",
    "A senior account manager has prepared a time-sensitive proposal for people interested in alternative assets.",
    "You were selected from our investor network to receive early access to a managed trading opportunity.",
    "Our international finance team is onboarding new clients for a private wealth program this week.",
]

FRAUD_SCENARIOS = [
    (
        "guaranteed_return",
        "The strategy guarantees a monthly return between 25 and 40 percent regardless of market conditions.",
        "Your principal is fully protected, so there is no possibility of losing funds.",
        "Transfer the starting capital today and the guaranteed profit will appear in your account within 24 hours.",
    ),
    (
        "crypto_wallet",
        "The program uses automated cryptocurrency arbitrage to produce daily profits from price differences.",
        "To activate the account, purchase USDT and send it directly to the wallet address supplied by the manager.",
        "The transfer cannot be reversed, but the platform promises to credit a large welcome bonus immediately.",
    ),
    (
        "withdrawal_fee",
        "Your investment dashboard currently shows a substantial profit and the withdrawal request has been approved.",
        "Before the funds can be released, the platform requires a tax clearance payment and an account unlock fee.",
        "Deposit the additional amount today or the withdrawal and all accumulated profits will be permanently cancelled.",
    ),
    (
        "impersonation",
        "I represent a well-known investment institution and can place your funds into an unpublished high-yield product.",
        "This offer is confidential and cannot be verified through the institution's normal customer service channels.",
        "Send the deposit to the account provided by my assistant before the private allocation closes.",
    ),
    (
        "romance_investment",
        "After getting to know you, I want to share the investment method that created most of my personal wealth.",
        "My relative has inside access to a trading platform and can tell us exactly when to buy and sell.",
        "Create an account through my private link and deposit funds so I can guide every trade for you.",
    ),
    (
        "pump_and_dump",
        "Our private group has advance information about a small stock that is expected to rise dramatically.",
        "Members are instructed to buy before the public announcement and are promised several times their original investment.",
        "Act immediately and do not discuss the recommendation outside the group until the price target is reached.",
    ),
]

SUSPICIOUS_SCENARIOS = [
    (
        "private_group",
        "The message invites recipients to a private investment group that shares selected market opportunities.",
        "It emphasizes exclusive access and limited membership but provides few details about fees or regulatory status.",
        "Interested readers are asked to contact an advisor before receiving the complete strategy.",
    ),
    (
        "unverified_advisor",
        "An independent advisor offers personalized portfolio recommendations through a direct messaging account.",
        "The advisor mentions strong historical performance but does not provide independently verified records.",
        "Recipients are encouraged to schedule a private conversation before deciding whether to participate.",
    ),
    (
        "promotional_platform",
        "A new online trading platform is promoting an introductory program for first-time investors.",
        "The message highlights potential returns and professional tools while providing limited information about company ownership.",
        "Users are invited to register for a demonstration and review the terms before funding an account.",
    ),
    (
        "limited_offer",
        "A market research service is offering early access to an investment report for selected subscribers.",
        "The invitation uses urgency and exclusivity, although it does not explicitly guarantee a return.",
        "Readers must request additional details to understand the risks, costs, and investment process.",
    ),
]

NORMAL_SCENARIOS = [
    (
        "regulated_fund",
        "The regulated fund has published its quarterly report, including holdings, fees, benchmark performance, and risk disclosures.",
        "Past performance is not a guarantee of future results, and investors may lose some or all of their principal.",
        "Please review the prospectus and consult a licensed financial professional before making a decision.",
    ),
    (
        "broker_notice",
        "Your brokerage has issued a routine notice about changes to trading fees and account documentation.",
        "The notice does not request a transfer or promise investment returns, and all changes can be verified in the official portal.",
        "Contact the customer service number shown on the regulated broker's website if you have questions.",
    ),
    (
        "education",
        "This educational seminar explains diversification, compound returns, inflation, and common investment risks.",
        "The material compares several asset classes without recommending a specific product or requesting money.",
        "Participants are encouraged to evaluate their goals and risk tolerance before investing.",
    ),
    (
        "portfolio_update",
        "Your financial adviser has prepared a scheduled portfolio review based on the objectives recorded in your account.",
        "The report explains recent market changes, current allocation, fees, and possible risks without guaranteeing performance.",
        "No action is required until you review the documents through the firm's verified client portal.",
    ),
]

DETAILS = [
    "The document includes several paragraphs describing the proposed process and expected timeline.",
    "The message uses professional financial language and presents the sender as an experienced market specialist.",
    "Recipients are encouraged to read the information carefully before responding.",
    "The proposal refers to current market volatility and recent changes in investor demand.",
    "The sender provides a detailed explanation intended to make the opportunity appear credible.",
]


def compose_examples(label, scenarios, count, rng):
    rows = []
    for index in range(count):
        scenario_name, *scenario_parts = scenarios[index % len(scenarios)]
        opener_index = (index // len(scenarios)) % len(OPENERS)
        parts = [OPENERS[opener_index], *scenario_parts, rng.choice(DETAILS)]
        rng.shuffle(parts)
        rows.append(
            {
                "text": " ".join(parts),
                "label": label,
                "source": "synthetic_compositional",
                "template_group": f"{label.lower()}_{scenario_name}_opener_{opener_index}",
            }
        )
    return rows


def compose_short_examples(label, scenarios, count, rng):
    rows = []
    for index in range(count):
        scenario_name, *scenario_parts = scenarios[index % len(scenarios)]
        opener_index = (index // len(scenarios)) % len(OPENERS)
        rows.append(
            {
                "text": f"{OPENERS[opener_index]} {rng.choice(scenario_parts)}",
                "label": label,
                "source": "synthetic_short",
                "template_group": f"{label.lower()}_{scenario_name}_opener_{opener_index}",
            }
        )
    return rows


def assign_dataset_splits(dataset):
    """Assign reproducible group-isolated 70/20/10 train/validation/test splits."""

    def best_group_split(frame, test_size, target_label_ratio, attempts=200):
        best = None
        for seed in range(SEED, SEED + attempts):
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_pos, holdout_pos = next(
                splitter.split(frame, frame["label"], groups=frame["template_group"])
            )
            holdout = frame.iloc[holdout_pos]
            holdout_ratio = holdout["label"].value_counts(normalize=True)
            label_error = sum(
                abs(holdout_ratio.get(label, 0) - ratio)
                for label, ratio in target_label_ratio.items()
            )
            size_error = abs((len(holdout) / len(frame)) - test_size)
            score = label_error + size_error
            if best is None or score < best[0]:
                best = (score, train_pos, holdout_pos)
        return best[1], best[2]

    dataset = dataset.copy()
    overall_ratio = dataset["label"].value_counts(normalize=True)
    train_validation_pos, test_pos = best_group_split(dataset, 0.10, overall_ratio)
    train_validation = dataset.iloc[train_validation_pos]

    train_pos, validation_pos = best_group_split(
        train_validation,
        2 / 9,
        overall_ratio,
    )

    dataset["split"] = ""
    dataset.iloc[test_pos, dataset.columns.get_loc("split")] = "test"
    dataset.loc[train_validation.iloc[train_pos].index, "split"] = "train"
    dataset.loc[train_validation.iloc[validation_pos].index, "split"] = "validation"
    return dataset


def build_dataset():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    rng = random.Random(SEED)
    raw = pd.read_csv(RAW_PATH)[["text", "label"]].dropna().drop_duplicates("text")
    raw = raw[~raw["text"].str.contains(TEMPLATE_LEAKAGE_TERMS, case=False, regex=True)]
    raw["source"] = "original_sms_dataset"
    raw["template_group"] = [f"original_{index}" for index in range(len(raw))]

    normal = raw[raw["label"] == "Normal"]
    investment_normal = normal[normal["text"].str.contains(INVESTMENT_TERMS, case=False, regex=True)]
    everyday_normal = normal.drop(investment_normal.index).sample(n=1500, random_state=SEED)

    suspicious = raw[raw["label"] == "Suspicious"]

    fraud = raw[raw["label"] == "Fraud"]
    fraud = fraud[fraud["text"].str.contains(INVESTMENT_TERMS, case=False, regex=True)]

    synthetic = pd.DataFrame(
        compose_examples("Fraud", FRAUD_SCENARIOS, 750, rng)
        + compose_examples("Suspicious", SUSPICIOUS_SCENARIOS, 650, rng)
        + compose_examples("Normal", NORMAL_SCENARIOS, 1600, rng)
        + compose_short_examples("Fraud", FRAUD_SCENARIOS, 450, rng)
        + compose_short_examples("Suspicious", SUSPICIOUS_SCENARIOS, 400, rng)
        + compose_short_examples("Normal", NORMAL_SCENARIOS, 400, rng)
    )

    dataset = pd.concat(
        [everyday_normal, investment_normal, suspicious, fraud, synthetic],
        ignore_index=True,
    )
    dataset = dataset.drop_duplicates("text").sample(frac=1, random_state=SEED).reset_index(drop=True)
    dataset = assign_dataset_splits(dataset)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    return dataset


if __name__ == "__main__":
    result = build_dataset()
    print(result.groupby(["label", "source"]).size())
    print("\nMean text length by label:")
    print(result.assign(length=result["text"].str.len()).groupby("label")["length"].mean())
