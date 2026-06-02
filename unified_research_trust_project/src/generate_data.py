import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, RANDOM_SEED


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def clipped_normal(rng: np.random.Generator, mean: float, sd: float, size: int) -> np.ndarray:
    return np.clip(rng.normal(mean, sd, size), 0, 100)


def generate_applications(n_samples: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    disciplines = np.array(["生命科学", "信息科学", "工程技术", "管理科学", "交叉学科"])
    discipline = rng.choice(disciplines, size=n_samples, p=[0.22, 0.24, 0.22, 0.14, 0.18])

    institution_tier = rng.choice(["普通高校/机构", "省部级重点单位", "双一流/国家级平台"], size=n_samples, p=[0.45, 0.35, 0.20])
    overseas_experience = rng.choice(["无", "短期交流", "长期经历"], size=n_samples, p=[0.58, 0.27, 0.15])
    age_group = rng.choice(["青年", "中青年", "资深"], size=n_samples, p=[0.38, 0.42, 0.20])

    research_years = np.where(
        age_group == "青年",
        rng.integers(1, 8, n_samples),
        np.where(age_group == "中青年", rng.integers(6, 18, n_samples), rng.integers(15, 35, n_samples)),
    )
    papers_5y = np.maximum(0, rng.poisson(5 + research_years * 0.35))
    h_index = np.maximum(1, np.round(rng.normal(4 + research_years * 0.7 + papers_5y * 0.15, 3))).astype(int)
    citations_5y = np.round(rng.lognormal(mean=np.log(30 + papers_5y * 18), sigma=0.65)).astype(int)
    prior_grants = np.maximum(0, rng.poisson(np.clip(research_years / 8, 0.1, 4.0)))

    novelty_score = clipped_normal(rng, 72, 13, n_samples)
    feasibility_score = clipped_normal(rng, 70, 14, n_samples)
    frontier_score = clipped_normal(rng, 68, 15, n_samples)
    methodology_score = clipped_normal(rng, 69, 13, n_samples)
    expected_impact = clipped_normal(rng, 66, 16, n_samples)
    budget_reasonableness = clipped_normal(rng, 74, 12, n_samples)
    data_availability = clipped_normal(rng, 70, 15, n_samples)
    ethics_compliance = clipped_normal(rng, 82, 10, n_samples)

    team_size = np.clip(rng.poisson(5, n_samples) + 1, 1, 14)
    interdisciplinarity = clipped_normal(rng, 56, 20, n_samples)
    collaboration_strength = clipped_normal(rng, 62, 18, n_samples)
    platform_support = clipped_normal(rng, 64, 17, n_samples)

    project_risk = clipped_normal(rng, 50, 20, n_samples)
    high_novelty = novelty_score > 80
    project_risk = np.where(high_novelty, np.clip(project_risk + rng.normal(8, 5, n_samples), 0, 100), project_risk)

    tier_effect = {
        "普通高校/机构": -0.10,
        "省部级重点单位": 0.04,
        "双一流/国家级平台": 0.16,
    }
    overseas_effect = {"无": -0.03, "短期交流": 0.03, "长期经历": 0.08}
    discipline_effect = {
        "生命科学": 0.05,
        "信息科学": 0.08,
        "工程技术": 0.02,
        "管理科学": -0.04,
        "交叉学科": 0.10,
    }

    # The latent rule intentionally mixes domain-reasonable signals, nonlinear terms, and noise.
    # This makes the model explainable while preserving uncertainty similar to real review settings.
    career_curve = -((research_years - 12) ** 2) / 220
    publication_signal = 0.33 * np.log1p(papers_5y) + 0.14 * np.log1p(citations_5y) + 0.06 * np.sqrt(h_index)
    quality_signal = (
        0.035 * novelty_score
        + 0.032 * feasibility_score
        + 0.025 * frontier_score
        + 0.026 * methodology_score
        + 0.028 * expected_impact
        + 0.020 * budget_reasonableness
        + 0.012 * data_availability
        + 0.010 * ethics_compliance
    )
    team_signal = (
        0.035 * np.minimum(team_size, 8)
        + 0.018 * interdisciplinarity
        + 0.014 * collaboration_strength
        + 0.013 * platform_support
    )
    interaction_signal = (
        0.00055 * (novelty_score - 60) * (feasibility_score - 60)
        + 0.00038 * (frontier_score - 55) * (interdisciplinarity - 50)
    )
    risk_penalty = -0.030 * project_risk - 0.010 * np.maximum(project_risk - 75, 0)
    background_signal = (
        pd.Series(institution_tier).map(tier_effect).to_numpy()
        + pd.Series(overseas_experience).map(overseas_effect).to_numpy()
        + pd.Series(discipline).map(discipline_effect).to_numpy()
    )

    latent_score = (
        -17.8
        + publication_signal
        + career_curve
        + 0.17 * prior_grants
        + quality_signal
        + team_signal
        + interaction_signal
        + risk_penalty
        + background_signal
        + rng.normal(0, 0.75, n_samples)
    )

    true_prob = sigmoid(latent_score)
    funded = rng.binomial(1, true_prob)

    df = pd.DataFrame(
        {
            "application_id": [f"APP-{i:04d}" for i in range(1, n_samples + 1)],
            "discipline": discipline,
            "papers_5y": papers_5y,
            "citations_5y": citations_5y,
            "h_index": h_index,
            "research_years": research_years,
            "prior_grants": prior_grants,
            "institution_tier": institution_tier,
            "overseas_experience": overseas_experience,
            "age_group": age_group,
            "novelty_score": np.round(novelty_score, 2),
            "feasibility_score": np.round(feasibility_score, 2),
            "frontier_score": np.round(frontier_score, 2),
            "methodology_score": np.round(methodology_score, 2),
            "expected_impact": np.round(expected_impact, 2),
            "budget_reasonableness": np.round(budget_reasonableness, 2),
            "project_risk": np.round(project_risk, 2),
            "data_availability": np.round(data_availability, 2),
            "ethics_compliance": np.round(ethics_compliance, 2),
            "team_size": team_size,
            "interdisciplinarity": np.round(interdisciplinarity, 2),
            "collaboration_strength": np.round(collaboration_strength, 2),
            "platform_support": np.round(platform_support, 2),
            "true_funding_probability": np.round(true_prob, 5),
            "funded": funded,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "applications.csv")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_applications(args.n_samples, args.seed)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Saved {len(df)} rows to {args.output}")
    print(f"Funding rate: {df['funded'].mean():.3f}")


if __name__ == "__main__":
    main()
