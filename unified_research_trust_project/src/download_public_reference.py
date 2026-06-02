import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from config import DATA_DIR, OUTPUT_DIR


OPENALEX_WORKS_API = "https://api.openalex.org/works"


def simplify_work(work: dict) -> dict:
    authorships = work.get("authorships") or []
    institutions = []
    countries = set()
    for authorship in authorships:
        for inst in authorship.get("institutions") or []:
            institutions.append(inst.get("display_name"))
            if inst.get("country_code"):
                countries.add(inst["country_code"])

    primary_topic = work.get("primary_topic") or {}
    topic_field = (primary_topic.get("field") or {}).get("display_name")
    topic_domain = (primary_topic.get("domain") or {}).get("display_name")
    topic_name = primary_topic.get("display_name")

    open_access = work.get("open_access") or {}
    citation_percentile = work.get("citation_normalized_percentile") or {}
    awards = work.get("awards") or []
    funders = work.get("funders") or []

    return {
        "openalex_id": work.get("id"),
        "title": work.get("display_name"),
        "publication_year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count"),
        "fwci": work.get("fwci"),
        "citation_percentile": citation_percentile.get("value"),
        "is_oa": open_access.get("is_oa"),
        "authorship_count": len(authorships),
        "institution_count": len([x for x in institutions if x]),
        "country_count": len(countries),
        "referenced_works_count": work.get("referenced_works_count"),
        "related_works_count": len(work.get("related_works") or []),
        "primary_topic": topic_name,
        "topic_field": topic_field,
        "topic_domain": topic_domain,
        "award_count": len(awards),
        "funder_count": len(funders),
        "funders": "; ".join(
            sorted({funder.get("display_name") for funder in funders if funder.get("display_name")})
        ),
        "doi": work.get("doi"),
    }


def fetch_openalex_reference(per_page: int = 80, mailto: str | None = None) -> list[dict]:
    params = {
        "search": "artificial intelligence scientific discovery research funding",
        "filter": "from_publication_date:2021-01-01,has_abstract:true",
        "per-page": per_page,
        "select": ",".join(
            [
                "id",
                "doi",
                "display_name",
                "publication_year",
                "cited_by_count",
                "fwci",
                "citation_normalized_percentile",
                "open_access",
                "authorships",
                "referenced_works_count",
                "related_works",
                "primary_topic",
                "awards",
                "funders",
            ]
        ),
    }
    if mailto:
        params["mailto"] = mailto

    response = requests.get(OPENALEX_WORKS_API, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("results", [])


def build_feature_mapping() -> pd.DataFrame:
    rows = [
        {
            "public_data_source": "OpenAlex Works",
            "public_field": "cited_by_count, fwci, citation_normalized_percentile",
            "simulated_feature": "citations_5y, expected_impact",
            "mapping_logic": "引用量、领域加权引用影响和引用分位数可作为科研影响力的现实代理变量。",
        },
        {
            "public_data_source": "OpenAlex Works",
            "public_field": "authorships, institution_count, country_count",
            "simulated_feature": "team_size, collaboration_strength",
            "mapping_logic": "作者数量、机构数量和国家数量反映团队规模与合作网络强度。",
        },
        {
            "public_data_source": "OpenAlex Works",
            "public_field": "primary_topic, topic_field, topic_domain",
            "simulated_feature": "discipline, frontier_score, interdisciplinarity",
            "mapping_logic": "主题、领域和跨领域主题可支撑学科类别、前沿度和跨学科程度的构造。",
        },
        {
            "public_data_source": "OpenAlex Works",
            "public_field": "referenced_works_count, related_works_count",
            "simulated_feature": "methodology_score, feasibility_score",
            "mapping_logic": "参考文献和相关工作数量可间接反映研究基础、方法依据和可行性支撑。",
        },
        {
            "public_data_source": "OpenAlex Works",
            "public_field": "open_access.is_oa, has_abstract filter",
            "simulated_feature": "data_availability",
            "mapping_logic": "开放获取和摘要可得性反映科研成果或项目材料的信息可获得程度。",
        },
        {
            "public_data_source": "OpenAlex Works",
            "public_field": "awards, funders",
            "simulated_feature": "funded, prior_grants",
            "mapping_logic": "资助奖项和资助方信息可作为已获资助背景的现实参照，但公开数据缺少未获资助申请，不能直接替代监督学习标签。",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-page", type=int, default=80)
    parser.add_argument("--mailto", type=str, default=None)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    works = fetch_openalex_reference(args.per_page, args.mailto)
    simplified = [simplify_work(work) for work in works]
    sample_df = pd.DataFrame(simplified)
    mapping_df = build_feature_mapping()

    raw_path = DATA_DIR / "openalex_reference_raw.json"
    sample_path = DATA_DIR / "openalex_reference_sample.csv"
    mapping_path = OUTPUT_DIR / "public_feature_mapping.csv"
    summary_path = OUTPUT_DIR / "public_data_summary.json"

    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(works, f, ensure_ascii=False, indent=2)
    sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")
    mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")

    summary = {
        "source": "OpenAlex Works API",
        "api_url": OPENALEX_WORKS_API,
        "rows_downloaded": int(len(sample_df)),
        "time_scope": "from 2021-01-01",
        "query": "artificial intelligence scientific discovery research funding",
        "columns": sample_df.columns.tolist(),
        "mapping_rows": int(len(mapping_df)),
        "important_limitation": (
            "OpenAlex provides real scholarly metadata and funding references, but it does not "
            "provide rejected grant applications, expert scores, or final review decisions. "
            "Therefore it is used as feature-design evidence rather than as the main supervised "
            "training dataset."
        ),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
