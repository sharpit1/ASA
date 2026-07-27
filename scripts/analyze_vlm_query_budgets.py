#!/usr/bin/env python3
"""Compute naturalness-qualified query-budget ASR for CLIP and SigLIP2 runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


BUDGETS = (20, 40, 60, 80, 100)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_index_from_report(path: Path) -> int:
    match = re.fullmatch(r"sample_(\d+)", path.parent.name)
    if match is None:
        raise ValueError(f"unexpected report parent: {path.parent}")
    return int(match.group(1))


def collect_reports(run_root: Path) -> dict[int, dict[str, Any]]:
    reports: dict[int, dict[str, Any]] = {}
    for path in sorted(run_root.glob("sample_*/report.json")):
        idx = sample_index_from_report(path)
        reports[idx] = load_json(path)
    return reports


def read_siglip2_clean_indices(run_root: Path) -> set[int]:
    path = run_root / "naturalness_verification" / "clean_filter_results.jsonl"
    rows = load_jsonl(path)
    selected = {
        int(row["sample_index"])
        for row in rows
        if row.get("status") == "selected" and row.get("clean_correct") is True
    }
    if len(rows) != 1000 or len(selected) != 859:
        raise ValueError(
            "unexpected SigLIP2 clean-filter counts: "
            f"examined={len(rows)} selected={len(selected)}"
        )
    return selected


def read_siglip2_naturalness(run_root: Path) -> dict[int, bool]:
    path = run_root / "naturalness_verification" / "results.jsonl"
    rows = load_jsonl(path)
    verdicts = {
        int(row["sample_index"]): bool(row["natural"])
        for row in rows
        if row.get("status") == "ok" and isinstance(row.get("natural"), bool)
    }
    if len(verdicts) != 615:
        raise ValueError(f"unexpected SigLIP2 naturalness verdict count: {len(verdicts)}")
    return verdicts


def clip_clean_indices(reports: dict[int, dict[str, Any]]) -> set[int]:
    # This run was clean-filtered from index 0 and contains only clean-correct reports.
    selected = set(reports)
    if len(selected) != 778:
        raise ValueError(f"unexpected CLIP clean-correct report count: {len(selected)}")
    return selected


def effective_records(
    *,
    victim: str,
    reports: dict[int, dict[str, Any]],
    clean_indices: set[int],
    siglip2_naturalness: dict[int, bool] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx in sorted(clean_indices):
        if idx not in reports:
            raise ValueError(f"missing {victim} report for clean-correct sample {idx}")
        report = reports[idx]
        final_success = report.get("final_attack_success") is True
        raw_query = int(report.get("victim_query_count", 100))
        if raw_query < 0 or raw_query > 100:
            raise ValueError(f"invalid query count for {victim} sample {idx}: {raw_query}")

        if siglip2_naturalness is None:
            # CLIP performs naturalness verification in-loop. A classifier success
            # rejected by that verifier remains visible in the history even when
            # final_attack_success is false.
            if report.get("attack_success_requires_naturalness") is not True:
                raise ValueError(f"CLIP sample {idx} did not require naturalness")
            if report.get("naturalness_verifier", {}).get("enabled") is not True:
                raise ValueError(f"CLIP sample {idx} verifier is disabled")
            checked_candidates = [
                candidate
                for history_row in report.get("history", [])
                for candidate in history_row.get("scored_candidates", [])
                if candidate.get("naturalness_checked") is True
            ]
            raw_success = bool(final_success or checked_candidates)
            natural = final_success
        else:
            raw_success = final_success
            natural = siglip2_naturalness.get(idx) is True if raw_success else False
            if raw_success and idx not in siglip2_naturalness:
                raise ValueError(f"missing SigLIP2 naturalness verdict for success {idx}")

        effective_success = bool(final_success and natural)
        effective_query = raw_query if effective_success else 100
        records.append(
            {
                "victim": victim,
                "sample_index": idx,
                "raw_success": raw_success,
                "natural": bool(natural),
                "effective_success": effective_success,
                "raw_query": raw_query,
                "effective_query": effective_query,
            }
        )
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    attacked = len(records)
    successes = [row for row in records if row["effective_success"]]
    failures = [row for row in records if not row["effective_success"]]
    raw_successes = [row for row in records if row["raw_success"]]
    natural_false = [
        row for row in records if row["raw_success"] and not row["natural"]
    ]
    budget_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        count = sum(
            row["effective_success"] and row["effective_query"] <= budget
            for row in records
        )
        budget_rows.append(
            {
                "budget": budget,
                "success_count": count,
                "attacked": attacked,
                "asr_percent": 100.0 * count / attacked,
            }
        )
    success_queries = [int(row["effective_query"]) for row in successes]
    all_queries = [int(row["effective_query"]) for row in records]
    failure_queries = [int(row["effective_query"]) for row in failures]
    return {
        "attacked": attacked,
        "raw_success_count": len(raw_successes),
        "natural_false_count": len(natural_false),
        "effective_success_count": len(successes),
        "effective_failure_count": len(failures),
        "budget_rows": budget_rows,
        "success_query_mean": mean(success_queries) if success_queries else None,
        "success_query_median": median(success_queries) if success_queries else None,
        "all_capped_query_mean": mean(all_queries),
        "all_capped_query_median": median(all_queries),
        "failure_query_mean": mean(failure_queries) if failure_queries else None,
    }


def parse_other_attacks(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    section = text.split("## 공격 결과", 1)[1].split("## 전체 1,000장 재평가", 1)[0]
    rows: list[dict[str, Any]] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 12 or cells[0] not in {
            "AdvFlow",
            "DIFAttack",
            "MCGAttack",
            "CGAttack",
        }:
            continue
        success_text, attacked_text = (part.strip() for part in cells[5].split("/"))
        rows.append(
            {
                "attack": cells[0],
                "victim": cells[1],
                "success_count": int(success_text),
                "attacked": int(attacked_text),
                "asr_percent": float(cells[6].rstrip("%")),
                "all_capped_query_mean": float(cells[7]),
                "success_query_mean": float(cells[8]),
            }
        )
    if len(rows) != 8:
        raise ValueError(f"expected 8 comparison rows, found {len(rows)}")
    return rows


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def format_budget_table(
    summaries: dict[str, dict[str, Any]],
) -> list[str]:
    lines = [
        "| Victim | Clean-correct | Q20 | Q40 | Q60 | Q80 | Q100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for victim in ("SigLIP2", "CLIP ViT-B/16"):
        summary = summaries[victim]
        cells = []
        for row in summary["budget_rows"]:
            cells.append(f'{row["success_count"]} ({row["asr_percent"]:.2f}%)')
        lines.append(
            f'| {victim} | {summary["attacked"]} | ' + " | ".join(cells) + " |"
        )
    return lines


def write_markdown(
    path: Path,
    summaries: dict[str, dict[str, Any]],
    other_attacks: list[dict[str, Any]],
) -> None:
    lines = [
        "# SigLIP2·CLIP query-budget ASR 분석",
        "",
        "## 산출 기준",
        "",
        "- 분모는 victim별 clean-correct 샘플(SigLIP2 859개, CLIP 778개)이다.",
        "- 성공 query는 각 `report.json`의 실제 종료 시점 `victim_query_count`를 사용한다.",
        "- classifier 공격 성공이어도 naturalness가 false이면 실패로 처리하고 유효 query를 100으로 둔다.",
        "- 나머지 실패도 유효 query를 100으로 둔다.",
        "- CLIP은 in-loop naturalness 결과, SigLIP2는 사후 Gemma 검증 결과를 사용한다.",
        "",
        "## Query 제한별 ASR",
        "",
        *format_budget_table(summaries),
        "",
        "## Query 통계",
        "",
        "| Victim | Classifier success 관측 | Naturalness로 최종 거절 | 최종 성공 | 최종 실패 | 성공만 Avg Q | 전체 capped Avg Q | 실패 Avg Q |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for victim in ("SigLIP2", "CLIP ViT-B/16"):
        item = summaries[victim]
        lines.append(
            f'| {victim} | {item["raw_success_count"]} | '
            f'{item["natural_false_count"]} | {item["effective_success_count"]} | '
            f'{item["effective_failure_count"]} | {item["success_query_mean"]:.2f} | '
            f'{item["all_capped_query_mean"]:.2f} | {item["failure_query_mean"]:.2f} |'
        )

    lines.extend(
        [
            "",
            "## Query 100에서 다른 공격과 비교",
            "",
            "| Victim | Attack | 성공 / Attacked | ASR | 전체 capped Avg Q | 성공만 Avg Q | ASA 대비 ASR 차이 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for victim in ("SigLIP2", "CLIP ViT-B/16"):
        asa = summaries[victim]
        asa_asr = 100.0 * asa["effective_success_count"] / asa["attacked"]
        combined = [
            {
                "attack": "ASA (본 결과)",
                "victim": victim,
                "success_count": asa["effective_success_count"],
                "attacked": asa["attacked"],
                "asr_percent": asa_asr,
                "all_capped_query_mean": asa["all_capped_query_mean"],
                "success_query_mean": asa["success_query_mean"],
            }
        ]
        combined.extend(row for row in other_attacks if row["victim"] == victim)
        for row in combined:
            delta = float(row["asr_percent"]) - asa_asr
            lines.append(
                f'| {victim} | {row["attack"]} | {row["success_count"]} / '
                f'{row["attacked"]} | {row["asr_percent"]:.2f}% | '
                f'{row["all_capped_query_mean"]:.2f} | '
                f'{row["success_query_mean"]:.2f} | {delta:+.2f}%p |'
            )
    lines.extend(
        [
            "",
            "## 비교 해석상 제한",
            "",
            "- 다른 네 공격은 문서상 classifier 성공 기준이며 naturalness verifier가 적용되지 않았다.",
            "- 다른 네 공격은 `epsilon=12/255` 픽셀 제약 공격이고 ASA는 생성·프롬프트 기반 공격이므로 ASR만으로 직접적인 우열을 확정할 수 없다.",
            "- SigLIP2 naturalness는 저장된 224×224 `attack_success.png`에 대한 사후 검증이다. 실행 중 full-resolution 후보를 검증한 CLIP과 입력이 완전히 같지는 않다.",
            "- 제공 문서에는 다른 공격의 개별 `progress.csv`가 없어 Q20/Q40/Q60/Q80 ASR은 재계산할 수 없다. 따라서 다른 공격 비교는 Q100에 한정한다.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--siglip2-root", type=Path, required=True)
    parser.add_argument("--other-summary-md", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    clip_reports = collect_reports(args.clip_root)
    siglip2_reports = collect_reports(args.siglip2_root)
    siglip2_clean = read_siglip2_clean_indices(args.siglip2_root)
    siglip2_naturalness = read_siglip2_naturalness(args.siglip2_root)

    clip_records = effective_records(
        victim="CLIP ViT-B/16",
        reports=clip_reports,
        clean_indices=clip_clean_indices(clip_reports),
    )
    siglip2_records = effective_records(
        victim="SigLIP2",
        reports=siglip2_reports,
        clean_indices=siglip2_clean,
        siglip2_naturalness=siglip2_naturalness,
    )
    summaries = {
        "SigLIP2": summarize_records(siglip2_records),
        "CLIP ViT-B/16": summarize_records(clip_records),
    }
    other_attacks = parse_other_attacks(args.other_summary_md)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_records_csv(args.output_dir / "siglip2_effective_queries.csv", siglip2_records)
    write_records_csv(args.output_dir / "clip_effective_queries.csv", clip_records)
    with (args.output_dir / "query_budget_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {"summaries": summaries, "other_attacks": other_attacks},
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    write_markdown(
        args.output_dir / "query_budget_comparison.md",
        summaries,
        other_attacks,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
