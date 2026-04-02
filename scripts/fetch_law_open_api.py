from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

from _bootstrap import ensure_project_src_on_path

ensure_project_src_on_path()

from army_reg_rag.utils.io import ensure_dir


SEARCH_ENDPOINT = "https://www.law.go.kr/DRF/lawSearch.do"
SERVICE_ENDPOINT = "https://www.law.go.kr/DRF/lawService.do"


def request_json(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return {"raw_text": response.text}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="law.go.kr Open API에서 공개 법령 자료를 베스트에포트 방식으로 수집합니다. MVP 필수 단계는 아니며, 전량 연혁 본문 구조화가 필요할 때 쓰는 선택 기능입니다."
    )
    parser.add_argument("--law", required=True, help="예: 군인의 지위 및 복무에 관한 기본법")
    parser.add_argument("--output-dir", default="data/raw/api", help="저장 디렉터리")
    args = parser.parse_args()

    load_dotenv()
    oc = os.getenv("LAW_API_KEY", "").strip()
    if not oc:
        raise SystemExit("LAW_API_KEY가 필요합니다. .env에 설정해 주세요. 다만 현재 MVP는 public page 수집만으로도 유지되도록 설계되어 있습니다.")

    output_dir = ensure_dir(args.output_dir)
    slug = args.law.replace(" ", "_")

    # 1) 기본 검색
    search_params = {"OC": oc, "target": "law", "type": "JSON", "query": args.law}
    search_data = request_json(SEARCH_ENDPOINT, search_params)
    (output_dir / f"{slug}__search.json").write_text(
        json.dumps(search_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    law_id = None
    mst = None
    if isinstance(search_data, dict):
        text = json.dumps(search_data, ensure_ascii=False)
        # 최소한의 베스트에포트 추출
        for key in ("법령ID", "ID", "id"):
            if key in search_data:
                law_id = search_data[key]
                break
        if '"MST"' in text or '"mst"' in text:
            try:
                payload = search_data
                # list 구조가 다양할 수 있으므로 재귀 탐색
                def _find_first(obj, names):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if k in names and v:
                                return v
                            found = _find_first(v, names)
                            if found:
                                return found
                    elif isinstance(obj, list):
                        for item in obj:
                            found = _find_first(item, names)
                            if found:
                                return found
                    return None
                mst = _find_first(payload, {"MST", "mst"})
                law_id = law_id or _find_first(payload, {"법령ID", "ID", "id"})
            except Exception:
                pass

    # 2) 현행 본문 (법령ID가 있으면 target=law, 없으면 target=eflaw + mst 시도)
    if law_id:
        current_params = {"OC": oc, "target": "law", "type": "JSON", "ID": law_id}
        current_data = request_json(SERVICE_ENDPOINT, current_params)
        (output_dir / f"{slug}__current.json").write_text(
            json.dumps(current_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("saved current law body via target=law")
    elif mst:
        current_params = {"OC": oc, "target": "eflaw", "type": "JSON", "MST": mst}
        current_data = request_json(SERVICE_ENDPOINT, current_params)
        (output_dir / f"{slug}__current_eflaw.json").write_text(
            json.dumps(current_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("saved current law body via target=eflaw")
    else:
        print("warning: could not infer law_id or MST from search result")

    # 3) 연혁
    if mst:
        history_params = {"OC": oc, "target": "lsHistory", "type": "JSON", "MST": mst}
        history_data = request_json(SERVICE_ENDPOINT, history_params)
        (output_dir / f"{slug}__history.json").write_text(
            json.dumps(history_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("saved history via target=lsHistory")
    else:
        print("warning: MST not found; history fetch skipped")

    # 4) 신구비교: 운영 환경에 따라 ID/MST 조합 요구사항이 다를 수 있어 베스트에포트로 보관
    if law_id:
        compare_params = {"OC": oc, "target": "oldAndNew", "type": "JSON", "ID": law_id}
        compare_data = request_json(SERVICE_ENDPOINT, compare_params)
        (output_dir / f"{slug}__old_and_new.json").write_text(
            json.dumps(compare_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("saved best-effort old/new comparison via target=oldAndNew")
    else:
        print("warning: law_id not found; old/new comparison fetch skipped")

    print(f"done: {args.law}")


if __name__ == "__main__":
    main()
