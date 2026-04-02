# 웹 배포 가이드

이 저장소는 로컬 실행뿐 아니라 GitHub와 외부 웹 배포를 전제로 정리되어 있습니다.  
가장 빠른 GitHub 실행 경로는 `Codespaces`이며, 외부 공개 URL이 필요하면 `Render + Docker`를 권장합니다.

## 1. GitHub에서 바로 실행: Codespaces

저장소 루트의 [`.devcontainer/devcontainer.json`](../.devcontainer/devcontainer.json)을 사용합니다.

### 실행 순서

1. 저장소 메인 페이지에서 `Code` → `Codespaces` → `Create codespace on main`을 선택합니다.
2. Codespace가 열리면 `postCreateCommand`가 의존성 설치와 샘플 코퍼스 적재를 수행합니다.
3. `postAttachCommand`가 Streamlit 앱을 자동으로 실행합니다.
4. 포트 `8501` 미리보기를 열면 GitHub 안에서 바로 앱을 사용할 수 있습니다.

### 필요한 secret

- `GEMINI_API_KEY`
- 선택: `LAW_API_KEY`

중요: GitHub 저장소의 `Actions secrets`와 `Codespaces secrets`는 별개입니다.

## 2. 외부 공개 URL 권장 경로: Render

저장소 루트의 [`render.yaml`](../render.yaml)과 [`Dockerfile`](../Dockerfile)을 그대로 사용합니다.

### 배포 순서

1. Render 대시보드에서 `New +` → `Blueprint`를 선택합니다.
2. GitHub 저장소 `SeungMin-Park-psm1757/MVP_Military_basic_rule_finder`를 연결합니다.
3. `render.yaml`을 읽으면 `GEMINI_API_KEY` 입력을 요구합니다.
4. 값을 입력하면 서비스가 생성되고 공개 URL이 발급됩니다.

### 공개 URL

- 권장 서비스명 기준 공개 주소: `https://military-basic-rule-chatbot.onrender.com`
- 실제 주소는 최초 배포 시 Render가 확정합니다.

### 왜 Render를 권장하나

- `render.yaml`의 `sync: false`를 사용해 API 키를 코드에 넣지 않고 주입할 수 있습니다.
- Docker 기반이라 현재 Streamlit 앱 구조를 거의 그대로 올릴 수 있습니다.
- 무료 web service 인스턴스를 공식 지원합니다.

### 현재 저장소 기준 배포 방식

이 저장소의 [`render.yaml`](../render.yaml)은 현재 `plan: free` 기준으로 맞춰져 있습니다.

- Chroma 경로: `/tmp/chroma`
- runtime 경로: `/tmp/runtime`
- 첫 실행 시 샘플 코퍼스 자동 적재

### 무료 플랜 한계

- Render free web service는 15분 유휴 시 spin-down 됩니다.
- 다음 요청이 오면 다시 기동되며, 약 1분 정도 걸릴 수 있습니다.
- free web service는 영구 디스크를 지원하지 않습니다.
- 따라서 로컬 파일 기반 Chroma와 사용량 추적 파일은 재시작·재배포·spin-down 시 초기화될 수 있습니다.

## 3. 대안 경로: Streamlit Community Cloud

이 저장소는 Streamlit Community Cloud에도 올릴 수 있습니다. 다만 이 경우 로컬 파일 저장소가 영구적이지 않을 수 있어, 재배포나 재시작 시 샘플 코퍼스를 다시 적재할 수 있습니다.

### 설정 순서

1. Streamlit Community Cloud에서 이 GitHub 저장소를 선택합니다.
2. 메인 파일은 `streamlit_app.py`로 지정합니다.
3. App settings의 `Secrets`에 `GEMINI_API_KEY`를 등록합니다.
4. 배포 후 발급된 `*.streamlit.app` 주소를 README 상단 링크에 반영합니다.

## 4. 중요한 점: GitHub Secrets와 서비스 Secrets는 다르다

GitHub 저장소의 `Actions secrets`에 `GEMINI_API_KEY`를 넣어도, 그것만으로 실행 중인 앱 런타임에 자동 전달되지는 않습니다.

- GitHub Actions secret: CI나 배포 워크플로에서 사용
- GitHub Codespaces secret: GitHub Codespaces 런타임에서 사용
- Render secret / Streamlit Cloud secret: 실제 서비스 런타임에서 사용

즉, GitHub Codespaces로 실행하든 외부 공개 배포를 하든, 해당 런타임에 맞는 secret 저장소에 `GEMINI_API_KEY`를 따로 등록해야 합니다.
