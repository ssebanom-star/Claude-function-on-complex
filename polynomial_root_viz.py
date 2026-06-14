"""
복소평면 위 n차 다항식의 절댓값 곡면 시각화

아이디어
--------
n차 방정식 f(x) = 0 의 해는 일반적으로 복소수이다.
실수축 하나로는 복소근을 그릴 수 없으므로, 정의역을 복소평면 전체로 확장한다.

    - x축, y축 : 복소수 x = a + bi 의 실수부(a)와 허수부(b)
    - z축      : 함숫값의 절댓값 |f(x)|

그러면 f(x) = 0 이 되는 지점에서 |f(x)| = 0 이므로,
곡면이 바닥(z = 0)에 닿아 "골짜기 / 구멍"처럼 보인다.
바로 그 지점이 n차 방정식의 해(근)이다.

사용 예
------
    # 직접 계수를 넣는 경우 (x^2 + 1, 근: ±i)
    python polynomial_root_viz.py --coeffs 1 0 1

    # 대화형으로 입력
    python polynomial_root_viz.py

    # 그림 파일로 저장 (화면 없는 환경)
    python polynomial_root_viz.py --coeffs 1 0 0 -1 --save roots.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


# ---------------------------------------------------------------------------
# 다항식 정의 / 계산
# ---------------------------------------------------------------------------
def parse_coeffs(tokens: list[str]) -> np.ndarray:
    """문자열 토큰을 복소수 계수 배열로 변환한다.

    계수는 최고차항부터 상수항 순서로 입력한다.
    예) x^3 - 1  ->  [1, 0, 0, -1]
    허수는 'j' 표기를 쓴다.  예) (1+2j)
    """
    coeffs = []
    for t in tokens:
        t = t.strip().replace("i", "j")  # 수학에서 쓰는 i 도 허용
        try:
            coeffs.append(complex(t))
        except ValueError as exc:
            raise ValueError(f"계수를 해석할 수 없습니다: {t!r}") from exc
    if len(coeffs) < 2:
        raise ValueError("최소 1차 이상(계수 2개 이상)을 입력해야 합니다.")
    return np.array(coeffs, dtype=complex)


def poly_to_string(coeffs: np.ndarray) -> str:
    """계수 배열을 사람이 읽기 좋은 다항식 문자열로 만든다."""
    n = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = n - i
        if c == 0:
            continue
        # 계수 표기 (실수면 깔끔하게)
        if c.imag == 0:
            cval = c.real
            cstr = f"{cval:g}"
        else:
            cstr = f"({c:g})"

        if power == 0:
            terms.append(cstr)
        elif power == 1:
            terms.append(f"{cstr}*x")
        else:
            terms.append(f"{cstr}*x^{power}")
    return " + ".join(terms) if terms else "0"


def evaluate(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """복소 격자 x 위에서 다항식 값을 벡터화 계산한다 (Horner 법)."""
    result = np.zeros_like(x, dtype=complex)
    for c in coeffs:
        result = result * x + c
    return result


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def setup_korean_font() -> bool:
    """한글을 표시할 수 있는 폰트를 찾아 matplotlib 기본 폰트로 설정한다.

    찾으면 True, 없으면 (영어 라벨을 써야 하면) False 를 반환한다.
    """
    import matplotlib
    from matplotlib import font_manager

    candidates = [
        "Malgun Gothic",       # Windows
        "AppleGothic",         # macOS
        "NanumGothic",         # 나눔고딕 (리눅스에 흔함)
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "UnDotum",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def visualize(
    coeffs: np.ndarray,
    span: float | None = None,
    resolution: int = 300,
    log_scale: bool = False,
    save: str | None = None,
) -> None:
    import matplotlib

    if save is not None:
        matplotlib.use("Agg")  # 화면 없는 환경용 백엔드

    import matplotlib.pyplot as plt
    from matplotlib import cm

    # 한글 폰트가 없으면 라벨을 영어로 자동 전환 (글자 깨짐 방지)
    kr = setup_korean_font()
    L = {
        "re": "Re(x)  (실수부)" if kr else "Re(x)",
        "im": "Im(x)  (허수부)" if kr else "Im(x)",
        "z": ("log(1+|f(x)|)" if log_scale else "|f(x)|"),
        "roots": "빨간 점 = 방정식의 해 (|f(x)| = 0 인 곳)"
        if kr
        else "Red dots = roots of the equation (where |f(x)| = 0)",
    }

    # 1) 근 계산 (numpy 가 동반행렬 고유값으로 구해 준다)
    roots = np.roots(coeffs)

    # 2) 보기 범위 자동 결정: 근들이 충분히 보이도록 여유를 둔다
    if span is None:
        if len(roots) and np.max(np.abs(roots)) > 0:
            span = float(np.max(np.abs(roots))) * 1.6 + 1.0
        else:
            span = 3.0

    # 3) 복소 격자 생성
    re = np.linspace(-span, span, resolution)
    im = np.linspace(-span, span, resolution)
    RE, IM = np.meshgrid(re, im)
    X = RE + 1j * IM

    # 4) 절댓값 곡면
    Z = np.abs(evaluate(coeffs, X))
    Z_plot = np.log1p(Z) if log_scale else Z  # 근처 골짜기를 강조하려면 log

    # 5) 그리기
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        RE,
        IM,
        Z_plot,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
        rstride=2,
        cstride=2,
    )

    # 6) 근을 바닥(z=0)에 표시 + 곡면 골짜기까지 잇는 수직선
    z_min = float(np.min(Z_plot))
    for r in roots:
        ax.scatter(
            r.real,
            r.imag,
            z_min,
            color="red",
            s=80,
            marker="o",
            depthshade=False,
            edgecolors="black",
            zorder=10,
        )
        ax.plot(
            [r.real, r.real],
            [r.imag, r.imag],
            [z_min, np.max(Z_plot)],
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )

    ax.set_xlabel(L["re"])
    ax.set_ylabel(L["im"])
    ax.set_zlabel(L["z"])
    ax.set_title(f"f(x) = {poly_to_string(coeffs)}\n{L['roots']}")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.1, label="|f(x)|")

    # 7) 근 목록 출력
    print(f"\nf(x) = {poly_to_string(coeffs)}")
    print(f"차수: {len(coeffs) - 1}")
    print(f"근 {len(roots)}개:")
    for i, r in enumerate(sorted(roots, key=lambda z: (z.real, z.imag)), 1):
        # 거의 실수인 근은 깔끔하게 표기
        if abs(r.imag) < 1e-9:
            print(f"  x{i} = {r.real:.6g}")
        else:
            print(f"  x{i} = {r.real:.6g} {'+' if r.imag >= 0 else '-'} {abs(r.imag):.6g}i")

    if save is not None:
        fig.savefig(save, dpi=130, bbox_inches="tight")
        print(f"\n그림을 저장했습니다: {save}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 입력 처리
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="복소평면 위 n차 다항식의 |f(x)| 곡면으로 방정식의 해를 시각화합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--coeffs",
        "-c",
        nargs="+",
        help="최고차항부터 상수항 순서의 계수. 예) x^2+1 은 '1 0 1'",
    )
    parser.add_argument("--span", "-s", type=float, default=None, help="보기 범위 (±span). 미지정 시 자동")
    parser.add_argument("--resolution", "-r", type=int, default=300, help="격자 해상도 (기본 300)")
    parser.add_argument("--log", action="store_true", help="z축을 log(1+|f|)로 (골짜기 강조)")
    parser.add_argument("--save", default=None, help="화면 대신 그림 파일로 저장 (예: out.png)")
    args = parser.parse_args(argv)

    if args.coeffs is None:
        print("n차 다항식의 계수를 최고차항부터 공백으로 구분해 입력하세요.")
        print("예) x^3 - 1  ->  1 0 0 -1")
        print("    x^2 + 1  ->  1 0 1   (근: ±i)")
        raw = input("계수: ").strip()
        if not raw:
            print("입력이 없습니다.", file=sys.stderr)
            return 1
        tokens = raw.split()
    else:
        tokens = args.coeffs

    try:
        coeffs = parse_coeffs(tokens)
    except ValueError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    visualize(
        coeffs,
        span=args.span,
        resolution=args.resolution,
        log_scale=args.log,
        save=args.save,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
