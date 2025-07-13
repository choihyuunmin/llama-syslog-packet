import os
import glob
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_metrics_from_csv(
    result_dir: str,
    output_path: str = "model_performance_metrics.png"
) -> None:
    """results 폴더 내 CSV 파일을 읽어 모델별 metric 평균을 계산하고, 그룹형 막대그래프를 저장합니다.

    Args:
        result_dir (str): CSV 파일들이 위치한 폴더 경로 (예: 'evaluation/result')
        output_path (str): 저장할 그래프 이미지 파일 경로

    Raises:
        FileNotFoundError: result_dir가 존재하지 않거나, CSV 파일이 없을 때
        ValueError: 필요한 metric 컬럼이 없을 때

    Example:
        plot_model_metrics_from_csv("evaluation/result", "output.png")
    """
    # 1. CSV 파일 수집
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"폴더가 존재하지 않습니다: {result_dir}")
    csv_files: List[str] = glob.glob(os.path.join(result_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"CSV 파일이 없습니다: {result_dir}")

    # 2. 데이터프레임 합치기
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    # 3. 사용할 metric 정의
    metric_map = {
        "ROUGE-1": "rouge1",
        "ROUGE-2": "rouge2",
        "ROUGE-L": "rougeL",
        "Cosine": "cosine_similarity",
        "F1": "f1_score",
    }
    required_columns = list(metric_map.values()) + ["model"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {col}")

    # 4. 모델별 metric 평균 계산
    avg_df = df.groupby("model")[list(metric_map.values())].mean().reset_index()

    # 5. 시각화를 위한 데이터 변환 (wide → long)
    plot_df = avg_df.melt(id_vars="model", var_name="Metric", value_name="Score")
    plot_df["Metric"] = plot_df["Metric"].map({v: k for k, v in metric_map.items()})

    # 6. 시각화
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="model",
        palette="deep"
    )
    plt.title("Model Performance Metrics Comparison (ROUGE, Cosine, F1)", pad=16)
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"그래프가 저장되었습니다: {output_path}")

if __name__ == "__main__":
    # 사용 예시
    plot_model_metrics_from_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "result"),
                               os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_performance_metrics.png")) 