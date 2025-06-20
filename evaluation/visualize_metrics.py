import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# 현재 스크립트의 디렉토리 경로 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(current_dir, 'result')

# Seaborn 스타일 설정
plt.style.use('seaborn')
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# CSV 파일들을 읽어서 데이터프레임으로 변환
csv_files = glob.glob(os.path.join(result_dir, '*.csv'))
dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# 모든 데이터프레임을 하나로 합치기
combined_df = pd.concat(dfs, ignore_index=True)

# pass@1을 제외한 메트릭들의 평균 계산
metrics = ['f1_score', 'rouge1', 'rouge2', 'rougeL', 'cosine_similarity', 'numeric_mae', 'numeric_mse']
avg_metrics = combined_df.groupby('model')[metrics].mean()

# numeric_mae와 numeric_mse는 다른 메트릭들과 스케일이 다르므로 별도로 처리
numeric_metrics = ['numeric_mae', 'numeric_mse']
other_metrics = [m for m in metrics if m not in numeric_metrics]

# 다른 메트릭들의 평균 계산
avg_other_metrics = avg_metrics[other_metrics].mean(axis=1)

# 시각화 - 일반 메트릭
plt.figure(figsize=(10, 6))
ax = avg_other_metrics.plot(kind='bar', color=sns.color_palette("husl", len(avg_other_metrics)))
plt.title('Average Performance Metrics by Model', pad=20)
plt.xlabel('Model', labelpad=10)
plt.ylabel('Average Score', labelpad=10)
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)

# 값 레이블 추가
for i, v in enumerate(avg_other_metrics):
    ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'average_metrics.png'), dpi=300, bbox_inches='tight')

# numeric_mae와 numeric_mse 시각화
plt.figure(figsize=(10, 6))
ax = avg_metrics[numeric_metrics].plot(kind='bar', color=sns.color_palette("Set2", len(numeric_metrics)))
plt.title('Numeric Error Metrics by Model', pad=20)
plt.xlabel('Model', labelpad=10)
plt.ylabel('Error Score', labelpad=10)
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)

# 값 레이블 추가
for i, v in enumerate(avg_metrics[numeric_metrics].iloc[:, 0]):
    ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')
for i, v in enumerate(avg_metrics[numeric_metrics].iloc[:, 1]):
    ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')

plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'numeric_metrics.png'), dpi=300, bbox_inches='tight') 