import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import joblib
import pandas as pd
from sklearn.metrics import r2_score

# 한글 폰트 설정 (Windows 사용자용)
plt.rcParams['font.family'] = 'Malgun Gothic'

# 모델 불러오기
model = joblib.load("rice_model.pkl")

# 정확도 계산을 위해 데이터 불러오기
data = pd.read_csv("rice_data.csv")
X = data[['rainfall', 'temperature', 'humidity', 'ph']]
y = data['yield']
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# 앱 제목
st.title("🌾 쌀 생산량 예측기")

# 정확도 표시
st.subheader("📈 모델 정확도 (R² Score)")
st.success(f"현재 모델의 R² 점수: {r2:.2f}")

# 사용자 입력
st.subheader("📥 기후 정보 입력")
rainfall = st.slider("강수량 (mm)", 500, 1500, 1000)
temperature = st.slider("기온 (°C)", 15.0, 30.0, 22.0)
humidity = st.slider("습도 (%)", 50.0, 100.0, 75.0)

# pH 범위 5.4 ~ 6.2
ph_values = np.arange(5.4, 6.21, 0.1)
predicted_yields = []

for ph in ph_values:
    input_data = np.array([[rainfall, temperature, humidity, ph]])
    yield_pred = model.predict(input_data)[0]
    predicted_yields.append(yield_pred)

# 기준 pH=6.0 예측 결과 표시
default_pred = model.predict([[rainfall, temperature, humidity, 6.0]])[0]
st.metric("🌾 예측 쌀 생산량 (톤)", f"{default_pred:,.2f}")

# 그래프 출력
st.subheader("📊 pH 농도에 따른 생산량 예측")
fig, ax = plt.subplots()
ax.plot(ph_values, predicted_yields, color='green', linewidth=2)
ax.set_xlabel("pH 농도 (5.4 ~ 6.2)")
ax.set_ylabel("예측 쌀 생산량 (톤)")
ax.set_title("pH 농도에 따른 쌀 생산량 변화")
ax.grid(True)
st.pyplot(fig)
