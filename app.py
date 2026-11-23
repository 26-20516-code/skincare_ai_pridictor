
import streamlit as st
import pickle
import Orange
from Orange.data import Domain, Table

st.set_page_config(
    page_title="SKIN AI",
    page_icon="🧴",
    layout="centered"
)

@st.cache_resource
def load_model():
    with open("skincare_interest_model.pkcls", "rb") as f:
        return pickle.load(f)

st.title("🧴 SKIN AI")
st.subheader("AI 기반 화장품 소비자 관심도 예측")
st.write(
    "제품의 특성을 입력하면 AI가 예상 소비자 관심도를 분석합니다."
)

try:
    model = load_model()
    attributes = model.domain.attributes

    st.divider()
    st.header("제품 정보 입력")

    values = {}

    with st.form("prediction_form"):
        for var in attributes:
            name = var.name

            if var.is_discrete:
                options = list(var.values)
                selected = st.selectbox(name, options)
                values[name] = options.index(selected)

            elif var.is_continuous:
                if name == "price_usd":
                    values[name] = st.number_input(
                        "제품 가격 (달러)", min_value=0.0,
                        value=30.0, step=1.0
                    )
                elif name == "rating":
                    values[name] = st.number_input(
                        "평점", min_value=0.0, max_value=5.0,
                        value=4.0, step=0.1
                    )
                elif name == "reviews":
                    values[name] = st.number_input(
                        "리뷰 수", min_value=0,
                        value=100, step=10
                    )
                else:
                    values[name] = st.number_input(
                        name, value=0.0
                    )

        submitted = st.form_submit_button("AI 예측하기")

    if submitted:
        row = [values[var.name] for var in attributes]
        data = Table.from_list(Domain(attributes), [row])
        prediction = float(model(data)[0])

        st.divider()
        st.header("AI 예측 결과")
        st.metric(
            "예상 소비자 관심도",
            f"{max(0, prediction):,.0f}개"
        )
        st.caption(
            "※ 실제 판매량이나 구매량이 아닌, "
            "좋아요 수를 기반으로 예측한 관심도입니다."
        )

except Exception as e:
    st.error(f"모델을 불러오거나 예측하는 중 오류가 발생했습니다: {e}")
    st.info("모델 파일과 라이브러리 설정을 확인해 주세요.")
