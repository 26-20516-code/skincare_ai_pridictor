
import streamlit as st
import pickle
import Orange

st.set_page_config(page_title="SKIN AI", page_icon="🧴")

@st.cache_resource
def load_model():
    with open("skincare_interest_model.pkcls", "rb") as f:
        return pickle.load(f)

st.title("🧴 SKIN AI")
st.subheader("AI 화장품 소비자 관심도 예측")
st.write("신제품 정보를 입력하고 예상 소비자 관심도를 확인해 보세요.")

try:
    model = load_model()
    domain = model.original_domain

    with st.form("prediction"):
        st.header("제품 정보")

        price = st.number_input("제품 가격 ($)", min_value=0.0, value=30.0)
        rating = st.slider("예상 평점", 0.0, 5.0, 4.5, 0.1)
        reviews = st.number_input("예상 리뷰 수", min_value=0, value=100)

        categories = list(domain["secondary_category"].values)
        category = st.selectbox("화장품 종류", categories)

        submitted = st.form_submit_button("AI 예측하기")

    if submitted:
        inputs = {
            "price_usd": price,
            "rating": rating,
            "reviews": reviews,
            "primary_category": "Skincare",
            "secondary_category": category,
            "new": "0",
            "online_only": "0",
            "limited_edition": "0",
            "sephora_exclusivity": "0"
        }

        row = []
        for var in domain.attributes:
            value = inputs.get(var.name, 0)

            if var.is_discrete:
                value = var.to_val(str(value))

            row.append(value)

        data = Orange.data.Table.from_list(domain, [row])
        prediction = float(model(data)[0])

        st.divider()
        st.header("AI 예측 결과")
        st.metric("예상 소비자 관심도", f"{max(0, prediction):,.0f}개")
        st.caption("좋아요 수를 기반으로 예측한 소비자 관심도이며 실제 판매량은 아닙니다.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")

