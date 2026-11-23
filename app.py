
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
st.write("간단한 제품 정보로 소비자 관심도를 예측합니다.")

try:
    model = load_model()
    attributes = model.domain.attributes

    variables = {v.name: v for v in attributes}

    with st.form("prediction"):
        st.header("제품 정보")

        price = st.number_input(
            "제품 가격 ($)", min_value=0.0, value=30.0
        )
        rating = st.slider(
            "예상 평점", 0.0, 5.0, 4.5, 0.1
        )
        reviews = st.number_input(
            "예상 리뷰 수", min_value=0, value=100
        )

        category_var = variables.get("secondary_category")

        if category_var and category_var.is_discrete:
            categories = list(category_var.values)
            category = st.selectbox("화장품 종류", categories)
        else:
            categories = []
            category = None

        submitted = st.form_submit_button("AI 예측하기")

    if submitted:
        inputs = {
            "price_usd": price,
            "rating": rating,
            "reviews": reviews,
            "primary_category": "Skincare",
            "secondary_category": category,
        }

        row = []

        for var in attributes:
            value = inputs.get(var.name)

            if var.is_discrete:
                options = list(var.values)

                if value is None or str(value) not in options:
                    value = next(
                        (x for x in options
                         if str(x).lower() in ["0", "false", "no"]),
                        options[0]
                    )

                row.append(float(options.index(str(value))))

            else:
                row.append(float(value if value is not None else 0))

        prediction_domain = Orange.data.Domain(attributes)
        data = Orange.data.Table.from_list(
            prediction_domain, [row]
        )

        prediction = float(model(data)[0])

        st.divider()
        st.header("AI 예측 결과")
        st.metric(
            "예상 소비자 관심도",
            f"{max(0, prediction):,.0f}개"
        )

        st.caption(
            "좋아요 수를 기반으로 예측한 관심도이며 "
            "실제 판매량이나 구매량은 아닙니다."
        )

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
