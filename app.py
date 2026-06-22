import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="EndoExplain Lab",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #09090f 0%, #11111f 50%, #1a0f1f 100%);
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #171720 0%, #231827 100%);
    border-right: 1px solid rgba(255, 92, 138, 0.25);
}

h1, h2, h3 {
    color: #ffffff;
    letter-spacing: -0.03em;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 18px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #f8fafc;
    border-radius: 12px;
    padding: 10px 16px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #ff4f7b, #a855f7);
    color: white;
}

[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.06);
    padding: 18px;
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.12);
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}

div.stAlert {
    border-radius: 16px;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1350px;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load("endo_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.markdown("""
# 🧬 EndoExplain Lab

### Explainable AI for exploring endometriosis risk, symptom patterns, and diagnostic delay.

This prototype combines machine learning, interpretability, interactive visualization, and healthcare bias analysis into one research-style dashboard.
""")

st.warning(
    "Educational prototype only. This is not a medical diagnosis tool and should not "
    "replace professional medical advice."
)

st.markdown("""
<div style="
    padding: 24px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(255,79,123,0.18), rgba(168,85,247,0.16));
    border: 1px solid rgba(255,255,255,0.14);
    margin-bottom: 28px;
">
<h3>Project Mission</h3>
<p style="font-size: 17px;">
EndoExplain Lab investigates how interpretable machine learning could support earlier recognition of overlooked symptom patterns in women’s healthcare.
</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Patient Symptom Inputs")

age = st.sidebar.slider("Age", 16, 45, 25)
pelvic_pain = st.sidebar.slider("Pelvic pain severity", 0, 10, 5)
painful_periods = st.sidebar.slider("Painful periods severity", 0, 10, 5)
chronic_fatigue = st.sidebar.slider("Chronic fatigue", 0, 10, 5)
gi_symptoms = st.sidebar.slider("Gastrointestinal symptoms", 0, 10, 5)
painful_intercourse = st.sidebar.slider("Painful intercourse", 0, 10, 0)
infertility_history = st.sidebar.selectbox("History of infertility?", [0, 1])
family_history = st.sidebar.selectbox("Family history of endometriosis?", [0, 1])
symptom_duration_years = st.sidebar.slider("Symptom duration in years", 0, 12, 2)
heavy_bleeding = st.sidebar.slider("Heavy bleeding", 0, 10, 5)
back_pain = st.sidebar.slider("Back pain", 0, 10, 5)

input_data = pd.DataFrame([{
    "age": age,
    "pelvic_pain": pelvic_pain,
    "painful_periods": painful_periods,
    "chronic_fatigue": chronic_fatigue,
    "gi_symptoms": gi_symptoms,
    "painful_intercourse": painful_intercourse,
    "infertility_history": infertility_history,
    "family_history": family_history,
    "symptom_duration_years": symptom_duration_years,
    "heavy_bleeding": heavy_bleeding,
    "back_pain": back_pain,
}])

input_data = input_data[feature_names]

prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]

probability_df = pd.DataFrame({
    "Risk Level": model.classes_,
    "Probability": probabilities
})

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Model Importance": model.feature_importances_,
    "Patient Value": input_data.iloc[0].values
})

feature_importance_df["Approximate Contribution"] = (
    feature_importance_df["Model Importance"] * feature_importance_df["Patient Value"]
)

feature_importance_df = feature_importance_df.sort_values(
    by="Approximate Contribution",
    ascending=False
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "Risk Assessment",
    "Model Explanation",
    "3D Risk Landscape",
    "Diagnostic Delay Simulator",
    "Clinical Timeline",
    "Research Navigator",
    "Missed Phenotype Detector",
    "Family Pattern Explorer",
    "Symptom Evolution",
    "Surgery Response",
    "Evidence Map"
])


patient001 = {
    "Age_Symptom_Onset": 19,
    "Diagnosis_Age": 50,
    "Diagnostic_Delay_Years": 31,
    "Constipation": True,
    "Tenesmus": True,
    "Obstructive_Bowel_Sensation": True,
    "Rectal_Mucus": True,
    "Rectal_Bleeding": True,
    "Pelvic_Pain": False,
    "Dysmenorrhoea": False,
    "Infertility_Investigation": False,
    "Symptoms_Improved_During_Amenorrhoea": True,
    "Rectovaginal_Endometriosis": True,
    "Postoperative_Improvement": True,
    "Improvement_Duration_Months": 3,
    "Symptom_Recurrence": True
}

with tab1:
    st.header("Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == "low":
            st.success("Predicted Risk: LOW")
        elif prediction == "medium":
            st.warning("Predicted Risk: MEDIUM")
        else:
            st.error("Predicted Risk: HIGH")

        st.metric("Highest Model Confidence", f"{max(probabilities) * 100:.1f}%")

        st.subheader("Risk Probabilities")
        st.dataframe(probability_df, use_container_width=True)

    with col2:
        fig = px.bar(
            probability_df,
            x="Risk Level",
            y="Probability",
            title="Risk Probability Distribution",
            range_y=[0, 1],
            text=probability_df["Probability"].apply(lambda x: f"{x:.1%}")
        )

        fig.update_traces(
            textposition="outside",
            marker_line_width=1.5
        )

        fig.update_layout(
            template="plotly_dark",
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=14),
            title_font=dict(size=22)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This tab uses the trained machine learning classifier from Replicate 1. "
        "It predicts a risk category and shows probabilities instead of only giving a label."
    )

with tab2:
    st.header("Model Explanation")

    st.write(
        "This tab explains which symptoms most influenced the model’s prediction. "
        "It uses the feature-importance idea from Replicate 2."
    )

    top_contributors = feature_importance_df.head(6)

    fig = px.bar(
        top_contributors.sort_values("Approximate Contribution"),
        x="Approximate Contribution",
        y="Feature",
        orientation="h",
        title="Top Symptom Contributors for This Input"
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14),
        title_font=dict(size=22)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Explanation Table")
    st.dataframe(feature_importance_df, use_container_width=True)

    st.warning(
        "These are approximate technical explanations, not medical causes. "
        "They show how the model used the input values, not what caused a condition."
    )

with tab3:
    st.header("3D Risk Landscape")

    st.write(
        "This graph shows how the predicted high-risk probability changes when pelvic pain "
        "and painful periods change, while the other symptoms stay fixed."
    )

    pain_range = np.arange(0, 11)
    period_range = np.arange(0, 11)

    high_index = list(model.classes_).index("high")
    z_values = []

    for period in period_range:
        row = []
        for pain in pain_range:
            temp = input_data.copy()
            temp["pelvic_pain"] = pain
            temp["painful_periods"] = period
            high_probability = model.predict_proba(temp)[0][high_index]
            row.append(high_probability)
        z_values.append(row)

    fig = go.Figure(data=[
        go.Surface(
            x=pain_range,
            y=period_range,
            z=np.array(z_values),
            colorscale="RdPu"
        )
    ])

    fig.update_layout(
        title="3D Surface: High-Risk Probability",
        template="plotly_dark",
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis_title="Pelvic Pain",
            yaxis_title="Painful Periods",
            zaxis_title="High-Risk Probability"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This makes the app more visual and research-like because users can explore "
        "how the model responds to changing symptoms."
    )

with tab4:
    st.header("Diagnostic Delay Simulator")

    st.write(
        "This simulator explores how repeated non-referral or dismissal can increase "
        "diagnostic delay pressure over time."
    )

    col1, col2 = st.columns(2)

    with col1:
        years_dismissed = st.slider("Years symptoms have been dismissed", 0, 12, symptom_duration_years)
        doctor_visits = st.slider("Doctor visits without referral", 0, 20, 4)
        referral_probability = st.slider("Estimated referral probability per visit", 0.0, 1.0, 0.25)

    probability_no_referral = (1 - referral_probability) ** doctor_visits

    delay_pressure = min(
        1.0,
        (years_dismissed / 12) * 0.5 + probability_no_referral * 0.5
    )

    with col2:
        st.metric("Probability of no referral after visits", f"{probability_no_referral * 100:.1f}%")
        st.metric("Diagnostic delay pressure score", f"{delay_pressure * 100:.1f}%")

    delay_df = pd.DataFrame({
        "Factor": ["Years dismissed", "No-referral probability", "Delay pressure"],
        "Value": [years_dismissed / 12, probability_no_referral, delay_pressure]
    })

    fig = px.bar(
        delay_df,
        x="Factor",
        y="Value",
        title="Diagnostic Delay Pressure Factors",
        range_y=[0, 1],
        text=delay_df["Value"].apply(lambda x: f"{x:.1%}")
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        template="plotly_dark",
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14),
        title_font=dict(size=22)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "This is not a clinical diagnostic-delay model. It is a conceptual simulation "
        "showing how repeated non-referral may compound over time."
    )

with tab5:

    st.header("Clinical Timeline Explorer")

    timeline_df = pd.DataFrame({
        "Age":[19,22,24,45,50,53],
        "Event":[
            "Constipation onset",
            "Amenorrhoea",
            "Major bowel episode",
            "Started HRT",
            "Histology confirmed",
            "Current status"
        ]
    })

    fig = px.scatter(
        timeline_df,
        x="Age",
        y=["Patient 001"] * len(timeline_df),
        hover_name="Event",
        size=[15]*len(timeline_df)
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.metric(
        "Diagnostic Delay",
        f"{patient001['Diagnostic_Delay_Years']} years"
    )

with tab6:

    st.header("Research Navigator")

    research_question = st.selectbox(
        "Choose Question",
        [
            "Amenorrhoea Effect",
            "Diagnostic Delay",
            "Surgical Recurrence",
            "Familial Clustering"
        ]
    )

    if research_question == "Amenorrhoea Effect":

        st.success(
            "Symptoms improved during hypoestrogenic amenorrhoea."
        )

        st.markdown("""
### Potential Mechanisms

1. Estrogen-sensitive lesion activity

2. Reduced inflammatory signalling

3. Reduced fibrosis progression

### Evidence Strength

Moderate to Strong
""")

    elif research_question == "Diagnostic Delay":

        st.metric(
            "Delay",
            f"{patient001['Diagnostic_Delay_Years']} years"
        )

        st.warning(
            "Symptoms began decades before diagnosis."
        )

    elif research_question == "Surgical Recurrence":

        st.metric(
            "Improvement Duration",
            f"{patient001['Improvement_Duration_Months']} months"
        )

        st.markdown("""
Possible explanations:

- residual disease
- fibrosis
- nerve involvement
- incomplete excision
""")

    elif research_question == "Familial Clustering":

        st.write("""
Mother:
- chronic constipation

Sister:
- dysmenorrhoea
- urinary symptoms
- hydronephrosis

Patient:
- bowel-dominant disease
""")

with tab7:

    st.header("Missed Phenotype Detector")

    classic_features = {
        "Pelvic Pain": patient001["Pelvic_Pain"],
        "Dysmenorrhoea": patient001["Dysmenorrhoea"],
        "Infertility Investigation": patient001["Infertility_Investigation"]
    }

    absent_count = sum(v == False for v in classic_features.values())

    atypicality = (
        absent_count /
        len(classic_features)
    ) * 100

    st.metric(
        "Phenotype Atypicality Score",
        f"{atypicality:.0f}%"
    )

    st.error(
        "Histologically confirmed disease despite lacking multiple classic features."
    )

    st.dataframe(
        pd.DataFrame({
            "Feature": classic_features.keys(),
            "Present": classic_features.values()
        })
    )

with tab8:

    st.header("Family Pattern Explorer")

    family_df = pd.DataFrame({
        "Relative":["Mother","Sister","Patient"],
        "Constipation":[1,0,1],
        "Urinary Symptoms":[0,1,0],
        "Dysmenorrhoea":[0,1,0]
    })

    fig = px.imshow(
        family_df.set_index("Relative"),
        text_auto=True,
        aspect="auto"
    )

    fig.update_layout(
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

with tab9:

    st.header("Symptom Evolution Dashboard")

    ages = [19,24,35,45,50,53]

    symptom_df = pd.DataFrame({
        "Age": ages,
        "Constipation":[2,7,8,9,10,10],
        "Tenesmus":[0,1,3,7,9,9],
        "Pelvic Symptoms":[0,0,0,1,2,3]
    })

    fig = px.line(
        symptom_df,
        x="Age",
        y=[
            "Constipation",
            "Tenesmus",
            "Pelvic Symptoms"
        ],
        markers=True
    )

    fig.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

with tab10:

    st.header("Surgery Response Analysis")

    surgery_df = pd.DataFrame({
        "Stage":[
            "Pre-op",
            "Post-op",
            "3 Months",
            "Recurrence"
        ],
        "Symptom Burden":[
            10,
            4,
            3,
            8
        ]
    })

    fig = px.line(
        surgery_df,
        x="Stage",
        y="Symptom Burden",
        markers=True
    )

    fig.update_layout(
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Temporary improvement followed by symptom recurrence."
    )

with tab11:

    st.header("Evidence Map")

    support_df = pd.DataFrame({
        "Supporting Evidence":[
            "Histology confirmed",
            "Improved during amenorrhoea",
            "Improved after surgery"
        ]
    })

    contradiction_df = pd.DataFrame({
        "Contradictory Features":[
            "No dysmenorrhoea",
            "No pelvic pain",
            "Non-cyclic symptoms"
        ]
    })

    col1, col2 = st.columns(2)

    with col1:
        st.success("Supporting Evidence")
        st.dataframe(support_df)

    with col2:
        st.warning("Atypical Features")
        st.dataframe(contradiction_df)

    st.metric(
        "Research Confidence",
        "Moderate"
    )



st.markdown("---")

st.write(
    "Built for a TKS portfolio project exploring explainable AI, women’s healthcare, "
    "model interpretability, and responsible machine learning."
)