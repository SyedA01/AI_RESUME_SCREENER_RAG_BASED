
import streamlit as st

from rag_engine import (
    analyze_resume,
    chat_with_resume
)


# ----------------------------------
# 1. Page configuration
# ----------------------------------

st.set_page_config(
    page_title="Resume AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume AI")
st.caption(
    "AI-powered resume analysis and job compatibility assistant"
)


# ----------------------------------
# 2. Initialize session state
# ----------------------------------

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ----------------------------------
# 3. Input section
# ----------------------------------

st.header("Analyze your resume")

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf"]
)

jd_text = st.text_area(
    "Paste your job description",
    height=220,
    placeholder="Paste the complete job description here..."
)

role_override = st.text_input(
    "Target role (optional)",
    placeholder="Example: GenAI Engineer, SDE, AI Engineer"
)


analyze_button = st.button(
    "Analyze Resume",
    type="primary",
    use_container_width=True
)


# ----------------------------------
# 4. Run analysis
# ----------------------------------

if analyze_button:

    if uploaded_file is None:
        st.error("Please upload a resume PDF.")

    elif not jd_text.strip():
        st.error("Please enter a job description.")

    else:

        try:

            with st.spinner(
                "Analyzing resume and job requirements..."
            ):

                result = analyze_resume(
                    uploaded_file.getvalue(),
                    jd_text,
                    role_override.strip() or None
                )

                st.session_state.analysis = result
                st.session_state.chat_history = []

            st.success("Analysis completed successfully!")

        except Exception as e:

            st.error(f"Analysis failed: {str(e)}")


# ----------------------------------
# 5. Display analysis
# ----------------------------------

analysis = st.session_state.analysis

if analysis:

    results = analysis["match_results"]
    profile = analysis["resume_profile"]
    job = analysis["job_profile"]

    st.divider()

    st.header("Analysis Dashboard")

    st.subheader(
        role_override.strip() or job.job_role
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Resume Match Score",
        f"{results['overall_score']}%"
    )

    col2.metric(
        "Required Skills Matched",
        len(results["matched_required"])
    )

    col3.metric(
        "Required Skills Missing",
        len(results["missing_required"])
    )

    st.progress(
        min(results["overall_score"] / 100, 1.0)
    )

    st.caption(
        "This is an indicative skill-coverage score, "
        "not a hiring probability."
    )

    # ----------------------------------
    # Skills analysis
    # ----------------------------------

    st.subheader("Skill Gap Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Matched required skills")

        for skill in results["matched_required"]:
            st.success(f"✓ {skill}")

        st.markdown("#### Matched preferred skills")

        for skill in results["matched_preferred"]:
            st.success(f"✓ {skill}")

    with col2:

        st.markdown("#### Missing required evidence")

        for skill in results["missing_required"]:
            st.warning(f"• {skill}")

        st.markdown("#### Missing preferred evidence")

        for skill in results["missing_preferred"]:
            st.info(f"• {skill}")

    # ----------------------------------
    # Candidate projects
    # ----------------------------------

    st.subheader("Your Projects")

    for project in profile.projects:

        with st.expander(project.name):

            st.write(project.description)

            st.write(
                "**Technologies:**",
                ", ".join(project.technologies)
            )

    # ----------------------------------
    # Recommendations
    # ----------------------------------

    st.divider()

    st.header("Personalized Recommendations")

    st.markdown(analysis["recommendations"])

    # ----------------------------------
    # Chatbot
    # ----------------------------------

    st.divider()

    st.header("💬 Chat with your Resume")

    st.caption(
        "Ask questions about your resume, job requirements, "
        "skill gaps and project improvements."
    )

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask something about your resume..."
    )

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer, sources = chat_with_resume(
                    question,
                    analysis,
                    st.session_state.chat_history[:-1]
                )

                st.markdown(answer)

                with st.expander("Retrieved resume evidence"):

                    for doc in sources:

                        st.write(doc.page_content)

                        st.caption(
                            f"Page: {doc.metadata.get('page', 'Unknown')}"
                        )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })
