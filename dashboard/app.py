"""
Streamlit Admin Dashboard for Course Advisor Agent.

Run: streamlit run dashboard/app.py
Requires API running at API_BASE_URL (default http://localhost:7860).
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:7860")

st.set_page_config(page_title="Course Advisor Admin", layout="wide", page_icon="🎓")


def api_get(path: str, token: str, params: dict | None = None) -> dict | list:
    resp = requests.get(
        f"{API_BASE}{path}",
        headers={"Authorization": f"Bearer {token}"},
        params=params or {},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, payload: dict) -> dict:
    resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def login_page() -> None:
    st.title("Admin Login")
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            try:
                data = api_post("/auth/login", {"email": email, "password": password})
                if data.get("user", {}).get("role") != "admin":
                    st.error("Admin role required for this dashboard.")
                    return
                st.session_state["token"] = data["access_token"]
                st.session_state["user"] = data["user"]
                st.rerun()
            except requests.HTTPError as exc:
                st.error(f"Login failed: {exc.response.text if exc.response else exc}")


def sidebar_nav() -> str:
    st.sidebar.title("Course Advisor")
    st.sidebar.caption(f"API: {API_BASE}")
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    return st.sidebar.radio(
        "Pages",
        ["Dashboard", "Users", "CRM Tickets", "Cost Monitor", "Behavior Trace"],
        label_visibility="collapsed",
    )


def page_dashboard(token: str) -> None:
    st.header("Dashboard")
    stats = api_get("/admin/stats", token)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Users", stats["total_users"])
    c2.metric("Conversations", stats["total_conversations"])
    c3.metric("Messages", stats["total_messages"])
    c4.metric("Leads", stats["total_leads"])
    c5.metric("Total Cost", f"${stats['total_cost']:.6f}")

    c6, c7, c8 = st.columns(3)
    c6.metric("Avg Cost", f"${stats['average_cost']:.8f}")
    c7.metric("Avg Latency", f"{stats['average_latency_ms']:.1f} ms")
    c8.metric("LLM Cost", f"${stats['llm_cost']:.6f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Daily Cost")
        daily = stats.get("daily_cost", [])
        if daily:
            df = pd.DataFrame(daily)
            st.plotly_chart(px.bar(df, x="date", y="cost"), use_container_width=True)
    with col_b:
        st.subheader("Provider Distribution")
        prov = stats.get("provider_distribution", [])
        if prov:
            df = pd.DataFrame(prov)
            st.plotly_chart(px.pie(df, names="name", values="count"), use_container_width=True)

    st.subheader("Top Expensive Users")
    st.dataframe(pd.DataFrame(stats.get("top_expensive_users", [])), use_container_width=True)
    st.subheader("Optimization Metrics")
    st.json(stats.get("optimization", {}))


def page_users(token: str) -> None:
    st.header("Users")
    data = api_get("/admin/users", token)
    st.dataframe(pd.DataFrame(data.get("users", [])), use_container_width=True)


def page_tickets(token: str) -> None:
    st.header("CRM Tickets")
    data = api_get("/admin/tickets", token)
    st.dataframe(pd.DataFrame(data.get("tickets", [])), use_container_width=True)


def page_costs(token: str) -> None:
    st.header("Cost Monitor")
    c1, c2, c3 = st.columns(3)
    user_id = c1.text_input("User ID")
    conversation_id = c2.text_input("Conversation ID")
    provider = c3.text_input("Provider")
    model = st.text_input("Model")
    params = {k: v for k, v in {
        "user_id": user_id or None,
        "conversation_id": conversation_id or None,
        "provider": provider or None,
        "model": model or None,
    }.items() if v}
    data = api_get("/admin/costs", token, params=params)
    agg = data.get("aggregate", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Cost", f"${agg.get('total_cost', 0):.6f}")
    m2.metric("LLM Cost", f"${agg.get('prompt_cost', 0) + agg.get('completion_cost', 0):.6f}")
    m3.metric("Embedding Cost", f"${agg.get('embedding_cost', 0):.6f}")
    m4.metric("Calls", agg.get("count", 0))
    logs = data.get("logs", [])
    if logs:
        df = pd.DataFrame(logs)
        st.dataframe(df, use_container_width=True)
        st.plotly_chart(
            px.bar(df.head(50), x="timestamp", y="total_cost", color="provider"),
            use_container_width=True,
        )


def page_traces(token: str) -> None:
    st.header("Behavior Trace")
    conversation_id = st.text_input("Filter by Conversation ID")
    params = {"conversation_id": conversation_id} if conversation_id else {}
    data = api_get("/admin/traces", token, params=params)
    traces = data.get("traces", [])
    if not traces:
        st.info("No traces found.")
        return
    selected = st.selectbox(
        "Select trace",
        traces,
        format_func=lambda t: f"{t.get('timestamp', '')[:19]} — {t.get('user_prompt', '')[:60]}",
    )
    if selected:
        st.subheader("Debugger Replay")
        for step in selected.get("replay_steps", []):
            with st.expander(f"▶ {step['step']}", expanded=step["step"] in ("User Prompt", "Assistant Reply")):
                st.write(step["data"])


def main() -> None:
    if "token" not in st.session_state:
        login_page()
        return
    token = st.session_state["token"]
    page = sidebar_nav()
    try:
        if page == "Dashboard":
            page_dashboard(token)
        elif page == "Users":
            page_users(token)
        elif page == "CRM Tickets":
            page_tickets(token)
        elif page == "Cost Monitor":
            page_costs(token)
        elif page == "Behavior Trace":
            page_traces(token)
    except requests.HTTPError as exc:
        st.error(f"API error: {exc.response.text if exc.response else exc}")


if __name__ == "__main__":
    main()
