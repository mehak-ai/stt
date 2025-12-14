import streamlit as st
import pdfplumber
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import tempfile
import json
import re

# --- LLM (use OpenAI, Groq or your own) ---
from openai import OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # Put your key in Streamlit Secrets

# ------------------------------
# AGENT: Convert Natural Language → Extraction Plan
# ------------------------------
def ai_extract_instructions(question, html_text):
    prompt = f"""
You are a web scraping expert. 
User question: "{question}"

HTML content:
{html_text[:5000]}

Generate a detailed JSON extraction plan:
- which HTML tags to parse
- which attributes matter
- how to extract the relevant text
- structure of the output

Return ONLY valid JSON.
"""
    res = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": "You convert questions to structured scrape instructions."},
                  {"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# ------------------------------
# Extract content using AI plan
# ------------------------------
def execute_plan(html, plan_json):
    soup = BeautifulSoup(html, "html.parser")

    try:
        plan = json.loads(plan_json)
    except:
        return {"error": "AI returned invalid JSON"}

    results = []

    selectors = plan.get("selectors", [])
    keys = plan.get("keys", ["text"])

    for sel in selectors:
        for tag in soup.select(sel):
            item = {}
            for key in keys:
                if key == "text":
                    item["text"] = tag.get_text(strip=True)
                else:
                    item[key] = tag.get(key)
            results.append(item)

    return results


# ------------------------------
# PDF Extraction
# ------------------------------
def extract_pdf_text(uploaded_pdf):
    full_text = ""
    with pdfplumber.open(uploaded_pdf) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    return full_text


# ------------------------------
# STATIC WEBSITE SCRAPER
# ------------------------------
def scrape_static_page(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    return soup.prettify(), r.text


# ------------------------------
# DYNAMIC WEBSITE SCRAPER (Playwright)
# ------------------------------
def scrape_dynamic_page(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)

        # wait for content
        page.wait_for_load_state("networkidle")

        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "html.parser")
    return soup.prettify(), html


# ==========================================================
#                   STREAMLIT APP UI
# ==========================================================

st.set_page_config(page_title="Agentic AI Scraper", page_icon="🤖", layout="wide")
st.title("🤖 Agentic AI Web & PDF Scraper")

mode = st.radio("Choose extraction mode:", ["Website (Static)", "Website (Dynamic)", "PDF"])

if mode == "Website (Static)":
    url = st.text_input("Enter URL to scrape")
    question = st.text_area("What do you want to extract? (e.g., 'All product titles and prices')")
    if st.button("Scrape Now"):
        pretty_html, raw_html = scrape_static_page(url)
        st.success("Page scraped!")

        st.subheader("AI Generating Extraction Plan...")
        plan = ai_extract_instructions(question, raw_html)
        st.code(plan, language="json")

        st.subheader("Extracting Data...")
        result = execute_plan(raw_html, plan)
        st.json(result)

elif mode == "Website (Dynamic)":
    url = st.text_input("Enter URL to scrape (dynamic websites like Amazon, YouTube, etc.)")
    question = st.text_area("What do you want to extract?")
    if st.button("Scrape Now"):
        pretty_html, raw_html = scrape_dynamic_page(url)
        st.success("Dynamic page rendered & scraped!")

        st.subheader("AI Generating Extraction Plan...")
        plan = ai_extract_instructions(question, raw_html)
        st.code(plan, language="json")

        st.subheader("Extracting Data...")
        result = execute_plan(raw_html, plan)
        st.json(result)

elif mode == "PDF":
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    question = st.text_area("What do you want to extract from the PDF?")

    if uploaded_pdf and st.button("Extract PDF"):
        text = extract_pdf_text(uploaded_pdf)

        st.subheader("PDF Text Extracted")
        st.write(text[:2000] + " ...")

        st.subheader("AI Extracting Relevant Content")
        prompt = f"""
Extract the exact information based on the user question: "{question}".

PDF text:
{text[:7000]}
"""
        answer = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

        st.success("Extraction Complete")
        st.write(answer)
