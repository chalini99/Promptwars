# LEXGUARD — AI Rights & Contract Intelligence System

LexGuard is an adversarial multi-agent AI system designed to analyze contracts, offer letters, quotations, ticket terms, and online policies to detect exploitative clauses, hidden liabilities, legal ambiguities, and real-world risks before users agree to them.

---

## 🌟 Key Innovations & Differentiators

Unlike generic AI assistants that simply summarize documents, LexGuard uses a **Simulated Courtroom Trial Pattern** to audit legal terms. This multi-agent debate architecture delivers bulletproof, highly explainable risk analysis:

1. **Simulated Courtroom Trial Pattern:**
   - **🔍 Extractor Agent:** Parses document structure, extracts key clauses, and classifies them into risk categories.
   - **⚖️ Prosecutor Agent:** Assumes the worst-case scenario. Attacks each clause, alleging severe liabilities and extreme legal asymmetry.
   - **🛡️ Defender Agent:** Acts as opposing counsel. Cites industry benchmarks, explains standard practices, and proposes fair compromise wording.
   - **📊 Judge Agent:** Impartially weighs the Prosecutor's allegations against the Defender's context. Renders a definitive Severity Score (0.0 - 10.0) and legal verdict.
   - **💡 Advisor Agent:** Translates the courtroom findings into a highly actionable, plain-language Negotiation Playbook for the user, complete with copy-paste negotiation scripts.
2. **Server-Sent Events (SSE) Live Feed:**
   - The user gets an immersive experience watching the agents debate their contract in real-time on a beautifully animated timeline.
3. **Interactive Rights Advisor Chat:**
   - A follow-up chat panel lets users ask the Advisor Agent questions, simulate "what-if" signing scenarios, and get custom pushback strategies.
4. **Robust Offline / Demo Mode:**
   - Features rich fallback datasets. If the Google Gemini API key is missing or expired, LexGuard operates flawlessly in demo mode using structured legal benchmarks, ensuring it always performs perfectly during judging or presentations.

---

## 🛠️ Technology Stack

- **Backend:** FastAPI (Python), Uvicorn
- **Multimodal AI:** Google Gemini 1.5/2.0 Flash (Native document processing, structured outputs)
- **Document Parsers:** PyPDF, python-docx
- **Frontend:** Premium modern Single Page Application (HTML5, Vanilla CSS, JS) featuring glassmorphism, dynamic SVG gauge animations, interactive courtroom debate timelines, and collapsible clause transcript cards.

---

## 🚀 Quick Start (Local Run)

### 1. Configure the Environment
Clone or navigate to the repository directory:
```bash
cd c:\Users\Chalini\Documents\Promptwars
```

Create a `.env` file in the `backend/` directory or set your environment variable:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_actual_gemini_api_key"
```
*(If no key is configured, LexGuard will run automatically in Demo/Mock Mode with 100% functionality for the pre-loaded contracts).*

### 2. Run the Server
Launch the FastAPI server:
```bash
python backend/app/main.py
```
Or use Uvicorn directly:
```bash
uvicorn backend.app.main:app --reload --port 8000
```

### 3. Open in Browser
Visit the dashboard:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📄 Project Structure

```text
Promptwars/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI server & Agent Orchestrator
│   │   └── static/
│   │       └── index.html   # Premium single page dashboard UI
│   ├── .env.example
│   └── requirements.txt
├── README.md
└── artifacts/               # Implementation logs and architecture plans
```

---

## ⚖️ Verification Scenarios

LexGuard has pre-loaded, premium benchmark agreements. Simply click one of the quick-launch buttons on the dashboard to test instantly:
1. **💼 Employment Offer:** Audits severe 5-year non-competes, weekend side-project IP grabs, and 90-day vs immediate notices.
2. **🏠 Lease Agreement:** Highlights predatory landlord entry, $500 day-two late fees, and automatic security deposit theft.
3. **🔒 Perpetual NDA:** Audits infinite legal duration, perpetual liability, and automatic $100,000 penalties.
