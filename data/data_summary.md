# Kayfa Sales Agent — Data Summary

This document summarizes all files in the `data/` folder. These files are the **knowledge base** for building an agentic AI sales agent that helps clients discover Kayfa courses and guides them toward enrollment.

**Folder structure:**

```
data/
├── json/
│   ├── kayfa_courses.json      # Structured catalog of individual courses
│   └── kayfa_roadmaps.json     # Structured catalog of tracks & diplomas
└── text/
    ├── kayfa_company_overview.md
    ├── kayfa_policies_and_faqs.md
    ├── kayfa_privacy_policy.md
    ├── kayfa_instructor_network.md
    ├── kayfa_paid_individual_courses.md
    ├── kayfa_paid_educational_tracks.md
    ├── kayfa_free_educational_content.md
    ├── kayfa_ai_diploma.md
    ├── kayfa_data_science_diploma.md
    ├── kayfa_soc_diploma.md
    ├── Kayfa_PenTest_Diploma.md
    └── Kayfa_Fullstack_Diploma.md
```

---

## How the data fits together

| Layer | Files | Best for |
|-------|-------|----------|
| **Structured lookup** | `kayfa_courses.json`, `kayfa_roadmaps.json` | Tool-based retrieval: search by skill, track, level, duration, links |
| **Sales pitches** | Diploma `.md` files in `text/` | Persuasive product descriptions, objection handling, closing lines |
| **Pricing & catalogs** | `kayfa_paid_*.md`, `kayfa_free_educational_content.md` | Quick price/duration tables for recommendations |
| **Trust & operations** | `kayfa_company_overview.md`, policies, privacy, instructors | Company identity, FAQs, refunds, contacts, credibility |

---

## JSON files (`data/json/`)

### `kayfa_courses.json`

**What it is:** A machine-readable catalog of **52 courses** (paid, free, tips, and live sessions).

**Each course entry includes:**
- `id`, `name`, `summary`
- `track` (e.g. `data_science`, `soc`, `frontend`)
- `level` (beginner / intermediate / advanced)
- `duration`, `prerequisites`, `link`
- `roadmaps` — which learning paths include this course

**Use it when the agent needs to:** look up a specific course, filter by topic or level, or link a client to the course page.

**Sample course categories in the file:**
- Data Science & Analysis (Python, ML, Power BI, Excel, SQL)
- Cybersecurity & SOC (Splunk, QRadar, Linux, Active Directory)
- Web Development (HTML, CSS, JavaScript, TypeScript, C#)
- AI (ChatGPT, AI tools, Python for AI)
- Creative (Graphic design, motion graphics, CapCut)
- Free content (tips, free sessions)
- Career & business (Career planning, marketing)

---

### `kayfa_roadmaps.json`

**What it is:** A structured catalog of **13 learning paths** — 10 on-demand tracks plus 3 live bootcamp diplomas.

**Each roadmap entry includes:**
- `id`, `name`, `summary`
- `skills`, `tools`, `duration`, `courses_count`, `link`
- `courses_list` — ordered list of course IDs from `kayfa_courses.json`

**Tracks (on-demand, self-paced):**

| ID | Name | Duration | Courses |
|----|------|----------|---------|
| `kayfa_data_science_track` | Data Science Track | 75h 31m | 12 |
| `kayfa_soc_track` | SOC Track | 44h 52m | 9 |
| `kayfa_web_development_track` | Web Development Track | 45h 45m | 9 |
| `kayfa_data_analysis_track` | Data Analysis Track | 38h 54m | 8 |
| `kayfa_frontend_track` | Frontend Track | 28h 12m | 6 |
| `kayfa_backend_track` | Backend Track | 23h 34m | 6 |
| `kayfa_ai_fundamentals_track` | AI Fundamentals Track | 8h 44m | 3 |
| `kayfa_graphics_motion_track` | Graphics & Motion Track | 6h 14m | 2 |
| `kayfa_video_editing_track` | Video Editing Track | 6h 8m | 1 |
| `kayfa_crash_courses_track` | Crash Courses Track | 2h 7m | 3 |

**Diplomas / bootcamps (live, intensive programs):**

| ID | Name | Duration |
|----|------|----------|
| `kayfa_ai_diploma` | AI Track Diploma | 5 months |
| `kayfa_data_science_diploma` | Data Science Track Diploma | 5 months |
| `kayfa_soc_diploma` | SOC Track Diploma | 5 months |

**Use it when the agent needs to:** recommend a full path (not just one course), explain skills/tools gained, or connect courses into a career roadmap.

---

## Markdown files (`data/text/`)

### Company & policies

| File | Summary |
|------|---------|
| **`kayfa_company_overview.md`** | Kayfa identity: Arabic e-learning platform by Kayfa Digital Solutions, IAO-accredited. Mission, target audience (students, professionals, freelancers), self-paced model, certificates, NGen program (ages 8–18). Contact emails, phone numbers (UAE, Egypt, Syria), and office addresses. |
| **`kayfa_policies_and_faqs.md`** | Subscription rules (annual, auto-renew), payment methods, refund policy for live vs recorded courses, course replacement rules, and common FAQs (access, deadlines, previews, support). |
| **`kayfa_privacy_policy.md`** | What personal data Kayfa collects, why it is used, and user rights (access, rectification, erasure, etc.). Contact: info@kayfa.io. |

---

### Catalogs (pricing & listings)

| File | Summary |
|------|---------|
| **`kayfa_paid_individual_courses.md`** | Table of **14 paid standalone courses** with video count, hours, USD price ($15–$65), and instructor. Examples: Business Statistics ($65), Linux Fundamentals ($40), Splunk SIEM Case Studies ($50). |
| **`kayfa_paid_educational_tracks.md`** | Table of **10 paid tracks** with total videos, hours, course count, and USD price ($25–$250). Top tracks: Data Science ($250), SOC ($250), Web Development ($200). |
| **`kayfa_free_educational_content.md`** | Table of **11 free offerings**: tips series (SOC, AI, programming, data science), free sessions, intro courses, and live session recordings. Good entry point for hesitant prospects. |
| **`kayfa_instructor_network.md`** | Table of **25 instructors** with course count and professional affiliation (e.g. Osama Salem — CEO Etica Technology; Mohamed Ali — Full-Stack Developer). Builds trust and expertise credibility. |

---

### Diploma / bootcamp sales documents (live programs)

These files are written as **sales-ready product briefs** — pitch lines, curriculum highlights, career outcomes, trust signals, and closing value statements. Some include objection handling.

| File | Summary |
|------|---------|
| **`kayfa_ai_diploma.md`** | **AI Track** — 5-month live program. Covers generative AI, RAG, AI agents, deep learning, CV, NLP, MLOps (Docker, CI/CD, MLflow). Business English, CV/interview prep. Accreditation options (University of Delaware, Leeds Academy). 15,000+ learners; partners: Microsoft, GIZ, Paymob. |
| **`kayfa_data_science_diploma.md`** | **Data Science Track** — 5-month hybrid program. Python, Power BI, SQL Server, Advanced Excel, business statistics, ML, 2 capstone projects. Career modules and triple certification options. |
| **`kayfa_soc_diploma.md`** | **SOC Track** — 5-month hybrid program. Network security, Linux, Windows AD, Splunk, IBM QRadar, incident response, threat hunting, forensics. Real attack simulations and capstone projects. |
| **`Kayfa_PenTest_Diploma.md`** | **Penetration Testing & Ethical Hacking** — 48-hour offensive security program (24 live sessions). Wireshark, Nmap, Burp Suite, Hydra, Hashcat, post-exploitation, professional reporting. Includes sales objection handling, job targets (junior pentester, bug bounty), and instructor: Eng. Saleh Al-Hourani. |
| **`Kayfa_Fullstack_Diploma.md`** | **Fullstack Development Diploma** — 6 months, 55 live sessions. Frontend (HTML → React → Next.js) and backend (Node.js, Express, MongoDB, MySQL, auth, deployment). 7 portfolio projects, personal subdomain, CV guidance. No prior coding required. |

---

## Quick reference: product tiers

| Tier | Where to look | Typical price range |
|------|---------------|---------------------|
| **Free content** | `kayfa_free_educational_content.md`, free entries in `kayfa_courses.json` | $0 |
| **Individual courses** | `kayfa_paid_individual_courses.md`, `kayfa_courses.json` | $15–$65 |
| **On-demand tracks** | `kayfa_paid_educational_tracks.md`, `kayfa_roadmaps.json` | $25–$250 |
| **Live diplomas / bootcamps** | Diploma `.md` files, bootcamp entries in `kayfa_roadmaps.json` | Contact / program-specific |

---

## Suggested exploration order for interns

1. **`kayfa_company_overview.md`** — Understand who Kayfa is and how to contact support.
2. **`kayfa_roadmaps.json`** — See the full product map (tracks + diplomas).
3. **`kayfa_courses.json`** — Drill into individual courses linked to each roadmap.
4. **Diploma `.md` files** — Study sales language, pitches, and objection handling.
5. **`kayfa_paid_*.md` and `kayfa_free_educational_content.md`** — Use for quick price and duration answers.
6. **`kayfa_policies_and_faqs.md`** — Answer subscription, refund, and access questions accurately.
7. **`kayfa_instructor_network.md`** — Add credibility when recommending programs.