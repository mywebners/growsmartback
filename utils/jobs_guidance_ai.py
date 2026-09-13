"""
Jobs Related Guidance — OpenAI script for Pakistan job portals + apply links.
Reads OPENAI_API_KEY from environment (growsmartback/.env via load_dotenv in app.py).
"""
import json
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError


def build_jobs_user_payload(data):
    field = (data.get("field_or_program") or "").strip()
    level = str(data.get("education_level") or "").strip().lower()

    # Prefer explicit field/program; fall back to older stream/degree fields
    if not field:
        if level == "matric":
            field = (data.get("matric_stream") or "").strip()
        elif level == "inter":
            field = (data.get("intermediate_stream") or "").strip()
        elif level == "bachelor":
            field = (data.get("bachelor_degree") or "").strip()

    return {
        "education_level": data.get("education_level"),
        "field_or_program": field,
        "matric_stream": data.get("matric_stream"),
        "intermediate_stream": data.get("intermediate_stream"),
        "bachelor_degree": data.get("bachelor_degree") or "",
        # Matric / Inter: field only. Bachelor: CGPA / percentage / transcript allowed.
        "matric_marks": {},
        "intermediate_marks": {},
        "bachelor_cgpa": data.get("bachelor_cgpa"),
        "bachelor_percentage": data.get("bachelor_percentage"),
        "has_transcript_image": bool(data.get("transcript_image")),
    }


SYSTEM_SCRIPT = (
    "You are GrowSmart Jobs Advisor for Pakistan.\n"
    "Your job: look at the candidate's education LEVEL and FIELD/PROGRAM and recommend "
    "REALISTIC jobs they can apply for RIGHT NOW in Pakistan.\n\n"
    "Input rules:\n"
    "- Matric / Intermediate: use level + field/program only (no marks required).\n"
    "- Bachelor: also use bachelor_cgpa and/or bachelor_percentage when provided, "
    "and if a transcript image is attached, extract degree name, CGPA/percentage, "
    "and key subjects to improve matching.\n\n"
    "Cover BOTH:\n"
    "1) Government / public: FPSC, PPSC, SPSC, KPPSC, BPSC, NTS, CTS, OTS, Pakistan Army/Navy/Air "
    "Force civilian or cadet tracks where eligible, State Bank / SECP / provincial departments.\n"
    "2) Private: Rozee.pk, Mustakbil, LinkedIn Jobs Pakistan, Indeed Pakistan, Bayt, BrightSpyre, "
    "company career pages when relevant.\n\n"
    "Rules:\n"
    "- Only suggest roles matching their CURRENT qualification level (do not require a degree "
    "they do not have).\n"
    "- Match suggestions to their field/program (e.g. Pre-Engineering, ICS, BS CS, BBA).\n"
    "- Prefer Pakistan-local opportunities.\n"
    "- For every job include 2–4 DIRECT apply/search links (https URLs) the user can open.\n"
    "- Prefer searchable URLs like:\n"
    "  https://www.rozee.pk/\n"
    "  https://www.mustakbil.com/\n"
    "  https://www.linkedin.com/jobs/search/?keywords=JOBTITLE&location=Pakistan\n"
    "  https://pk.indeed.com/jobs?q=JOBTITLE&l=Pakistan\n"
    "  https://www.fpsc.gov.pk/\n"
    "  https://www.ppsc.gop.pk/\n"
    "  https://www.nts.org.pk/\n"
    "- Be honest: if CGPA/percentage is weak, suggest entry-level / internship / skills-based roles too.\n"
    "- Return STRICT JSON only (no markdown) with this shape:\n"
    "{\n"
    '  "summary": string,\n'
    '  "jobs": [\n'
    "    {\n"
    '      "title": string,\n'
    '      "sector": "Government" | "Private" | "Government / Private",\n'
    '      "fit": "Strong" | "Good" | "Possible",\n'
    '      "why": string,\n'
    '      "apply_links": [{"portal": string, "url": string}]\n'
    "    }\n"
    "  ],\n"
    '  "portals": [{"name": string, "url": string}]\n'
    "}\n"
    "- jobs: 6 to 10 items. portals: 6 to 10 useful Pakistan job portals.\n"
)


DEFAULT_PORTALS = [
    {"name": "Rozee.pk", "url": "https://www.rozee.pk/"},
    {"name": "Mustakbil", "url": "https://www.mustakbil.com/"},
    {"name": "LinkedIn Jobs (Pakistan)", "url": "https://www.linkedin.com/jobs/"},
    {"name": "Indeed Pakistan", "url": "https://pk.indeed.com/"},
    {"name": "FPSC", "url": "https://www.fpsc.gov.pk/"},
    {"name": "PPSC", "url": "https://www.ppsc.gop.pk/"},
    {"name": "NTS", "url": "https://www.nts.org.pk/"},
]


def call_jobs_openai(data, openai_api_key, openai_model="gpt-4o-mini"):
    """
    Returns (ok: bool, payload: dict).
    On failure payload is {"message": "..."}.
    """
    if not openai_api_key:
        return False, {
            "message": "OPENAI_API_KEY missing. Add it to growsmartback/.env then restart backend."
        }

    level = str(data.get("education_level") or "").strip().lower()
    if level not in ("matric", "inter", "bachelor"):
        return False, {"message": "education_level must be matric, inter, or bachelor"}

    profile = build_jobs_user_payload(data)
    if not profile.get("field_or_program"):
        return False, {"message": "Please provide your field or program qualification."}

    transcript = data.get("transcript_image")
    user_text = (
        "Candidate qualification (JSON):\n"
        f"{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
        "For Matric/Intermediate use level + field/program only.\n"
        "For Bachelor also use CGPA/percentage and transcript image when provided.\n"
        "Recommend Pakistan jobs + direct apply links for this profile."
    )

    content_parts = [{"type": "text", "text": user_text}]
    if isinstance(transcript, str) and transcript.startswith("data:image"):
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": transcript},
        })

    req_body = {
        "model": openai_model or "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_SCRIPT},
            {"role": "user", "content": content_parts},
        ],
        "temperature": 0.35,
        "response_format": {"type": "json_object"},
    }

    try:
        req = urlrequest.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(req_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            data_obj = json.loads(raw)
            content = data_obj["choices"][0]["message"]["content"]
            parsed = json.loads(content)
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        return False, {"message": f"OpenAI HTTP error: {detail[:400]}"}
    except (URLError, KeyError, ValueError, TimeoutError) as e:
        return False, {"message": f"OpenAI request failed: {str(e)}"}

    jobs = parsed.get("jobs") if isinstance(parsed.get("jobs"), list) else []
    portals = parsed.get("portals") if isinstance(parsed.get("portals"), list) else []
    summary = str(parsed.get("summary") or "").strip()

    cleaned_jobs = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        title = str(job.get("title") or "").strip()
        if not title:
            continue
        links = []
        for link in (job.get("apply_links") or []):
            if not isinstance(link, dict):
                continue
            url = str(link.get("url") or "").strip()
            portal = str(link.get("portal") or "Apply").strip()
            if url.startswith("http"):
                links.append({"portal": portal, "url": url})
        cleaned_jobs.append({
            "title": title,
            "sector": str(job.get("sector") or "Private").strip(),
            "fit": str(job.get("fit") or "Good").strip(),
            "why": str(job.get("why") or "").strip(),
            "apply_links": links[:5],
        })

    cleaned_portals = []
    for p in portals:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "").strip()
        url = str(p.get("url") or "").strip()
        if name and url.startswith("http"):
            cleaned_portals.append({"name": name, "url": url})

    have = {p["url"] for p in cleaned_portals}
    for d in DEFAULT_PORTALS:
        if d["url"] not in have:
            cleaned_portals.append(d)

    if len(cleaned_jobs) < 3:
        return False, {"message": "OpenAI returned too few job suggestions. Try again."}

    return True, {
        "success": True,
        "summary": summary,
        "jobs": cleaned_jobs[:10],
        "portals": cleaned_portals[:12],
        "source": "openai",
        "env": "OPENAI_API_KEY",
    }
