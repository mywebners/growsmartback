"""
Jobs Related Guidance — OpenAI script for Pakistan job portals + apply links.
Reads OPENAI_API_KEY from environment (growsmartback/.env via load_dotenv in app.py).
"""
import json
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError

from utils.openai_errors import friendly_openai_error


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
    "Recommend realistic jobs this student can apply for NOW.\n\n"
    "Rules:\n"
    "1) Match education level strictly:\n"
    "   - matric = helper / clerk / trainee / entry roles only\n"
    "   - inter = internship / junior / trainee roles (no full bachelor jobs)\n"
    "   - bachelor = junior / associate jobs in their field\n"
    "2) Match the field/program closely (e.g. ICS → IT support; Pre-Med → lab/health support).\n"
    "3) Include both Government and Private options.\n"
    "4) Each job needs 2–4 real https apply/search links "
    "(Rozee, Mustakbil, LinkedIn Pakistan, Indeed Pakistan, FPSC, PPSC, NTS).\n"
    "5) Write summary and why in simple clear English (short sentences).\n"
    "6) If CGPA/percentage is low, suggest trainee/internship roles and say so politely.\n"
    "7) Return ONLY valid JSON (no markdown):\n"
    "{\n"
    '  "summary": "2-3 short sentences",\n'
    '  "jobs": [\n'
    "    {\n"
    '      "title": "job title",\n'
    '      "sector": "Government" | "Private" | "Government / Private",\n'
    '      "fit": "Strong" | "Good" | "Possible",\n'
    '      "why": "1 short clear sentence",\n'
    '      "apply_links": [{"portal": "name", "url": "https://..."}]\n'
    "    }\n"
    "  ],\n"
    '  "portals": [{"name": "portal", "url": "https://..."}]\n'
    "}\n"
    "Give 7-10 jobs and 7-10 portals."
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
    cgpa = profile.get("bachelor_cgpa")
    pct = profile.get("bachelor_percentage")
    user_text = (
        "Student details for job advice:\n"
        f"- Education level: {level}\n"
        f"- Field / program: {profile.get('field_or_program')}\n"
        f"- Bachelor CGPA: {cgpa if cgpa not in (None, '') else 'not given'}\n"
        f"- Bachelor percentage: {pct if pct not in (None, '') else 'not given'}\n"
        f"- Transcript image attached: {'yes' if isinstance(transcript, str) and transcript.startswith('data:image') else 'no'}\n\n"
        "Task:\n"
        "1) Suggest 7-10 realistic Pakistan jobs for THIS student.\n"
        "2) Keep titles suitable for their education level.\n"
        "3) For each job give a short clear why + 2-4 working https links.\n"
        "4) Also list useful Pakistan job portals.\n"
        "5) Write all text in simple clear English.\n"
        "Return JSON only."
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
        "temperature": 0.25,
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
        return False, {"message": friendly_openai_error(detail, e.code)}
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
