"""
US-based CV Maker — OpenAI generates a professional US resume from user profile.
"""
import json
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError

from utils.openai_errors import friendly_openai_error


SYSTEM_SCRIPT = (
    "You are GrowSmart CV Writer for United States resumes.\n"
    "Make a clean, ATS-friendly US resume from the student facts only.\n\n"
    "Rules:\n"
    "1) Do NOT invent jobs, degrees, dates, companies, or numbers.\n"
    "2) You may improve grammar and wording only.\n"
    "3) Use American English.\n"
    "4) Write a clear Professional Summary (3-4 short lines) for the target role.\n"
    "5) Experience bullets must start with action verbs and stay factual.\n"
    "6) Keep the resume short (about one page).\n"
    "7) plain_text must be ready to copy, with clear section titles.\n"
    "8) Write all text in simple clear English.\n"
    "9) Return ONLY valid JSON (no markdown):\n"
    "{\n"
    '  "header": {\n'
    '    "full_name": "", "headline": "", "email": "", "phone": "",\n'
    '    "location": "", "linkedin": "", "portfolio": ""\n'
    "  },\n"
    '  "summary": "",\n'
    '  "skills": [""],\n'
    '  "experience": [{"title": "", "company": "", "location": "", "dates": "", "bullets": [""]}],\n'
    '  "education": [{"degree": "", "school": "", "location": "", "dates": "", "details": ""}],\n'
    '  "projects": [{"name": "", "description": "", "tech": ""}],\n'
    '  "certifications": [""],\n'
    '  "languages": [""],\n'
    '  "achievements": [""],\n'
    '  "plain_text": ""\n'
    "}\n"
    "Leave unused lists empty."
)


def _clean_list(items):
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def call_cv_openai(data, openai_api_key, openai_model="gpt-4o-mini"):
    """
    Returns (ok: bool, payload: dict).
    On failure payload is {"message": "..."}.
    """
    if not openai_api_key:
        return False, {
            "message": "OPENAI_API_KEY missing. Add it to growsmartback/.env then restart backend."
        }

    full_name = str(data.get("full_name") or "").strip()
    email = str(data.get("email") or "").strip()
    if not full_name:
        return False, {"message": "Full name is required."}
    if not email:
        return False, {"message": "Email is required."}

    profile = {
        "full_name": full_name,
        "email": email,
        "phone": str(data.get("phone") or "").strip(),
        "location": str(data.get("location") or "").strip(),
        "linkedin": str(data.get("linkedin") or "").strip(),
        "portfolio": str(data.get("portfolio") or "").strip(),
        "target_role": str(data.get("target_role") or "").strip(),
        "summary_notes": str(data.get("summary_notes") or "").strip(),
        "skills": str(data.get("skills") or "").strip(),
        "education": data.get("education") if isinstance(data.get("education"), list) else [],
        "experience": data.get("experience") if isinstance(data.get("experience"), list) else [],
        "projects": str(data.get("projects") or "").strip(),
        "certifications": str(data.get("certifications") or "").strip(),
        "languages": str(data.get("languages") or "").strip(),
        "achievements": str(data.get("achievements") or "").strip(),
    }

    target = profile["target_role"] or "General professional role"
    user_text = (
        "Make a US resume from these student facts only.\n\n"
        f"Target role: {target}\n"
        f"Full name: {profile['full_name']}\n"
        f"Email: {profile['email']}\n"
        f"Phone: {profile['phone'] or 'not given'}\n"
        f"Location: {profile['location'] or 'not given'}\n"
        f"LinkedIn: {profile['linkedin'] or 'not given'}\n"
        f"Portfolio: {profile['portfolio'] or 'not given'}\n"
        f"Skills: {profile['skills'] or 'not given'}\n"
        f"Summary notes: {profile['summary_notes'] or 'not given'}\n"
        f"Projects: {profile['projects'] or 'not given'}\n"
        f"Certifications: {profile['certifications'] or 'not given'}\n"
        f"Languages: {profile['languages'] or 'not given'}\n"
        f"Achievements: {profile['achievements'] or 'not given'}\n"
        f"Education JSON: {json.dumps(profile['education'], ensure_ascii=False)}\n"
        f"Experience JSON: {json.dumps(profile['experience'], ensure_ascii=False)}\n\n"
        "Task:\n"
        "1) Write a clear summary and headline for the target role.\n"
        "2) Polish experience/education/projects using only given facts.\n"
        "3) Build a clean skills list.\n"
        "4) Create complete plain_text resume.\n"
        "5) Use simple clear English.\n"
        "Return JSON only."
    )

    req_body = {
        "model": openai_model or "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_SCRIPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.3,
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
        with urlrequest.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8")
            data_obj = json.loads(raw)
            content = data_obj["choices"][0]["message"]["content"]
            parsed = json.loads(content)
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        return False, {"message": friendly_openai_error(detail, e.code)}
    except (URLError, KeyError, ValueError, TimeoutError) as e:
        return False, {"message": f"OpenAI request failed: {str(e)}"}

    if not isinstance(parsed, dict):
        return False, {"message": "OpenAI returned invalid CV payload."}

    header = parsed.get("header") if isinstance(parsed.get("header"), dict) else {}
    experience = parsed.get("experience") if isinstance(parsed.get("experience"), list) else []
    education = parsed.get("education") if isinstance(parsed.get("education"), list) else []
    projects = parsed.get("projects") if isinstance(parsed.get("projects"), list) else []

    cleaned_experience = []
    for item in experience:
        if not isinstance(item, dict):
            continue
        cleaned_experience.append({
            "title": str(item.get("title") or "").strip(),
            "company": str(item.get("company") or "").strip(),
            "location": str(item.get("location") or "").strip(),
            "dates": str(item.get("dates") or "").strip(),
            "bullets": _clean_list(item.get("bullets")),
        })

    cleaned_education = []
    for item in education:
        if not isinstance(item, dict):
            continue
        cleaned_education.append({
            "degree": str(item.get("degree") or "").strip(),
            "school": str(item.get("school") or "").strip(),
            "location": str(item.get("location") or "").strip(),
            "dates": str(item.get("dates") or "").strip(),
            "details": str(item.get("details") or "").strip(),
        })

    cleaned_projects = []
    for item in projects:
        if not isinstance(item, dict):
            continue
        cleaned_projects.append({
            "name": str(item.get("name") or "").strip(),
            "description": str(item.get("description") or "").strip(),
            "tech": str(item.get("tech") or "").strip(),
        })

    result = {
        "header": {
            "full_name": str(header.get("full_name") or full_name).strip(),
            "headline": str(header.get("headline") or profile["target_role"]).strip(),
            "email": str(header.get("email") or email).strip(),
            "phone": str(header.get("phone") or profile["phone"]).strip(),
            "location": str(header.get("location") or profile["location"]).strip(),
            "linkedin": str(header.get("linkedin") or profile["linkedin"]).strip(),
            "portfolio": str(header.get("portfolio") or profile["portfolio"]).strip(),
        },
        "summary": str(parsed.get("summary") or "").strip(),
        "skills": _clean_list(parsed.get("skills")),
        "experience": cleaned_experience,
        "education": cleaned_education,
        "projects": cleaned_projects,
        "certifications": _clean_list(parsed.get("certifications")),
        "languages": _clean_list(parsed.get("languages")),
        "achievements": _clean_list(parsed.get("achievements")),
        "plain_text": str(parsed.get("plain_text") or "").strip(),
        "source": "openai",
        "style": "US-based",
    }

    if not result["plain_text"]:
        return False, {"message": "OpenAI did not return plain_text CV content."}

    return True, result
