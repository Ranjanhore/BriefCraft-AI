    {
        "id": "institution_team",
        "audience": "institution",
        "name": "Institution Team",
        "price_inr": 14999,
        "billing": "monthly",
        "credits": 300000,
        "features": [
            "Multi-user institution workspace",
            "Shared credit wallet",
            "Agent access controls",
            "Monthly usage report",
        ],
        "recommended": True,
    },
    {
        "id": "institution_enterprise",
        "audience": "institution",
        "name": "Institution Enterprise",
        "price_inr": None,
        "billing": "custom",
        "credits": 1500000,
        "features": [
            "Dedicated workspace",
            "Custom backend agents",
            "Higher credit limits",
            "Onboarding and priority support",
        ],
        "recommended": False,
    },
]


def _account_user_id(
    request: Request,
    x_user_id: Optional[str] = None,
    authorization: Optional[str] = None,
) -> str:
    """Return a stable account id without assuming a specific auth provider."""
    explicit = (x_user_id or "").strip()
    if explicit:
        return explicit
    token = (authorization or request.headers.get("authorization") or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if token:
        return "token-" + uuid.uuid5(uuid.NAMESPACE_URL, token).hex[:24]
    return "guest"


def _ensure_account(user_id: str) -> Dict[str, Any]:
    if user_id not in ACCOUNT_STORE:
        ACCOUNT_STORE[user_id] = {
            "user_id": user_id,
            "credit_balance": DEFAULT_CREDIT_GRANT,
            "plan_id": "free_trial",
            "plan_name": "Free Trial",
            "account_type": "individual",
            "currency": "INR",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        CREDIT_LEDGER[user_id] = [
            {
                "id": str(uuid.uuid4()),
                "type": "grant",
                "amount": DEFAULT_CREDIT_GRANT,
                "balance_after": DEFAULT_CREDIT_GRANT,
                "reason": "Welcome credits",
                "ts": time.time(),
            }
        ]
    return ACCOUNT_STORE[user_id]


def _grant_credits(user_id: str, amount: int, reason: str, package_id: Optional[str] = None) -> Dict[str, Any]:
    acct = _ensure_account(user_id)
    acct["credit_balance"] = int(acct.get("credit_balance", 0)) + max(0, int(amount))
    acct["updated_at"] = time.time()
    row = {
        "id": str(uuid.uuid4()),
        "type": "grant",
        "amount": max(0, int(amount)),
        "balance_after": acct["credit_balance"],
        "reason": reason,
        "package_id": package_id,
        "ts": time.time(),
    }
    CREDIT_LEDGER.setdefault(user_id, []).append(row)
    return acct


def _consume_credits(user_id: str, amount: int, reason: str, agent_id: Optional[str] = None, project_id: Optional[str] = None) -> Dict[str, Any]:
    amount = max(0, int(amount))
    acct = _ensure_account(user_id)
    balance = int(acct.get("credit_balance", 0))
    if amount > balance:
        raise HTTPException(
            status_code=402,
            detail={
                "message": "Insufficient credits.",
                "required": amount,
                "balance": balance,
                "upgrade_endpoint": "/account/packages",
            },
        )
    acct["credit_balance"] = balance - amount
    acct["updated_at"] = time.time()
    CREDIT_LEDGER.setdefault(user_id, []).append({
        "id": str(uuid.uuid4()),
        "type": "debit",
        "amount": -amount,
        "balance_after": acct["credit_balance"],
        "reason": reason,
        "agent_id": agent_id,
        "project_id": project_id,
        "ts": time.time(),
    })
    return acct


def register_backend_agent(
    agent_id: str,
    name: str,
    description: str,
    endpoint: str,
    capabilities: Optional[List[str]] = None,
    credit_cost: int = 25,
    category: str = "creative",
    enabled: bool = True,
) -> None:
    """Register an agent so the frontend can discover it automatically."""
    AGENT_REGISTRY[agent_id] = {
        "id": agent_id,
        "name": name,
        "description": description,
        "endpoint": endpoint,
        "capabilities": capabilities or [],
        "credit_cost": credit_cost,
        "category": category,
        "enabled": enabled,
    }


def _register_default_agents() -> None:
    register_backend_agent(
        "ACCOUNT_AGENT",
        "Account Agent",
        "Manages user credits, balances, plans, upgrades, and package recommendations.",
        "/account/agent",
        ["credits", "balance", "packages", "upgrade", "billing"],
        credit_cost=0,
        category="account",
    )
    register_backend_agent(
        "UCD_AGENT",
        "Universal Creative Director",
        "Routes creative requests, asks missing questions, and coordinates specialist agents.",
        "/ucd/chat",
        ["brief", "concept", "handoff", "orchestration"],
        credit_cost=15,
        category="orchestrator",
    )
    register_backend_agent(
        "CAD_AGENT",
        "Professional CAD Agent",
        "Generates or traces production CAD layouts with SVG/PDF/DXF outputs.",
        "/api/cad/pro/generate",
        ["cad", "layout", "trace", "dxf", "pdf", "svg"],
        credit_cost=250,
        category="production",
    )


_register_default_agents()


def _account_agent_reply(user_id: str, message: str) -> Dict[str, Any]:
    acct = _ensure_account(user_id)
    text = (message or "").lower()
    if any(w in text for w in ["package", "plan", "upgrade", "price", "pay", "institution", "individual"]):
        msg = (
            "Here are the available packages. Individuals usually start with Individual Pro; "
            "institutions should compare Institution Team and Enterprise based on team size and monthly generation volume."
        )
        return {"message": msg, "account": acct, "packages": PACKAGE_PLANS, "action": "show_packages"}
    if any(w in text for w in ["ledger", "history", "usage", "spent"]):
        return {"message": "Here is your credit usage history.", "account": acct, "ledger": CREDIT_LEDGER.get(user_id, [])[-50:], "action": "show_ledger"}
    return {
        "message": f"Your current credit balance is {acct['credit_balance']} tokens on {acct['plan_name']}.",
        "account": acct,
        "packages": PACKAGE_PLANS,
        "action": "show_balance",
    }


def _dump_model(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)

# =============================================================================
# TEXT UTILITIES
# =============================================================================
_EMOJI = ["😊","🙂","😂","😄","👍","😃","🎉","✅","❌","🎓","📚","💡","•","→","►"]
_PLACEHOLDER_FRAGMENTS = [
    "अब इस अंश को सरल भाषा में समझते हैं",
    "इस अध्याय का विषय है",
    "suno dhyan se",
]

def clean_text(text: str) -> str:
    """Strip markdown, emoji, extra whitespace for TTS."""
    if not text:
        return ""
    for ch in _EMOJI + ["*","_","#","`","~","**","__"]:
        text = text.replace(ch, "")
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def is_placeholder(text: Optional[str]) -> bool:
    if not text or len(text.strip()) < 25:
        return True
    t = text.lower()
    return any(p.lower() in t for p in _PLACEHOLDER_FRAGMENTS)

def is_krutidev(text: Optional[str]) -> bool:
    """Detect Krutidev legacy font encoding (ASCII-faked Hindi)."""
    if not text or len(text) < 15:
        return True
    dev = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    asc = sum(1 for c in text if c.isascii() and c.isalpha())
    return asc > 8 and dev < 4

# =============================================================================
# DATABASE HELPERS  (all return empty gracefully when DB unavailable)
# =============================================================================
def db_chapter_meta(chapter_id: str) -> dict:
    sb = _get_sb()
    if not sb or not chapter_id:
        return {}
    try:
        r = (sb.table("syllabus_chapters")
               .select("id,chapter_title,chapter_code,book_name,author,board,class_level,subject,language")
