from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx

from .store import RetrievedChunk


@dataclass(frozen=True)
class OsrsboxDoc:
    url: str
    title: str
    text: str


def _norm_ws(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _chunk_text(text: str, *, max_chars: int = 900, overlap: int = 120) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not text:
        return []

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        parts.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return parts


def _tokenize_query(q: str) -> list[str]:
    q = (q or "").lower()
    toks = re.findall(r"[a-z0-9_']+", q)
    # keep short list of meaningful tokens
    return [t for t in toks if t and t not in {"the", "a", "an", "of", "in", "on", "for", "osrs"}]


async def _fetch_json(url: str, *, timeout_s: float = 20.0) -> Any:
    headers = {"User-Agent": "OSRS-AI-Wiki-Chat/0.1"}
    async with httpx.AsyncClient(timeout=timeout_s, headers=headers, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()


def _best_name_match_score(name: str, query: str) -> int:
    n = (name or "").lower()
    toks = _tokenize_query(query)
    if not toks:
        return 0
    score = sum(1 for t in toks if t in n)
    # prefer exact-ish matches
    if _norm_ws(name).lower() == _norm_ws(query).lower():
        score += 5
    return score


async def osrsbox_fetch_items_by_query(
    query: str,
    *,
    base_url: str = "https://www.osrsbox.com/osrsbox-db/",
    limit: int = 5,
) -> list[OsrsboxDoc]:
    """Fetch a few item JSON docs matching a query via items-summary + per-item fetch."""

    base = (base_url or "").strip()
    if base and not base.endswith("/"):
        base += "/"

    summary_url = base + "items-summary.json"
    summary = await _fetch_json(summary_url)

    # Summary is typically a list of {"id": int, "name": str}
    items = []
    if isinstance(summary, list):
        items = summary
    elif isinstance(summary, dict):
        # Some OSRSBox summaries are dicts keyed by a numeric string.
        wrapped = summary.get("items")
        if isinstance(wrapped, list):
            items = wrapped
        else:
            vals = list(summary.values())
            if vals and all(isinstance(v, dict) for v in vals):
                items = vals

    scored: list[tuple[int, dict]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "")
        if not name:
            continue
        scored.append((_best_name_match_score(name, query), it))

    scored.sort(key=lambda t: t[0], reverse=True)
    picked = [it for s, it in scored if s > 0][: max(1, int(limit))]

    out: list[OsrsboxDoc] = []
    for it in picked:
        iid = it.get("id")
        if iid is None:
            continue
        try:
            iid_int = int(iid)
        except Exception:
            continue

        url = base + f"items-json/{iid_int}.json"
        data: Any | None = None
        try:
            data = await _fetch_json(url)
        except httpx.HTTPStatusError as e:
            status = getattr(e.response, "status_code", None)
            if status != 404:
                continue

            # Fallback: the GitHub raw files sometimes exist even when the osrsbox.com
            # static mirror doesn't.
            gh_url = f"https://raw.githubusercontent.com/osrsbox/osrsbox-db/master/docs/items-json/{iid_int}.json"
            try:
                data = await _fetch_json(gh_url)
                url = gh_url
            except Exception:
                continue

        doc = osrsbox_item_json_to_doc(data, url=url) if data is not None else None
        if doc:
            out.append(doc)

    return out


async def osrsbox_fetch_monsters_by_query(
    query: str,
    *,
    base_url: str = "https://www.osrsbox.com/osrsbox-db/",
    limit: int = 3,
) -> list[OsrsboxDoc]:
    """Fetch a few monster JSON docs matching a query.

    Uses npcs-summary.json to locate candidate IDs, then fetches per-monster JSON.
    This avoids downloading monsters-complete.json (large).
    """

    base = (base_url or "").strip()
    if base and not base.endswith("/"):
        base += "/"

    summary_url = base + "npcs-summary.json"
    summary = await _fetch_json(summary_url)

    entries = []
    if isinstance(summary, list):
        entries = summary
    elif isinstance(summary, dict):
        wrapped = summary.get("npcs")
        if isinstance(wrapped, list):
            entries = wrapped
        else:
            vals = list(summary.values())
            if vals and all(isinstance(v, dict) for v in vals):
                entries = vals

    scored: list[tuple[int, dict]] = []
    for it in entries:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "")
        if not name:
            continue
        scored.append((_best_name_match_score(name, query), it))

    scored.sort(key=lambda t: t[0], reverse=True)
    picked = [it for s, it in scored if s > 0][: max(5, int(limit) * 5)]

    out: list[OsrsboxDoc] = []
    for it in picked:
        mid = it.get("id")
        if mid is None:
            continue
        try:
            mid_int = int(str(mid))
        except Exception:
            continue

        url = base + f"monsters-json/{mid_int}.json"
        data: Any | None = None
        try:
            data = await _fetch_json(url)
        except httpx.HTTPStatusError as e:
            status = getattr(e.response, "status_code", None)
            if status != 404:
                continue

            gh_url = f"https://raw.githubusercontent.com/osrsbox/osrsbox-db/master/docs/monsters-json/{mid_int}.json"
            try:
                data = await _fetch_json(gh_url)
                url = gh_url
            except Exception:
                continue
        except Exception:
            continue

        doc = osrsbox_monster_json_to_doc(data, url=url)
        if doc:
            out.append(doc)
        if len(out) >= max(1, int(limit)):
            break

    return out


def _fmt_reqs(reqs: Any) -> str:
    if not isinstance(reqs, dict) or not reqs:
        return ""
    parts = []
    for k, v in reqs.items():
        try:
            lvl = int(v)
        except Exception:
            continue
        parts.append(f"{k} {lvl}")
    return ", ".join(parts)


def osrsbox_item_json_to_doc(data: Any, *, url: str) -> OsrsboxDoc | None:
    if not isinstance(data, dict):
        return None

    item_id = data.get("id")
    name = str(data.get("name") or "").strip()
    if not name:
        return None

    item_id_int: int | None
    if item_id is None:
        item_id_int = None
    else:
        try:
            item_id_int = int(str(item_id))
        except Exception:
            item_id_int = None

    examine = _norm_ws(str(data.get("examine") or ""))
    members = data.get("members")
    tradeable = data.get("tradeable")
    stackable = data.get("stackable")
    equipable = data.get("equipable_by_player")
    cost = data.get("cost")
    highalch = data.get("highalch")
    lowalch = data.get("lowalch")
    weight = data.get("weight")
    buy_limit = data.get("buy_limit")

    equipment = data.get("equipment") if isinstance(data.get("equipment"), dict) else None
    weapon = data.get("weapon") if isinstance(data.get("weapon"), dict) else None

    wiki_url = str(data.get("wiki_url") or "").strip()

    lines: list[str] = []
    lines.append(f"OSRSBox Item: {name}" + (f" (id={item_id_int})" if item_id_int is not None else ""))
    if examine:
        lines.append(f"Examine: {examine}")

    meta_bits: list[str] = []
    if isinstance(members, bool):
        meta_bits.append("members" if members else "f2p")
    if isinstance(tradeable, bool):
        meta_bits.append("tradeable" if tradeable else "untradeable")
    if isinstance(stackable, bool) and stackable:
        meta_bits.append("stackable")
    if isinstance(equipable, bool) and equipable:
        meta_bits.append("equipable")
    if meta_bits:
        lines.append("Flags: " + ", ".join(meta_bits))

    def _add_num(label: str, v: Any) -> None:
        if v is None:
            return
        try:
            if isinstance(v, bool):
                return
            if isinstance(v, float):
                lines.append(f"{label}: {v:.3f}")
            else:
                lines.append(f"{label}: {int(v)}")
        except Exception:
            pass

    _add_num("Shop cost", cost)
    _add_num("High alch", highalch)
    _add_num("Low alch", lowalch)
    if weight is not None:
        try:
            lines.append(f"Weight: {float(weight):.3f} kg")
        except Exception:
            pass
    _add_num("GE buy limit", buy_limit)

    if equipment:
        reqs = _fmt_reqs(equipment.get("requirements"))
        slot = equipment.get("slot")
        if slot:
            lines.append(f"Equipment slot: {slot}")
        if reqs:
            lines.append(f"Requirements: {reqs}")

        # Keep this compact; list core bonuses.
        keys = [
            "attack_stab",
            "attack_slash",
            "attack_crush",
            "attack_magic",
            "attack_ranged",
            "defence_stab",
            "defence_slash",
            "defence_crush",
            "defence_magic",
            "defence_ranged",
            "melee_strength",
            "ranged_strength",
            "magic_damage",
            "prayer",
        ]
        bonuses = []
        for k in keys:
            v = equipment.get(k)
            if v is None:
                continue
            try:
                bonuses.append(f"{k}={int(v)}")
            except Exception:
                continue
        if bonuses:
            lines.append("Bonuses: " + ", ".join(bonuses))

    if weapon:
        spd = weapon.get("attack_speed")
        wtype = weapon.get("weapon_type")
        if wtype:
            lines.append(f"Weapon type: {wtype}")
        if spd is not None:
            try:
                lines.append(f"Attack speed (ticks): {int(spd)}")
            except Exception:
                pass

    if wiki_url:
        lines.append(f"OSRS Wiki: {wiki_url}")

    title = f"OSRSBox Item: {name}" + (f" ({item_id_int})" if item_id_int is not None else "")
    text = "\n".join(lines).strip()
    return OsrsboxDoc(url=url, title=title, text=text)


def osrsbox_monster_json_to_doc(data: Any, *, url: str) -> OsrsboxDoc | None:
    if not isinstance(data, dict):
        return None

    mid = data.get("id")
    name = str(data.get("name") or "").strip()
    if not name:
        return None

    mid_int: int | None
    if mid is None:
        mid_int = None
    else:
        try:
            mid_int = int(str(mid))
        except Exception:
            mid_int = None

    lines: list[str] = []
    lines.append(f"OSRSBox Monster: {name}" + (f" (id={mid_int})" if mid_int is not None else ""))

    for k in [
        "combat_level",
        "hitpoints",
        "max_hit",
        "attack_speed",
        "attack_level",
        "strength_level",
        "defence_level",
        "magic_level",
        "ranged_level",
    ]:
        v = data.get(k)
        if v is None:
            continue
        try:
            lines.append(f"{k.replace('_', ' ').title()}: {int(v)}")
        except Exception:
            continue

    atk_type = data.get("attack_type")
    if isinstance(atk_type, list) and atk_type:
        lines.append("Attack type: " + ", ".join(str(x) for x in atk_type[:4]))

    slayer_level = data.get("slayer_level")
    if slayer_level is not None:
        try:
            lines.append(f"Slayer level required: {int(slayer_level)}")
        except Exception:
            pass

    examine = _norm_ws(str(data.get("examine") or ""))
    if examine:
        lines.append(f"Examine: {examine}")

    wiki_url = str(data.get("wiki_url") or "").strip()
    if wiki_url:
        lines.append(f"OSRS Wiki: {wiki_url}")

    # Notable drops (keep short): choose a few rarest drops if present.
    drops = data.get("drops")
    if isinstance(drops, list) and drops:
        parsed = []
        for d in drops:
            if not isinstance(d, dict):
                continue
            n = d.get("name")
            r = d.get("rarity")
            if not n or r is None:
                continue
            try:
                rf = float(r)
            except Exception:
                continue
            parsed.append((rf, str(n)))
        parsed.sort(key=lambda t: t[0])
        if parsed:
            rare_names = [n for _rf, n in parsed[:6]]
            lines.append("Notable rare drops: " + ", ".join(rare_names))

    title = f"OSRSBox Monster: {name}" + (f" ({mid_int})" if mid_int is not None else "")
    text = "\n".join(lines).strip()
    return OsrsboxDoc(url=url, title=title, text=text)


def doc_to_chunks(doc: OsrsboxDoc, *, max_chunks: int = 3) -> list[RetrievedChunk]:
    chunks = _chunk_text(doc.text, max_chars=900, overlap=120)
    out: list[RetrievedChunk] = []
    for ch in chunks[: max(1, int(max_chunks))]:
        out.append(RetrievedChunk(text=ch, url=doc.url, title=doc.title))
    return out
