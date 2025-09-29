# streamlit_app.py
# -------------------------------------------------------------
# KSV Slaapnacht – Teamindeler
# Upload hier het CSV-bestand dat je uit Google Forms/Spreadsheets exporteert.
# De app maakt vaste "basisteams" (~7–8 kids) die voor avond- en ochtendspellen
# gelijk blijven. Als iemand alleen 's avonds of 's ochtends meedoet, blijft het
# team bestaan met 1 persoon minder/meer in die sessie.
# -------------------------------------------------------------

import io
import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="KSV Slaapnacht – Teamindeler", page_icon="🤾", layout="wide")

st.title("🤾 KSV Slaapnacht – Teamindeler")
st.markdown(
    """
Deze tool maakt vaste teams (~7–8) die **voor avond- en ochtendspellen gelijk** zijn. 
Bij afwezigheid toont de app per sessie de overgebleven spelers.

**CSV-eisen (Google Forms):**
- Voornaam
- Achternaam
- Vorig team (keuzes: C1, C2, D1, D2, D3, E1, E2, E3, F1, F2)
- Ik ben erbij (meerkeuze): Avondspellen (19.00–21.00), Slapen, Ochtendspellen (09.00–12.00)
- Opmerkingen
- Contactpersoon (voor- en achternaam)
- Mobiel contactpersoon

> Tip: Exporteer vanuit Google Sheets → *Bestand* → *Downloaden* → **CSV**.
"""
)

# ------------------------- Helpers -------------------------

def _slugify(text: str) -> str:
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return text.strip().lower()

CATEGORY_ORDER = ["c", "d", "e", "f"]  # C is het oudst, F het jongst

@dataclass
class Player:
    first: str
    last: str
    raw_team: str  # bijv. "D2"
    category: str  # c/d/e/f
    evening: bool
    morning: bool
    remarks: str
    contact_name: str
    contact_phone: str

    @property
    def full(self) -> str:
        return f"{self.first} {self.last}".strip()


def parse_dataframe(df: pd.DataFrame) -> List[Player]:
    # Probeer kolomnamen tolerant te mappen
    cols_map = {
        "voornaam": None,
        "achternaam": None,
        "vorig team": None,
        "ik ben erbij": None,
        "opmerkingen": None,
        "contactpersoon": None,
        "mobiel": None,
    }

    canon = { _slugify(c): c for c in df.columns }

    def find_col(candidates: List[str]) -> str:
        for cand in candidates:
            key = _slugify(cand)
            # Zoeken op beginsels
            for k,v in canon.items():
                if key in k:
                    return v
        return None

    cols_map["voornaam"] = find_col(["voornaam","first name","voor naam"])
    cols_map["achternaam"] = find_col(["achternaam","last name","achter naam"])
    cols_map["vorig team"] = find_col(["vorig team","welk team","team vorig seizoen","team"])
    cols_map["ik ben erbij"] = find_col(["ik ben erbij","aanwezig","attendance","ik doe mee"])
    cols_map["opmerkingen"] = find_col(["opmerkingen","remarks","allergie","notities"])
    cols_map["contactpersoon"] = find_col(["contactpersoon","contact naam","ouder naam"])
    cols_map["mobiel"] = find_col(["mobiel","telefoon","gsm","phone"])

    missing = [k for k,v in cols_map.items() if v is None and k in ("voornaam","achternaam","vorig team","ik ben erbij")]
    if missing:
        st.error(f"Ontbrekende verplichte kolommen in CSV: {', '.join(missing)}")
        return []

    players: List[Player] = []

    for _, row in df.iterrows():
        first = str(row.get(cols_map["voornaam"], "")).strip()
        last = str(row.get(cols_map["achternaam"], "")).strip()
        raw_team = str(row.get(cols_map["vorig team"], "")).strip()
        remarks = str(row.get(cols_map["opmerkingen"], "")).strip()
        contact_name = str(row.get(cols_map["contactpersoon"], "")).strip()
        contact_phone = str(row.get(cols_map["mobiel"], "")).strip()

        # Attendance kan in Google Forms als komma-gescheiden string komen of als booleans in meerdere kolommen
        att_raw = row.get(cols_map["ik ben erbij"], "")
        if isinstance(att_raw, str):
            tokens = [_slugify(x) for x in att_raw.replace(";", ",").split(",")]
            evening = any("avond" in t for t in tokens)
            morning = any("ochtend" in t for t in tokens)
        else:
            # fallback
            evening = bool(att_raw)
            morning = bool(att_raw)

        # Categoriseer op eerste letter van het team (c/d/e/f)
        cat = _slugify(raw_team)[:1]
        cat = cat if cat in {"c","d","e","f"} else ""

        players.append(Player(
            first=first,
            last=last,
            raw_team=raw_team.upper(),
            category=cat,
            evening=evening,
            morning=morning,
            remarks=remarks,
            contact_name=contact_name,
            contact_phone=contact_phone,
        ))

    # Filter lege namen
    players = [p for p in players if p.first or p.last]
    return players


def compute_num_teams(n: int) -> int:
    if n <= 8:
        return 1
    # Doelgemiddelde 7.5 per team; kies een N dat teamgroottes 7–8 oplevert
    cand = max(1, round(n / 7.5))
    # Zorg dat we minimaal 7 en max 8 per team *ongeveer* halen
    while cand > 1 and math.ceil(n / cand) > 8:
        cand += 1
    while math.floor(n / cand) < 7:
        cand -= 1
        if cand <= 1:
            return 1
    return cand


def build_base_teams(players: List[Player]) -> List[List[Player]]:
    # Basisteams op basis van iedereen die A) avond of B) ochtend speelt
    pool = [p for p in players if p.evening or p.morning]
    n = len(pool)
    if n == 0:
        return []

    T = compute_num_teams(n)
    teams: List[List[Player]] = [[] for _ in range(T)]

    # Groepeer per leeftijdscategorie en deel serpentine uit om te balanceren
    by_cat: Dict[str, List[Player]] = defaultdict(list)
    for p in pool:
        by_cat[p.category].append(p)

    for cat in CATEGORY_ORDER:
        group = by_cat.get(cat, [])
        # Sorteer op achternaam voor deterministische output
        group.sort(key=lambda x: (x.last.lower(), x.first.lower()))
        # Serpentine distributie: 0..T-1, T-1..0, etc.
        idx = 0
        step = 1
        for player in group:
            teams[idx].append(player)
            idx += step
            if idx == T:
                idx = T-1
                step = -1
            elif idx == -1:
                idx = 0
                step = 1

    # Fine-tune: balanceer teamgroottes richting 7–8 door verschuiven van overvolle teams
    changed = True
    # Max 10 iteraties om te voorkomen dat we blijven pendelen
    iters = 0
    while changed and iters < 10:
        iters += 1
        sizes = [len(t) for t in teams]
        max_i = max(range(T), key=lambda i: sizes[i])
        min_i = min(range(T), key=lambda i: sizes[i])
        changed = False
        if sizes[max_i] - sizes[min_i] > 1:
            # verplaats een speler van max_i naar min_i, bij voorkeur iemand uit een cat die oververtegenwoordigd is in max_i
            # kies jongste categorie eerst om oudere cat beter te spreiden
            for cat in reversed(CATEGORY_ORDER):
                cand_idx = next((k for k,p in enumerate(teams[max_i]) if p.category == cat), None)
                if cand_idx is not None:
                    player = teams[max_i].pop(cand_idx)
                    teams[min_i].append(player)
                    changed = True
                    break

    # Sorteer namen per team
    for t in teams:
        t.sort(key=lambda p: (CATEGORY_ORDER.index(p.category) if p.category in CATEGORY_ORDER else 99, p.last.lower(), p.first.lower()))

    return teams


def session_view(teams: List[List[Player]], when: str) -> List[List[Player]]:
    assert when in ("evening","morning")
    def is_in(p: Player) -> bool:
        return p.evening if when == "evening" else p.morning
    return [[p for p in team if is_in(p)] for team in teams]


def teams_to_dataframe(teams: List[List[Player]]) -> pd.DataFrame:
    rows = []
    for i, team in enumerate(teams, start=1):
        for p in team:
            rows.append({
                "Team": f"Team {i}",
                "Naam": p.full,
                "Categorie": p.category.upper() if p.category else "",
                "Vorig team": p.raw_team,
                "Contact": p.contact_name,
                "Mobiel": p.contact_phone,
                "Opmerkingen": p.remarks,
            })
    return pd.DataFrame(rows)


def download_csv(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )

# ------------------------- UI -------------------------
left, right = st.columns([1,1])

with left:
    uploaded = st.file_uploader("Upload CSV uit Google Forms/Sheets", type=["csv"]) 

    sample = st.toggle("Laad voorbeelddata", value=False, help="Handig om de werking te testen.")
    if sample:
        sample_csv = io.StringIO(
            """Voornaam,Achternaam,Vorig team,Ik ben erbij,Opmerkingen,Contactpersoon,Mobiel\n"
            "Sam,Visser,D2,Avondspellen,notenallergie,Oma Visser,0612345678\n"
            "Mia,de Jong,E1,Avondspellen; Ochtendspellen,,Jan de Jong,0611111111\n"
            "Noah,Peters,C1,Ochtendspellen,,An Peters,0622222222\n"
            "Liam,Bakker,D3,Avondspellen; Slapen; Ochtendspellen,,B Bakker,0633333333\n"
            "Sophie,Meijer,E2,Avondspellen,,K Meijer,0644444444\n"
            "Lucas,de Vries,F1,Ochtendspellen,,P de Vries,0655555555\n"
            "Emma,Smits,D1,Avondspellen; Ochtendspellen,,R Smits,0666666666\n"
            "Sara,Boers,E3,Avondspellen; Ochtendspellen,,C Boers,0677777777\n"
            "Adam,van Leeuwen,C2,Avondspellen; Ochtendspellen,,T van Leeuwen,0688888888\n"
            "Eva,Post,F2,Avondspellen,,J Post,0699999999\n"
        )
        uploaded = sample_csv

with right:
    st.markdown("### Instellingen")
    keep_order = st.toggle("Team volgorde fixeren (deterministisch)", value=True)


if uploaded is None:
    st.info("Upload een CSV of zet 'Laad voorbeelddata' aan.")
    st.stop()

# Lees CSV
try:
    df = pd.read_csv(uploaded)
except Exception:
    uploaded.seek(0)
    df = pd.read_csv(uploaded, sep=';')

players = parse_dataframe(df)
if not players:
    st.stop()

# Basis teams
base_teams = build_base_teams(players)
if not base_teams:
    st.warning("Geen deelnemers met avond of ochtend geselecteerd.")
    st.stop()

# Sessie-views
evening_teams = session_view(base_teams, "evening")
morning_teams = session_view(base_teams, "morning")

# ------------------------- Output -------------------------

tab1, tab2, tab3 = st.tabs(["Basisteams", "Avondspellen", "Ochtendspellen"])

with tab1:
    st.subheader("Basisteams (voor beide sessies gelijk)")
    for i, team in enumerate(base_teams, start=1):
        st.markdown(f"#### Team {i} – {len(team)} spelers")
        st.dataframe(pd.DataFrame({
            "Naam": [p.full for p in team],
            "Cat": [p.category.upper() for p in team],
            "Vorig team": [p.raw_team for p in team],
            "Opmerkingen": [p.remarks for p in team],
        }))
    df_base = teams_to_dataframe(base_teams)
    download_csv(df_base, "basisteams.csv", "Download basisteams (CSV)")

with tab2:
    st.subheader("Teamindeling – Avondspellen (19.00–21.00)")
    for i, team in enumerate(evening_teams, start=1):
        st.markdown(f"#### Team {i} – {len(team)} spelers")
        st.dataframe(pd.DataFrame({
            "Naam": [p.full for p in team],
            "Cat": [p.category.upper() for p in team],
            "Vorig team": [p.raw_team for p in team],
        }))
    df_evening = teams_to_dataframe(evening_teams)
    download_csv(df_evening, "teams_avond.csv", "Download avond-teams (CSV)")

with tab3:
    st.subheader("Teamindeling – Ochtendspellen (09.00–12.00)")
    for i, team in enumerate(morning_teams, start=1):
        st.markdown(f"#### Team {i} – {len(team)} spelers")
        st.dataframe(pd.DataFrame({
            "Naam": [p.full for p in team],
            "Cat": [p.category.upper() for p in team],
            "Vorig team": [p.raw_team for p in team],
        }))
    df_morning = teams_to_dataframe(morning_teams)
    download_csv(df_morning, "teams_ochtend.csv", "Download ochtend-teams (CSV)")

st.divider()
with st.expander("📋 Instructies – Google Form opzetten"):
    st.markdown(
        """
# **Vragen in Google Forms**
# 1. *Voornaam* (korte tekst)
# 2. *Achternaam* (korte tekst)
# 3. *Welk team heb je vorig seizoen gezeten?* (meerkeuze, exact: C1, C2, D1, D2, D3, E1, E2, E3, F1, F2)
# 4. *Ik ben erbij* (meerkeuze-vakjes): **Avondspellen (19.00–21.00)**, **Slapen**, **Ochtendspellen (09.00–12.00)**
# 5. *Opmerkingen (bijv. allergieën)* (lange tekst)
# 6. *Contactpersoon (voor- en achternaam)* (korte tekst)
# 7. *Mobiel contactpersoon* (korte tekst)

# **Workflow**
# - Deel of embed de Forms-link op de KSV-website.
# - Open de gekoppelde Google Sheet → *Bestand* → *Downloaden* → **CSV**.
# # - Upload de CSV hier om teams te genereren en download de indelingen per sessie.
# """
     )

st.success("Klaar! Basisteams blijven gelijk; per sessie tonen we alleen de aanwezigen. Categorieën worden zo gelijk mogelijk verdeeld.")

