"""
Synthetic time-entry generator, v1 (100 rows).
Fictional corporate/transactional firm data for the billing-narrative capstone.

Output: synthetic_time_entries_v1.xlsx with 4 tabs
  - Time Entries      : submitted fields + gold KEY: answer-key columns
  - TK Matter History : each timekeeper's matters and 90-day hours (for matter resolution)
  - Matter Master     : CMID -> client/matter, codenames, client aliases
  - Notes             : how to use the file

Run: python generate_synthetic_v1.py   (requires openpyxl)
Fixed seed (210) so output is reproducible.
"""
import random, datetime, collections
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

OUTPUT = "synthetic_time_entries_v1.xlsx"
random.seed(210)

# ---------------- Reference data ----------------
M = {  # cmid: (client, matter)
 "1111.001":("Firm","Admin"),
 "1111.002":("Firm","Business Development"),
 "2201.003":("Harborlight Bio","Series A Financing"),
 "2956.001":("Solace Media","Licensing - Content Distribution"),
 "3012.001":("Halvorsen Industrial","General Corporate"),
 "3012.004":("Halvorsen Industrial","Acquisition of Brightline Pumps"),
 "4471.001":("Ridgewater Partners","General Fund Matters"),
 "4471.002":("Ridgewater Partners","Fund III Formation"),
 "4590.001":("Tidewater AI","General Corporate"),
 "4590.002":("Tidewater AI","Board Matters"),
 "5520.001":("Cobalt Ridge Capital","Add-On Acquisition (Project Lark)"),
 "5520.002":("Cobalt Ridge Capital","Portfolio Co. Sale (Project Wren)"),
 "6103.001":("Meridian Therapeutics","SEC Reporting"),
 "6103.002":("Meridian Therapeutics","Follow-On Offering"),
 "6103.003":("Meridian Therapeutics","Board & Governance"),
 "7215.003":("Northgate Bank","Credit Facility - Asterion Holdings"),
 "7215.004":("Northgate Bank","Credit Facility - Kestrel Foods"),
 "8340.001":("Pelican Row Holdings","Acquisition of 450 Harbor St."),
 "8340.002":("Pelican Row Holdings","Lease - 12 Mission Ave."),
 "9028.001":("Lanternfish Software","General Corporate"),
 "9028.002":("Lanternfish Software","Tax Structuring - Reorg"),
 "9028.003":("Lanternfish Software","Exec Comp / 409A"),
}
TK = {
 "TK101":("A. Chen","Partner"),
 "TK102":("R. Patel","Sr. Associate"),
 "TK103":("L. Moreno","Associate"),
 "TK104":("D. Okafor","Associate"),
 "TK105":("J. Whitfield","Associate"),
 "TK106":("S. Kim","Associate"),
 "TK107":("M. Alvarez","Paralegal"),
}
# timekeeper -> {cmid: hours billed in last 90 days}; used as sampling weights and resolution history
ASSIGN = {
 "TK101":{"3012.004":62.4,"5520.001":48.1,"5520.002":21.7,"4590.002":9.5,"1111.001":6.0,"1111.002":14.5},
 "TK102":{"3012.001":8.2,"3012.004":118.6,"5520.001":74.3,"1111.001":5.1},
 "TK103":{"4471.001":12.4,"4471.002":96.8,"5520.002":33.9,"1111.001":4.4},
 "TK104":{"6103.001":41.2,"6103.002":88.0,"6103.003":16.3,"1111.001":3.8},
 "TK105":{"7215.003":71.5,"7215.004":19.2,"8340.001":55.0,"8340.002":7.6,"1111.001":4.0},
 "TK106":{"9028.001":22.3,"9028.002":14.1,"9028.003":10.8,"2201.003":46.9,"4590.001":30.2,"2956.001":18.4,"1111.001":6.3,"1111.002":3.5},
 "TK107":{"2201.003":28.0,"6103.001":19.5,"8340.001":24.2,"3012.004":37.1,"1111.001":9.0},
}
# Narrative templates per matter: (text, hint). hint = distinctive keyword that identifies the matter, or None if generic/vague
N = {
 "1111.001":[("Attend to general admin matters.",None),("timekeeping","timekeeping"),("attnd to various administrative matters",None),("CLE - securities update webinar","CLE"),("Update matter list and time entrys for wk","timekeeping")],
 "1111.002":[("lunch w/ potential clent @ Kokkari","prospective client"),("Draft pitch for RFP; conf w marketing re same","pitch/RFP"),("Attend networking event - SF founders mixer","networking event")],
 "2201.003":[("attend to financng matters",None),("revise SPA and IRA per lead investor cmts; tcw client re board seat","SPA/IRA (financing docs)"),("Prep closing checklist for Series A","Series A"),("Rvw cap table and pro forma, flag option pool issue","pro forma cap table"),("assemble sig pckts for Series A closing","Series A"),("tcw K. Osei re lead investor side letter",None)],
 "2956.001":[("markup license agrmt; revise IP ownership & termination provs","license agreement"),("tcw client re distribution territories","distribution"),("Rvw counterparty cmts to content license","content license"),("attend to licensing matters",None)],
 "3012.001":[("Attend to misc corp matters",None),("update minute book and annual consents","annual consents"),("Rvw NDA for supplier; send redline","supplier NDA")],
 "3012.004":[("review and markup SPA; revise indemnification provisons, tcw buyer counsel re escrow","SPA indemnity/escrow"),("attend to diligence",None),("Rvw Brightline material contracts in data room, update DD memo","Brightline"),("draft disclosure schdules; tcw R. Patel re same","disclosure schedules"),("Attend TCW w seller counsel re purchase price adj. and working capital peg","working capital peg"),("prepare HSR filing checklist","HSR"),("Update DD tracker",None)],
 "4471.001":[("attend to fund matters",None),("Rvw capital call notice for Fund II","Fund II capital call"),("tcw GP re annual LP meeting logistics","LP meeting")],
 "4471.002":[("Draft side letter for LP (Calpine Pension); conf w/ A. Nguyen re MFN elections","side letter / MFN"),("Rvw LPA comments and circ to team","LPA"),("revise PPM risk factors","PPM"),("attend to fund formation matters","fund formation"),("Prepare subscription docs + investor questionnaire","subscription docs"),("tcw GP re Fund III first close timing","Fund III")],
 "4590.001":[("Attend to general corp matters",None),("Rvw option grants for Q3; prepare board consent","option grants"),("draft consulting agrmt for advisor","advisor agreement"),("tcw CEO re hiring questions",None)],
 "4590.002":[("Attend BOD mtg","board meeting"),("Attend board mtg; prepare minutes","board minutes"),("prep board deck legal update","board deck")],
 "5520.001":[("Attend to closing matters; draft disclosure schedules. prepare signature pckts, tcws w/ client and seller counsel re same","closing (block)"),("Rvw Project Lark LOI and comment","Project Lark"),("markup APA for add-on acq","APA add-on"),("attend to deal matters",None),("tcw lender counsel re acquisition financing for add-on","add-on financing")],
 "5520.002":[("Draft CIM legal section for Project Wren","Project Wren"),("rvw buyer markups to Wren SPA","Wren"),("coordinate VDR set up for sale process","sale process VDR"),("attend to sale process",None)],
 "6103.001":[("revise 8-K re officer departure, circulate to CFO for review","8-K"),("Rvw 10-Q draft MD&A","10-Q"),("prepare Section 16 filings - Form 4s","Form 4"),("attend to SEC matters",None)],
 "6103.002":[("Review S-3 draft and comment on risk factors; review comfort letter bring-down","S-3 / comfort letter"),("attend to offering matters",None),("tcw underwriters counsel re lock-up agrmts","underwriters / lock-up"),("Rvw underwriting agmt draft","underwriting agreement"),("DD call w/ mgmt and underwriters","underwriter DD call")],
 "6103.003":[("draft board minutes from 9/15 mtg","board minutes"),("Rvw committee charters, flag updates","committee charters"),("attend to governance matters",None)],
 "7215.003":[("review credit agrmt and guarantee; draft UCC finacing statements","UCC / guarantee"),("Email w/ borrower counsel",None),("rvw Asterion perfection cert","Asterion"),("attend to loan docs",None),("Revise closing checklist for credit facility, tcw bank re conditions precedent","conditions precedent")],
 "7215.004":[("Rvw Kestrel term sheet","Kestrel"),("draft commitment letter for Kestrel facility","Kestrel"),("attend to financing matters",None)],
 "8340.001":[("review title commitment and survey, prepare title objection letter; tcw lender re loan docs","title/survey"),("Revw estoppels","estoppels"),("rvw 450 Harbor PSA amendment","450 Harbor"),("attend to closing",None),("Prep closing binder","closing binder")],
 "8340.002":[("markup lease for 12 Mission; revise CAM provisions","12 Mission lease"),("Rvw tenant cmts to lease","lease comments"),("attend to lease matters","lease")],
 "9028.001":[("attend to general corp matters",None),("Rvw vendor MSA, send redline to client","vendor MSA"),("tcw founder re equity questions",None)],
 "9028.002":[("Analyze Section 368 reorg issues; memo to M. Farris re same","Sec. 368 reorg"),("attend to tax structuring",None),("rvw step plan for reorg","reorg step plan")],
 "9028.003":[("Rvw option grant docs for 409A compliance. tcw HR re same","409A"),("revise exec offer letter + severance terms","exec offer/severance"),("attend to comp matters",None)],
}
# Client name variants a timekeeper might type (aliases, abbreviations, misspellings)
ALIAS = {
 "Harborlight Bio":["Harborlight","Harborlite Bio","HLB"],
 "Solace Media":["Solace","Solace Media Inc"],
 "Halvorsen Industrial":["Halvorsen","Halverson","Halvorsen Ind."],
 "Ridgewater Partners":["Ridgewater","RWP","Ridgewtr"],
 "Tidewater AI":["Tidewater","Tidewatr AI","Tidewater.ai"],
 "Cobalt Ridge Capital":["Cobalt Ridge","CRC","Cobalt"],
 "Meridian Therapeutics":["Meridian","Meridian Tx","Meridan"],
 "Northgate Bank":["Northgate","NGB","Northgate Bnk"],
 "Pelican Row Holdings":["Pelican Row","Pelican","PRH"],
 "Lanternfish Software":["Lanternfish","Lanternfsh","LFS"],
 "Firm":["Firm","firm"],
}
CODENAME = {"5520.001":"Project Lark","5520.002":"Project Wren","3012.004":"Project Brightline","6103.002":"Project Ember","9028.002":"Project Tern"}

def cmid_typo(c):
    """Mis-format a CMID the way people actually type it."""
    a,b = c.split(".")
    return random.choice([a+b, a+"."+b.lstrip("0"), a+"-"+b, a+"."+b[1:], a+" "+b])

# Business days 9/14/2026 - 9/25/2026
dates=[]
d=datetime.date(2026,9,14)
while d<=datetime.date(2026,9,25):
    if d.weekday()<5: dates.append(d)
    d+=datetime.timedelta(days=1)

HOURS=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.4,1.5,1.6,1.8,2.0,2.2,2.5,2.8,3.1,3.4,4.0,4.8]

# ---------------- Input-type mix (100 rows) ----------------
plan = ["FULL"]*40+["CLIENT_ONLY"]*24+["NONE"]*14+["CODENAME"]*8+["CMID_ONLY"]*8+["WRONG_CMID"]*6
random.shuffle(plan)

rows=[]
for i,itype in enumerate(plan):
    # Pick timekeeper, then a matter weighted by their 90-day history; retry until it fits the input type
    for _ in range(200):
        tk=random.choice(list(TK))
        opts=list(ASSIGN[tk])
        w=[ASSIGN[tk][c] for c in opts]
        cm=random.choices(opts,weights=w)[0]
        client,matter=M[cm]
        text,hint=random.choice(N[cm])
        if itype=="CODENAME" and cm not in CODENAME: continue
        if itype in("CLIENT_ONLY","WRONG_CMID") and client=="Firm": continue
        if itype=="WRONG_CMID":
            others=[c for c in opts if c!=cm and not c.startswith("1111")]
            if not others or hint is None: continue
        break

    e_cmid,e_client,e_matter="","",""   # what the timekeeper "entered"
    basis=""                             # why the true matter is resolvable (answer key)
    if itype=="FULL":
        e_cmid,e_client,e_matter=cm,client,matter; basis="Entered CMID"
    elif itype=="CMID_ONLY":
        e_cmid=cmid_typo(cm); basis=f"Normalize CMID format ({e_cmid} -> {cm})"
    elif itype=="WRONG_CMID":
        wrong=random.choice([c for c in ASSIGN[tk] if c!=cm and not c.startswith("1111")])
        e_cmid=wrong; e_client,e_matter=M[wrong]
        basis=f"Narrative ('{hint}') contradicts entered CMID {wrong}; timekeeper active on {cm}"
    elif itype=="CODENAME":
        e_matter=CODENAME[cm]
        basis=f"Codename '{CODENAME[cm]}' maps to {cm}"
    elif itype=="CLIENT_ONLY":
        e_client=random.choice(ALIAS[client])
        cands=[c for c in ASSIGN[tk] if M[c][0]==client]
        if len(cands)==1: basis=f"Only active {client} matter for {TK[tk][0]}"
        elif hint: basis=f"Narrative keyword ('{hint}') among {len(cands)} {client} matters"
        else:
            top=max(cands,key=lambda c:ASSIGN[tk][c])
            basis="AMBIGUOUS - "+f"{len(cands)} candidate matters; highest 90-day hours = {top}"
    elif itype=="NONE":
        cands=[c for c in ASSIGN[tk]]
        if hint: basis=f"Narrative keyword ('{hint}') + {TK[tk][0]} active matters"
        else:
            basis=f"AMBIGUOUS - no client/matter signal; {len(cands)} active matters for {TK[tk][0]}"

    date=random.choice(dates)
    action = "BOD" if cm=="4590.002" or "board mtg" in text.lower() or "bod" in text.lower() else "DEF"
    rows.append([f"T-{1001+i:06d}","WIP",action,
                 tk,TK[tk][0],date.strftime("%m/%d/%Y"),e_cmid,e_client,e_matter,random.choice(HOURS),text,
                 cm,client,matter,itype,basis])

# Sort by date, then timekeeper; renumber IDs
rows.sort(key=lambda r:(datetime.datetime.strptime(r[5],"%m/%d/%Y"),r[3]))
for i,r in enumerate(rows): r[0]=f"T-{1001+i:06d}"

# ---------------- Write workbook ----------------
wb=Workbook()
ws=wb.active; ws.title="Time Entries"
hdr=["Time ID","Status","Action","TK ID","Timekeeper","Date","CMID","Client","Matter","Hours","Narrative",
     "KEY: True CMID","KEY: True Client","KEY: True Matter","KEY: Input Type","KEY: Resolution Basis"]
ws.append(hdr)
for r in rows: ws.append(r)

thin=Side(style="thin",color="BFBFBF")
hfill=PatternFill("solid",fgColor="1F3864"); kfill=PatternFill("solid",fgColor="7F6000")
kbody=PatternFill("solid",fgColor="FFF2CC")
def style(ws,keycols=()):
    for c in ws[1]:
        c.font=Font(name="Arial",bold=True,color="FFFFFF",size=10)
        c.fill=kfill if c.column in keycols else hfill
        c.alignment=Alignment(wrap_text=True,vertical="center")
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font=Font(name="Arial",size=10)
            c.alignment=Alignment(wrap_text=True,vertical="top")
            c.border=Border(bottom=thin)
            if c.column in keycols: c.fill=kbody
    ws.freeze_panes="B2"
    ws.auto_filter.ref=ws.dimensions
style(ws,keycols=range(12,17))
for col,wd in zip("ABCDEFGHIJKLMNOP",[11,7,7,7,12,11,11,18,24,7,60,12,20,28,13,50]):
    ws.column_dimensions[col].width=wd
for r in range(2,ws.max_row+1): ws.cell(r,10).number_format="0.0"

ws2=wb.create_sheet("TK Matter History")
ws2.append(["TK ID","Timekeeper","Role","CMID","Client","Matter","Hours (last 90 days)"])
for tk,d in ASSIGN.items():
    for c,h in sorted(d.items()):
        ws2.append([tk,TK[tk][0],TK[tk][1],c,M[c][0],M[c][1],h])
style(ws2)
for col,wd in zip("ABCDEFG",[8,12,13,10,22,34,12]): ws2.column_dimensions[col].width=wd

ws3=wb.create_sheet("Matter Master")
ws3.append(["CMID","Client","Matter","Codename","Known Aliases"])
for c,(cl,mt) in sorted(M.items()):
    ws3.append([c,cl,mt,CODENAME.get(c,""),", ".join(ALIAS.get(cl,[]))])
style(ws3)
for col,wd in zip("ABCDE",[10,22,34,18,40]): ws3.column_dimensions[col].width=wd

ws4=wb.create_sheet("Notes")
notes=["Synthetic data - all clients, people, and matters are fictional.",
"Time Entries: blue-header columns are what the timekeeper submitted (may be blank, misspelled, or wrong).",
"Gold 'KEY:' columns are the answer key - hide/drop them before feeding to the model.",
"Input types: FULL (all fields), CMID_ONLY (CMID, sometimes mis-formatted), CLIENT_ONLY (client name/alias only), CODENAME (project codename only), NONE (narrative only), WRONG_CMID (CMID contradicts narrative).",
"TK Matter History: which matters each timekeeper has billed and how much in the last 90 days - use for resolving entries with missing CMID.",
"AMBIGUOUS in Resolution Basis = not resolvable from fields + narrative alone; should route to human review."]
for n in notes: ws4.append([n])
ws4.column_dimensions["A"].width=120
for row in ws4.iter_rows():
    for c in row: c.font=Font(name="Arial",size=10); c.alignment=Alignment(wrap_text=True)

wb.save(OUTPUT)
print(f"Wrote {OUTPUT}: {len(rows)} rows")
print(collections.Counter(r[14] for r in rows))
