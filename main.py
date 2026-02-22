"""
XFA PDF Renderer v3
-------------------
Usage:  python render_xfa.py <input.pdf> [output.pdf]
"""

import sys, os, re

try:
    import pikepdf
except ImportError:
    sys.exit("pip install pikepdf")
try:
    from lxml import etree
except ImportError:
    sys.exit("pip install lxml")
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    sys.exit("pip install reportlab")


# ── units ────────────────────────────────────────────────────────────────────
_U = re.compile(r'^([\-\d.]+)(mm|in|pt|cm)?$')
def to_pt(s):
    if not s: return 0.0
    m = _U.match(str(s).strip())
    if not m: return 0.0
    n, u = float(m.group(1)), m.group(2) or 'pt'
    return n * {'mm':2.8346,'cm':28.346,'in':72.0,'pt':1.0}[u]


# ── chunk extraction  ─────────────────────────────────────────────────────────
def _deref(obj, pdf):
    """Resolve a pikepdf indirect object to its actual value."""
    # pikepdf stores indirect refs as pikepdf.Object; check with is_indirect
    try:
        if obj.is_indirect:
            return pdf.make_indirect(obj)   # already resolved by pikepdf
    except Exception:
        pass
    # older API: objgen-based lookup
    try:
        og = obj.objgen
        if og[1] != 0 or og[0] != 0:
            return pdf.get_object(og)
    except Exception:
        pass
    return obj


def _read(stream_obj, pdf):
    """Read bytes from a (possibly indirect) stream object."""
    obj = _deref(stream_obj, pdf)
    if not isinstance(obj, pikepdf.Stream):
        return None
    for fn in ('read_bytes', 'read_raw_bytes'):
        try:
            return bytes(getattr(obj, fn)())
        except Exception:
            pass
    return None


def extract_xfa_chunks(path):
    pdf = pikepdf.open(path)

    # ── /AcroForm ───────────────────────────────────────────────────────────
    # Use pdf.Root which is always a proper Dictionary
    root_dict = pdf.Root

    # pikepdf Dictionaries support "in" and [] natively for /Name keys
    acroform_obj = root_dict.get("/AcroForm")
    if acroform_obj is None:
        raise ValueError("/AcroForm not found")

    # Make sure we have the actual dict (could be indirect)
    acroform_obj = _deref(acroform_obj, pdf)

    # If it's still not a Dictionary we cannot proceed
    if not isinstance(acroform_obj, (pikepdf.Dictionary, pikepdf.Stream)):
        raise ValueError(f"/AcroForm resolved to unexpected type: {type(acroform_obj)}")

    # ── /XFA ────────────────────────────────────────────────────────────────
    xfa_obj = acroform_obj.get("/XFA")
    if xfa_obj is None:
        raise ValueError("/XFA not found in /AcroForm")

    xfa_obj = _deref(xfa_obj, pdf)

    chunks = {}

    # Case 1 – single XFA stream
    if isinstance(xfa_obj, pikepdf.Stream):
        raw = _read(xfa_obj, pdf)
        if raw:
            chunks["xfa"] = raw
        return chunks

    # Case 2 – array  [/name <stream> /name <stream> …]
    if isinstance(xfa_obj, pikepdf.Array):
        i = 0
        while i < len(xfa_obj):
            item = _deref(xfa_obj[i], pdf)

            # Determine name
            name = None
            if isinstance(item, pikepdf.Name):
                name = str(item).lstrip("/")
            elif isinstance(item, pikepdf.String):
                name = str(item)

            if name is not None and i + 1 < len(xfa_obj):
                raw = _read(xfa_obj[i + 1], pdf)
                if raw:
                    chunks[name] = raw
                i += 2
            elif isinstance(item, pikepdf.Stream):
                raw = _read(item, pdf)
                if raw:
                    chunks[f"chunk_{i}"] = raw
                i += 1
            else:
                i += 1
        return chunks

    raise ValueError(f"Unexpected /XFA type: {type(xfa_obj).__name__}")


# ── XML helpers ───────────────────────────────────────────────────────────────
def parse_xml(raw):
    try:
        return etree.fromstring(raw)
    except Exception:
        return None

def local(el):
    t = el.tag
    if not isinstance(t, str):          # skip PIs, comments, etc.
        return ""
    return t.split("}")[-1] if "}" in t else t

def find1(el, *names):
    for c in el:
        lt = local(c)
        if lt and lt in names:
            return c
    return None

def iterall(el, name):
    for c in el.iter():
        if local(c) == name:
            yield c

def text_of(value_el):
    for c in value_el.iter():
        lt = local(c)
        if lt in ("text","integer","date","decimal"):
            return (c.text or "").strip()
        if lt in ("body","p"):
            return " ".join((n.text or "").strip() for n in c.iter() if n.text and n.text.strip())
    return ""


# ── dataset flattening ────────────────────────────────────────────────────────
def flatten(root_el):
    out = {}
    # Drill into <data>
    data_el = next((c for c in root_el.iter() if local(c) == "data"), root_el)
    def walk(node):
        for c in node:
            if local(c) in ("xmpmeta","RDF"): continue
            t = (c.text or "").strip()
            if len(c) == 0:
                out[local(c)] = t
            else:
                if t: out[local(c)] = t
                walk(c)
    walk(data_el)
    return out


# ── template parsing ──────────────────────────────────────────────────────────
CONTAINERS = {"subform","pageArea","template","form1","purchaseOrder",
              "header","detail","total","comments","footer",
              "commentsHeader","detailHeader","Body","Header","Footer"}

class Elem:
    def __init__(self):
        self.kind=self.name=self.text=self.caption=self.halign=""
        self.x=self.y=self.w=self.h=self.font_size=self.caption_reserve=0.0
        self.font_weight="normal"; self.border_bottom=False; self.fill=None

def parse_el(node, ph, px=0., py=0.):
    out = []
    if not isinstance(node.tag, str):   # skip PIs/comments at top level
        return out
    lt = local(node)
    nx = px + to_pt(node.get("x","0"))
    ny = py + to_pt(node.get("y","0"))

    if lt in ("draw","field"):
        e = Elem()
        e.kind=lt; e.name=node.get("name","")
        e.x=nx; e.y=ny
        e.w=to_pt(node.get("w","0"))
        e.h=to_pt(node.get("h",node.get("minH","0"))) or to_pt("6.35mm")
        f = next(iterall(node,"font"),None)
        if f is not None:
            e.font_size=max(to_pt(f.get("size","9pt")),5.)
            e.font_weight=f.get("weight","normal")
        p = find1(node,"para")
        if p is not None: e.halign=p.get("hAlign","left")
        v = find1(node,"value")
        if v is not None: e.text=text_of(v)
        cap = find1(node,"caption")
        if cap is not None:
            cv = find1(cap,"value")
            if cv is not None: e.caption=text_of(cv)
            e.caption_reserve=to_pt(cap.get("reserve","0"))
        brd = find1(node,"border")
        if brd:
            edges=[c for c in brd if local(c)=="edge"]
            if len(edges)>=3:
                e.border_bottom = edges[2].get("presence","visible")!="hidden"
        for col in iterall(node,"color"):
            par = col.getparent()
            if par is not None and local(par)=="fill":
                parts=col.get("value","").split(",")
                if len(parts)==3:
                    try: e.fill=tuple(int(p.strip())/255. for p in parts)
                    except: pass
                break
        out.append(e)

    if lt in CONTAINERS:
        sx,sy=nx,ny
        mg=find1(node,"margin")
        if mg is not None:
            sx+=to_pt(mg.get("leftInset","0"))
            sy+=to_pt(mg.get("topInset","0"))
        for c in node:
            out.extend(parse_el(c,ph,sx,sy))
    return out


# ── rendering ─────────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = letter

def fy(y,h): return PAGE_H-y-h

def sfont(c,w,s):
    c.setFont("Helvetica-Bold" if w=="bold" else "Helvetica", max(s,5))

def draw_text(c,txt,x,y,w,h,s,wt,ha):
    if not txt or w<2: return
    sfont(c,wt,s)
    ty = y+(h-s)/2+1
    mc = max(1,int(w/max(s*0.55,1)))
    d = txt[:mc]
    if ha=="right":
        tw=c.stringWidth(d,"Helvetica-Bold" if wt=="bold" else "Helvetica",s)
        c.drawString(x+w-tw-2,ty,d)
    elif ha=="center":
        c.drawCentredString(x+w/2,ty,d)
    else:
        c.drawString(x+2,ty,d)

def render_el(e, data, c):
    px=e.x; py=fy(e.y,e.h)
    if e.w<1 and e.kind=="draw": return
    if e.fill:
        c.setFillColorRGB(*e.fill)
        c.rect(px,py,e.w,e.h,fill=1,stroke=0)
        c.setFillColorRGB(0,0,0)
    if e.border_bottom:
        c.setStrokeColorRGB(.3,.3,.3); c.setLineWidth(.5)
        c.line(px,py,px+e.w,py); c.setStrokeColorRGB(0,0,0)
    c.setFillColorRGB(0,0,0)
    if e.kind=="draw" and e.text:
        draw_text(c,e.text,px,py,e.w,e.h,e.font_size,e.font_weight,e.halign)
    elif e.kind=="field":
        if e.caption:
            sfont(c,"normal",e.font_size)
            c.drawString(px+1,py+(e.h-e.font_size)/2,e.caption+":")
        val=data.get(e.name) or data.get(e.name.lower(),"")
        if val:
            draw_text(c,str(val),px+e.caption_reserve,py,
                      e.w-e.caption_reserve,e.h,e.font_size,"normal",e.halign)


def split_pages(els):
    pages=[[]]
    for e in els:
        idx=int(e.y//PAGE_H)
        while len(pages)<=idx: pages.append([])
        c=Elem(); c.__dict__.update(e.__dict__); c.y=e.y-idx*PAGE_H
        pages[idx].append(c)
    return [p for p in pages if p]


# ── chunk finders ─────────────────────────────────────────────────────────────
def find_template(chunks):
    for key in ("template","xfa"):
        raw=chunks.get(key)
        if not raw: continue
        root=parse_xml(raw)
        if root is None: continue
        if local(root)=="template": return root
        for c in root.iter():
            if local(c)=="template": return c
    for raw in chunks.values():
        root=parse_xml(raw)
        if root is None: continue
        if local(root)=="template": return root
        for c in root.iter():
            if local(c)=="template": return c
    return None

def find_data(chunks):
    for key in ("datasets","xfa"):
        raw=chunks.get(key)
        if not raw: continue
        root=parse_xml(raw)
        if root is None: continue
        if local(root) in ("datasets","data"): return flatten(root)
        for c in root.iter():
            if local(c)=="datasets": return flatten(c)
    for raw in chunks.values():
        root=parse_xml(raw)
        if root is None: continue
        if local(root)=="datasets": return flatten(root)
    return {}


# ── main ──────────────────────────────────────────────────────────────────────
def render(inp, out):
    print(f"\n{'='*60}\n  XFA Renderer v3\n  In : {inp}\n  Out: {out}\n{'='*60}\n")

    print("[1/4] Extracting XFA chunks...")
    chunks = extract_xfa_chunks(inp)
    print(f"      Chunks: {list(chunks.keys())}")
    if not chunks:
        sys.exit("No XFA chunks found.")

    print("[2/4] Finding template...")
    tmpl = find_template(chunks)
    if tmpl is None:
        print("  Chunk previews:")
        for n,r in chunks.items(): print(f"    [{n}] {r[:100]}")
        sys.exit("Template not found.")
    print(f"      Tag: {tmpl.tag}")

    print("[3/4] Loading data...")
    data = find_data(chunks)
    print(f"      {len(data)} values")

    print("[4/4] Rendering...")
    els = [e for e in parse_el(tmpl, PAGE_H, 0, 0) if e.w>0 or e.kind=="field"]
    print(f"      {len(els)} elements")
    pages = split_pages(els)
    print(f"      {len(pages)} page(s)")

    c = canvas.Canvas(out, pagesize=letter)
    for pi, pels in enumerate(pages):
        print(f"      Page {pi+1}: {len(pels)} elements")
        for e in pels:
            try: render_el(e, data, c)
            except Exception as ex: print(f"        [warn] {e.name}: {ex}")
        c.showPage()
    c.save()
    print(f"\n  Done -> {out}\n")


if __name__=="__main__":
    if len(sys.argv)<2: print(__doc__); sys.exit(1)
    inp=sys.argv[1]
    if not os.path.exists(inp): sys.exit(f"Not found: {inp}")
    out=sys.argv[2] if len(sys.argv)>=3 else os.path.splitext(inp)[0]+"_rendered.pdf"
    render(inp,out)
    #