"""
XFA PDF Renderer v5
-------------------
Usage:  python render_xfa.py <input.pdf> [output.pdf]
"""
# coding: utf-8
import sys, os, re, textwrap, base64, io, tempfile

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
try:
    from PIL import Image as PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


# ── units ─────────────────────────────────────────────────────────────────────
_U = re.compile(r'^([\-\d.]+)(mm|in|pt|cm)?$')
def to_pt(s):
    if not s: return 0.0
    m = _U.match(str(s).strip())
    if not m: return 0.0
    n, u = float(m.group(1)), m.group(2) or 'pt'
    return n * {'mm':2.8346,'cm':28.346,'in':72.0,'pt':1.0}[u]


# ── number formatting ──────────────────────────────────────────────────────────
def fmt_currency(s):
    try:
        return f"${float(s):,.2f}"
    except (ValueError, TypeError):
        return str(s)

def fmt_number(s):
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else f"{f:g}"
    except (ValueError, TypeError):
        return str(s)

CURRENCY_FIELDS = {
    'numTotal','numStateTax','numFederalTax','numShippingCharge','numGrandTotal',
    'numUnitPrice','numAmount',
}

def format_field_value(name, raw):
    if not raw:
        return raw
    if name in CURRENCY_FIELDS:
        return fmt_currency(raw)
    if name == 'numStateTaxRate':
        try:
            return f"State Tax @ {float(raw):.2f}%"
        except (ValueError, TypeError):
            return raw
    if name == 'numFederalTaxRate':
        try:
            return f"Federal Tax @ {float(raw):.2f}%"
        except (ValueError, TypeError):
            return raw
    if name in ('numQty',):
        return fmt_number(raw)
    # Checkbox fields — show filled square if checked
    if name.startswith('chk') and raw in ('1','true','True'):
        return "■"
    return raw


# ── XFA chunk extraction ───────────────────────────────────────────────────────
def _deref(obj, pdf):
    try:
        if obj.is_indirect:
            return pdf.make_indirect(obj)
    except Exception:
        pass
    try:
        og = obj.objgen
        if og[1] != 0 or og[0] != 0:
            return pdf.get_object(og)
    except Exception:
        pass
    return obj

def _read(stream_obj, pdf):
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
    root_dict = pdf.Root
    acroform_obj = root_dict.get("/AcroForm")
    if acroform_obj is None:
        raise ValueError("/AcroForm not found")
    acroform_obj = _deref(acroform_obj, pdf)
    xfa_obj = acroform_obj.get("/XFA")
    if xfa_obj is None:
        raise ValueError("/XFA not found in /AcroForm")
    xfa_obj = _deref(xfa_obj, pdf)
    chunks = {}
    if isinstance(xfa_obj, pikepdf.Stream):
        raw = _read(xfa_obj, pdf)
        if raw:
            chunks["xfa"] = raw
        return chunks
    if isinstance(xfa_obj, pikepdf.Array):
        i = 0
        while i < len(xfa_obj):
            item = _deref(xfa_obj[i], pdf)
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


# ── XML helpers ────────────────────────────────────────────────────────────────
def parse_xml(raw):
    try:
        return etree.fromstring(raw)
    except Exception:
        return None

def local(el):
    t = el.tag
    if not isinstance(t, str):
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
    if value_el is None:
        return ""
    for c in value_el.iter():
        lt = local(c)
        if lt in ("text","integer","date","decimal","float"):
            return (c.text or "").strip()
        if lt in ("body","p"):
            parts = []
            for n in c.iter():
                if n.text and n.text.strip():
                    parts.append(n.text.strip())
            return " ".join(parts)
    return ""

def get_image_data(node):
    """Extract base64 image from a draw node; return (bytes, contentType) or (None,None)."""
    v = find1(node, "value")
    if v is None:
        return None, None
    for c in v.iter():
        if local(c) == "image":
            ct = c.get("contentType","")
            raw_b64 = (c.text or "").replace('\n','').replace(' ','').replace('\r','')
            if raw_b64:
                try:
                    return base64.b64decode(raw_b64), ct
                except Exception:
                    return None, None
    return None, None


# ── Dataset helpers ────────────────────────────────────────────────────────────
def flatten(root_el):
    out = {}
    data_el = next((c for c in root_el.iter() if local(c) == "data"), root_el)
    def walk(node):
        for c in node:
            if local(c) in ("xmpmeta","RDF"):
                continue
            t = (c.text or "").strip()
            k = local(c)
            if not k:
                continue
            if len(c) == 0:
                if k not in out:
                    out[k] = t
            else:
                if t and k not in out:
                    out[k] = t
                walk(c)
    walk(data_el)
    return out

def get_repeated_data(root_el, subform_name):
    data_el = next((c for c in root_el.iter() if local(c) == "data"), root_el)
    results = []
    def walk(node):
        for c in node:
            if local(c) == subform_name:
                row = {}
                for field in c:
                    fn = local(field)
                    if fn:
                        row[fn] = (field.text or "").strip()
                results.append(row)
            else:
                walk(c)
    walk(data_el)
    return results


# ── Rendered element ───────────────────────────────────────────────────────────
class Elem:
    def __init__(self):
        self.kind = ""           # "draw" | "field" | "image"
        self.name = ""
        self.text = ""
        self.caption = ""
        self.caption_reserve = 0.0
        self.halign = "left"
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
        self.h = 0.0
        self.font_size = to_pt("9pt")
        self.font_weight = "normal"
        self.border_bottom = False
        self.border_all = False
        self.fill = None
        self.multiline = False
        self.is_button = False
        self.is_signature = False
        self.caption_placement = "left"  # or "top"
        self.image_bytes = None   # raw image bytes for 'image' kind
        self.image_type = ""      # mime type


# ── Template parsing ───────────────────────────────────────────────────────────
FLOW_CONTAINERS = {
    "subform","pageArea","template","form1","purchaseOrder",
    "header","detail","total","comments","footer",
    "commentsHeader","detailHeader","Body","Header","Footer",
    "exclGroup",
}

def parse_template_node(node, abs_x=0.0, abs_y=0.0):
    out = []
    lt = local(node)
    if not lt:
        return out

    nx = abs_x + to_pt(node.get("x","0"))
    ny = abs_y + to_pt(node.get("y","0"))

    if lt in ("draw", "field"):
        e = Elem()
        e.kind = lt
        e.name = node.get("name","")
        e.x = nx
        e.y = ny
        raw_w = to_pt(node.get("w","0"))
        raw_h = to_pt(node.get("h", node.get("minH","0")))
        e.h = raw_h if raw_h > 0 else to_pt("6.35mm")
        e.w = raw_w

        f = next(iterall(node, "font"), None)
        if f is not None:
            e.font_size = max(to_pt(f.get("size","9pt")), 5.0)
            e.font_weight = f.get("weight","normal")

        p = find1(node, "para")
        if p is not None:
            e.halign = p.get("hAlign","left")

        # Static value — but check if it's an image first
        img_bytes, img_type = get_image_data(node)
        if img_bytes:
            e.kind = "image"
            e.image_bytes = img_bytes
            e.image_type = img_type
        else:
            v = find1(node, "value")
            if v is not None:
                e.text = text_of(v)

        cap = find1(node, "caption")
        if cap is not None:
            cv = find1(cap, "value")
            if cv is not None:
                e.caption = text_of(cv)
            e.caption_reserve = to_pt(cap.get("reserve","0"))
            e.caption_placement = cap.get("placement","left")

        ui = find1(node, "ui")
        if ui is not None:
            if find1(ui, "button") is not None:
                e.is_button = True
            if find1(ui, "signature") is not None:
                e.is_signature = True

        brd = find1(node, "border")
        if brd is not None:
            edges = [c for c in brd if local(c) == "edge"]
            if len(edges) >= 3:
                e.border_bottom = edges[2].get("presence","visible") != "hidden"
            if len(edges) >= 4:
                vis = [c.get("presence","visible") != "hidden" for c in edges[:4]]
                e.border_all = all(vis)

        for col in iterall(node, "color"):
            par = col.getparent()
            if par is not None and local(par) == "fill":
                parts = col.get("value","").split(",")
                if len(parts) == 3:
                    try:
                        e.fill = tuple(int(pp.strip())/255.0 for pp in parts)
                    except Exception:
                        pass
                break

        te = find1(node, "textEdit")
        if te is not None and te.get("multiLine","0") == "1":
            e.multiline = True

        out.append(e)

    elif lt in FLOW_CONTAINERS:
        sx, sy = nx, ny
        mg = find1(node, "margin")
        if mg is not None:
            sx += to_pt(mg.get("leftInset","0"))
            sy += to_pt(mg.get("topInset","0"))
        for child in node:
            out.extend(parse_template_node(child, sx, sy))

    return out


# ── Image rendering ────────────────────────────────────────────────────────────
_img_cache = {}

def draw_image_elem(c, e, rl_y, effective_w):
    """Render an image element using PIL + ReportLab drawImage."""
    if not _PIL_OK or not e.image_bytes:
        # Fallback: gray placeholder
        c.setFillColorRGB(0.85, 0.85, 0.85)
        c.rect(e.x, rl_y, effective_w, e.h, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0)
        return
    cache_key = id(e.image_bytes)
    if cache_key not in _img_cache:
        try:
            img = PILImage.open(io.BytesIO(e.image_bytes))
            if img.mode not in ('RGB','RGBA','L'):
                img = img.convert('RGB')
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            img.save(tmp.name, 'PNG')
            tmp.close()
            _img_cache[cache_key] = tmp.name
        except Exception:
            _img_cache[cache_key] = None
    path = _img_cache.get(cache_key)
    if path:
        try:
            c.drawImage(path, e.x, rl_y, width=effective_w, height=e.h,
                        preserveAspectRatio=True, mask='auto')
        except Exception:
            pass


# ── Text helpers ───────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = letter
MARGIN_TOP = to_pt("19.05mm")
ROW_H = to_pt("6.35mm")

def _fname(weight):
    return "Helvetica-Bold" if weight == "bold" else "Helvetica"

def _set_font(c, weight, size):
    c.setFont(_fname(weight), max(size, 5.0))

def _fit_text(c, txt, fname, size, max_w):
    s = str(txt)
    while len(s) > 1 and c.stringWidth(s, fname, size) > max_w - 4:
        s = s[:-1]
    return s

def draw_text_in_box(c, txt, x, rl_y, w, h, size, weight, halign="left"):
    if not txt or w < 4:
        return
    _set_font(c, weight, size)
    fn = _fname(weight)
    ty = rl_y + (h - size) / 2.0 + 1.0
    display = _fit_text(c, txt, fn, size, w)
    tw = c.stringWidth(display, fn, size)
    if halign == "right":
        c.drawString(x + w - tw - 3, ty, display)
    elif halign == "center":
        c.drawCentredString(x + w / 2.0, ty, display)
    else:
        c.drawString(x + 2, ty, display)

def draw_multiline_in_box(c, txt, x, rl_y, w, h, size, weight):
    if not txt or w < 4:
        return
    _set_font(c, weight, size)
    line_h = size * 1.2
    chars_per_line = max(1, int(w / max(size * 0.56, 1)))
    lines = []
    for para in txt.replace('\r\n','\n').replace('\r','\n').split('\n'):
        lines.extend(textwrap.wrap(para, chars_per_line) if para.strip() else [""])
    cur_y = rl_y + h - size - 2
    for line in lines:
        if cur_y < rl_y:
            break
        c.drawString(x + 2, cur_y, line)
        cur_y -= line_h

def count_wrapped_lines(txt, w, size):
    chars_per_line = max(1, int(w / max(size * 0.56, 1)))
    lines = []
    for para in txt.replace('\r\n','\n').replace('\r','\n').split('\n'):
        lines.extend(textwrap.wrap(para, chars_per_line) if para.strip() else [""])
    return len(lines)


# ── Coordinate conversion ──────────────────────────────────────────────────────
def rl_y_for(flow_y, elem_y, elem_h):
    return PAGE_H - (flow_y + elem_y) - elem_h


# ── Layout engine ──────────────────────────────────────────────────────────────
class LayoutEngine:
    def __init__(self, cv):
        self.cv = cv
        self.page_y = MARGIN_TOP
        self.page_num = 1
        self._decoration_cb = None

    def set_decoration_cb(self, cb):
        self._decoration_cb = cb

    def remaining(self):
        # Reserve space for bottom page chrome (footer banner at y=738pt)
        return PAGE_H - self.page_y - to_pt("19mm")

    def new_page(self):
        self.cv.showPage()
        self.page_num += 1
        self.page_y = MARGIN_TOP
        if self._decoration_cb:
            self._decoration_cb(self.cv)

    def ensure_space(self, needed):
        if self.remaining() < needed:
            self.new_page()
            return True
        return False

    def draw_elems(self, elems, data=None, section_y=None):
        sy = self.page_y if section_y is None else section_y
        c = self.cv
        for e in elems:
            effective_w = e.w if e.w > 0 else (PAGE_W - e.x - to_pt("12mm"))
            rl_y = rl_y_for(sy, e.y, e.h)
            if rl_y + e.h < 0 or rl_y > PAGE_H:
                continue

            # ── Image ─────────────────────────────────────────────────────
            if e.kind == "image":
                draw_image_elem(c, e, rl_y, effective_w)
                continue

            # ── Background fill ───────────────────────────────────────────
            if e.fill:
                c.setFillColorRGB(*e.fill)
                c.rect(e.x, rl_y, effective_w, e.h, fill=1, stroke=0)
                c.setFillColorRGB(0, 0, 0)

            # ── Button ────────────────────────────────────────────────────
            if e.is_button:
                c.setFillColorRGB(0.85, 0.83, 0.80)
                c.rect(e.x, rl_y, effective_w, e.h, fill=1, stroke=0)
                c.setStrokeColorRGB(0.5, 0.5, 0.5)
                c.setLineWidth(0.5)
                c.rect(e.x, rl_y, effective_w, e.h, fill=0, stroke=1)
                c.setFillColorRGB(0, 0, 0)
                label = e.caption or e.text
                if label:
                    draw_text_in_box(c, label, e.x, rl_y, effective_w, e.h,
                                     e.font_size, e.font_weight, "center")
                continue

            # ── Borders ───────────────────────────────────────────────────
            c.setStrokeColorRGB(0.4, 0.4, 0.4)
            c.setLineWidth(0.5)
            if e.border_all:
                c.rect(e.x, rl_y, effective_w, e.h, fill=0, stroke=1)
            elif e.border_bottom:
                c.line(e.x, rl_y, e.x + effective_w, rl_y)
            c.setStrokeColorRGB(0, 0, 0)
            c.setFillColorRGB(0, 0, 0)

            # ── Resolve value ─────────────────────────────────────────────
            raw_val = ""
            if data and e.name:
                raw_val = data.get(e.name, data.get(e.name.lower(), ""))
            if not raw_val:
                raw_val = e.text
            display_val = format_field_value(e.name, raw_val) if e.name else raw_val

            # ── Signature field ───────────────────────────────────────────
            if e.is_signature:
                sig_w = effective_w * 0.65
                c.setLineWidth(0.5)
                c.line(e.x, rl_y, e.x + sig_w, rl_y)
                if e.caption:
                    cap_h = to_pt("5mm")
                    draw_text_in_box(c, e.caption, e.x, rl_y - cap_h,
                                     sig_w, cap_h, e.font_size, "normal", "left")
                continue

            # ── Draw (static text) ────────────────────────────────────────
            if e.kind == "draw":
                if display_val:
                    if e.multiline:
                        draw_multiline_in_box(c, display_val, e.x, rl_y,
                                              effective_w, e.h,
                                              e.font_size, e.font_weight)
                    else:
                        draw_text_in_box(c, display_val, e.x, rl_y,
                                         effective_w, e.h,
                                         e.font_size, e.font_weight, e.halign)
            # ── Field ────────────────────────────────────────────────────
            else:
                if e.caption_placement == "top":
                    cap_h = e.caption_reserve  # reserve = height of caption band
                    val_area_h = e.h - cap_h
                    if e.caption:
                        draw_text_in_box(c, e.caption, e.x + 1, rl_y + val_area_h,
                                         effective_w, cap_h, e.font_size, "bold", "left")
                    if display_val:
                        if e.multiline:
                            draw_multiline_in_box(c, str(display_val), e.x, rl_y,
                                                  effective_w, val_area_h, e.font_size, "normal")
                        else:
                            draw_text_in_box(c, str(display_val), e.x, rl_y,
                                             effective_w, val_area_h, e.font_size, "normal", e.halign)
                else:
                    if e.caption:
                        draw_text_in_box(c, e.caption + ":", e.x + 1, rl_y,
                                         e.caption_reserve, e.h,
                                         e.font_size, "normal", "left")
                    if display_val:
                        vx = e.x + e.caption_reserve
                        vw = effective_w - e.caption_reserve
                        if e.multiline:
                            draw_multiline_in_box(c, str(display_val), vx, rl_y,
                                                  vw, e.h, e.font_size, "normal")
                        else:
                            draw_text_in_box(c, str(display_val), vx, rl_y, vw, e.h,
                                             e.font_size, "normal", e.halign)


# ── Section utilities ──────────────────────────────────────────────────────────
def section_h(elems):
    if not elems:
        return ROW_H
    return max(e.y + e.h for e in elems)

def get_subform(tmpl, name):
    for node in tmpl.iter():
        if local(node) == "subform" and node.get("name") == name:
            return node
    return None

def get_page_area(tmpl):
    for node in tmpl.iter():
        if local(node) == "pageArea":
            return node
    return None

def elems_of(tmpl, subform_name):
    node = get_subform(tmpl, subform_name)
    return parse_template_node(node, 0.0, 0.0) if node is not None else []


# ── Chunk finders ──────────────────────────────────────────────────────────────
def find_template(chunks):
    for key in ("template","xfa"):
        raw = chunks.get(key)
        if not raw: continue
        root = parse_xml(raw)
        if root is None: continue
        if local(root) == "template": return root
        for c in root.iter():
            if local(c) == "template": return c
    for raw in chunks.values():
        root = parse_xml(raw)
        if root is None: continue
        if local(root) == "template": return root
        for c in root.iter():
            if local(c) == "template": return c
    return None

def find_datasets_root(chunks):
    for key in ("datasets","xfa"):
        raw = chunks.get(key)
        if not raw: continue
        root = parse_xml(raw)
        if root is None: continue
        if local(root) in ("datasets","data"): return root
        for c in root.iter():
            if local(c) == "datasets": return c
    for raw in chunks.values():
        root = parse_xml(raw)
        if root is None: continue
        if local(root) == "datasets": return root
    return None

def find_data(chunks):
    root = find_datasets_root(chunks)
    return flatten(root) if root is not None else {}

def find_detail_rows(chunks):
    root = find_datasets_root(chunks)
    return get_repeated_data(root, "detail") if root is not None else []


# ── Main render ────────────────────────────────────────────────────────────────
def render(inp, out):
    print(f"\n{'='*60}\n  XFA Renderer v5\n  In : {inp}\n  Out: {out}\n{'='*60}\n")

    print("[1/5] Extracting XFA chunks...")
    chunks = extract_xfa_chunks(inp)
    print(f"      Chunks: {list(chunks.keys())}")
    if not chunks:
        sys.exit("No XFA chunks found.")

    print("[2/5] Finding template...")
    tmpl = find_template(chunks)
    if tmpl is None:
        sys.exit("Template not found.")
    print(f"      Tag: {tmpl.tag}")

    print("[3/5] Loading data...")
    data = find_data(chunks)
    print(f"      {len(data)} scalar values")
    detail_rows = find_detail_rows(chunks)
    print(f"      {len(detail_rows)} detail rows")

    print("[4/5] Setting up canvas...")
    cv = canvas.Canvas(out, pagesize=letter)
    engine = LayoutEngine(cv)

    # ── Page decorations ───────────────────────────────────────────────────────
    page_area_node = get_page_area(tmpl)
    page_area_elems = (parse_template_node(page_area_node, 0.0, 0.0)
                       if page_area_node is not None else [])

    def draw_chrome(cv_):
        tmp = LayoutEngine(cv_)
        tmp.draw_elems(page_area_elems, data=data, section_y=0.0)

    engine.set_decoration_cb(draw_chrome)
    draw_chrome(cv)

    print("[5/5] Rendering sections...")

    # ── Header ─────────────────────────────────────────────────────────────────
    h_elems = elems_of(tmpl, "header")
    if h_elems:
        h = section_h(h_elems)
        engine.draw_elems(h_elems, data=data)
        engine.page_y += h + to_pt("4mm")

    # ── Detail table ───────────────────────────────────────────────────────────
    dh_elems = elems_of(tmpl, "detailHeader")
    row_elems = elems_of(tmpl, "detail")

    def draw_detail_header():
        if dh_elems:
            engine.draw_elems(dh_elems, data=None)
            engine.page_y += section_h(dh_elems)

    draw_detail_header()

    row_h = section_h(row_elems) if row_elems else ROW_H
    for row_data in detail_rows:
        if engine.remaining() < row_h:
            engine.new_page()
            draw_detail_header()
        if row_elems:
            formatted = {k: format_field_value(k, v) for k, v in row_data.items()}
            engine.draw_elems(row_elems, data=formatted)
            engine.page_y += row_h

    engine.page_y += to_pt("3mm")

    # ── Totals ─────────────────────────────────────────────────────────────────
    tot_elems = elems_of(tmpl, "total")
    if tot_elems:
        h = section_h(tot_elems)
        engine.ensure_space(h + ROW_H)
        engine.draw_elems(tot_elems, data=data)
        engine.page_y += h + to_pt("3mm")

    # ── Comments header button ─────────────────────────────────────────────────
    ch_elems = elems_of(tmpl, "commentsHeader")
    txt_comments = data.get("txtComments","")
    comm_elems = elems_of(tmpl, "comments")

    # Footer elements (needed for height cap calculation)
    foot_elems = elems_of(tmpl, "footer")

    # Compute dynamic height for comments text area
    if txt_comments and comm_elems:
        font_size = to_pt("9pt")
        line_h = font_size * 1.2
        box_w = to_pt("177.8mm")
        nlines = count_wrapped_lines(txt_comments, box_w, font_size)
        caption_h = to_pt("6.35mm")
        dyn_comments_h = max(caption_h + nlines * line_h + to_pt("4mm"), to_pt("12.7mm"))
        # Cap comments height so footer fits on the same page
        ch_h_for_cap = section_h(ch_elems) if ch_elems else 0.0
        foot_h_for_cap = section_h(foot_elems) if foot_elems else 0.0
        reserved = ch_h_for_cap + foot_h_for_cap + to_pt("10mm")
        max_comm_h = engine.remaining() - reserved
        if max_comm_h > to_pt("20mm"):
            dyn_comments_h = min(dyn_comments_h, max_comm_h)
        for e in comm_elems:
            if e.name == "txtComments":
                e.h = dyn_comments_h
                e.multiline = True

    # Render each section sequentially - new page only when needed
    if ch_elems:
        ch_h = section_h(ch_elems)
        engine.ensure_space(ch_h + ROW_H)
        engine.draw_elems(ch_elems, data=data)
        engine.page_y += ch_h

    if comm_elems and txt_comments:
        comm_h = section_h(comm_elems)
        engine.ensure_space(min(comm_h, engine.remaining()))  # use available space
        engine.draw_elems(comm_elems, data=data)
        engine.page_y += comm_h + to_pt("2mm")

    # ── Footer ─────────────────────────────────────────────────────────────────
    if foot_elems:
        foot_h = section_h(foot_elems)
        engine.ensure_space(foot_h + to_pt("5mm"))
        engine.draw_elems(foot_elems, data=data)
        engine.page_y += foot_h

    # Cleanup temp image files
    for path in _img_cache.values():
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception:
                pass

    cv.save()
    print(f"\n  Done -> {out}  ({engine.page_num} page(s))\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    inp = sys.argv[1]
    if not os.path.exists(inp):
        sys.exit(f"Not found: {inp}")
    out = (sys.argv[2] if len(sys.argv) >= 3
           else os.path.splitext(inp)[0] + "_rendered.pdf")
    render(inp, out)
