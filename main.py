"""
XFA PDF Inspector
-----------------
Extracts and inspects the XML streams embedded inside a dynamic XFA PDF.

Usage:
    python inspect_xfa.py <path_to_pdf>

Optional flags:
    --chunk <name>    Print only a specific XFA chunk (e.g. template, datasets, config)
    --list            List all available chunk names without printing content
    --save            Save each chunk as a separate .xml file next to the PDF
"""

import sys
import os
import argparse
from io import BytesIO

try:
    import pikepdf
except ImportError:
    print("pikepdf not found. Install it with:  pip install pikepdf")
    sys.exit(1)

try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    import xml.dom.minidom as minidom
    LXML_AVAILABLE = False
    print("[warning] lxml not found, falling back to xml.dom.minidom (less pretty). "
          "Install lxml with:  pip install lxml\n")


def pretty_print_xml(raw_bytes: bytes) -> str:
    """Pretty-print raw XML bytes."""
    try:
        if LXML_AVAILABLE:
            root = etree.fromstring(raw_bytes)
            return etree.tostring(root, pretty_print=True, encoding="unicode")
        else:
            parsed = minidom.parseString(raw_bytes)
            return parsed.toprettyxml(indent="  ")
    except Exception as e:
        # If it's not valid XML, just decode and return raw
        return f"[could not parse as XML: {e}]\n" + raw_bytes.decode("utf-8", errors="replace")


def extract_xfa_chunks(pdf_path: str) -> list[tuple[str, bytes]]:
    """
    Open a PDF and extract all XFA chunks as (name, raw_bytes) pairs.
    XFA is stored as an array in pdf.Root.AcroForm.XFA:
      [ "preamble", <stream>, "template", <stream>, ... ]
    """
    pdf = pikepdf.open(pdf_path)
    root = pdf.Root

    acroform = root.get("/AcroForm")
    if acroform is None:
        raise ValueError("No /AcroForm found — this PDF may not have a form at all.")

    xfa = acroform.get("/XFA")
    if xfa is None:
        raise ValueError("No /XFA key in /AcroForm — this is not an XFA form.")

    chunks = []

    # XFA can be a single stream or an array of name/stream pairs
    if isinstance(xfa, pikepdf.Stream):
        raw = bytes(xfa.read_bytes())
        chunks.append(("xfa", raw))
    elif isinstance(xfa, pikepdf.Array):
        items = list(xfa)
        # Array is [ name, stream, name, stream, ... ]
        i = 0
        while i < len(items):
            item = items[i]
            if isinstance(item, pikepdf.Name):
                name = str(item).lstrip("/")
                i += 1
                if i < len(items) and isinstance(items[i], pikepdf.Stream):
                    raw = bytes(items[i].read_bytes())
                    chunks.append((name, raw))
                    i += 1
                else:
                    chunks.append((name, b""))
            elif isinstance(item, pikepdf.Stream):
                raw = bytes(item.read_bytes())
                chunks.append((f"chunk_{i}", raw))
                i += 1
            else:
                i += 1
    else:
        raise ValueError(f"Unexpected XFA type: {type(xfa)}")

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Inspect XFA XML inside a PDF.")
    parser.add_argument("pdf", help="Path to the XFA PDF file")
    parser.add_argument("--chunk", help="Print only this chunk name (e.g. template)")
    parser.add_argument("--list", action="store_true", help="List chunk names only")
    parser.add_argument("--save", action="store_true", help="Save chunks as .xml files")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"File not found: {args.pdf}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  XFA Inspector: {os.path.basename(args.pdf)}")
    print(f"{'='*60}\n")

    try:
        chunks = extract_xfa_chunks(args.pdf)
    except ValueError as e:
        print(f"[error] {e}")
        sys.exit(1)

    if not chunks:
        print("No XFA chunks found.")
        sys.exit(0)

    print(f"Found {len(chunks)} XFA chunk(s):\n")
    for name, data in chunks:
        size_kb = len(data) / 1024
        print(f"  • {name:<20} ({size_kb:.1f} KB)")

    if args.list:
        return

    # Filter to a specific chunk if requested
    if args.chunk:
        chunks = [(n, d) for n, d in chunks if n.lower() == args.chunk.lower()]
        if not chunks:
            print(f"\n[error] No chunk named '{args.chunk}' found.")
            sys.exit(1)

    print()

    for name, raw in chunks:
        print(f"\n{'─'*60}")
        print(f"  CHUNK: {name}")
        print(f"{'─'*60}\n")

        if not raw.strip():
            print("  (empty)\n")
            continue

        pretty = pretty_print_xml(raw)
        print(pretty)

        # Save to file if requested
        if args.save:
            base = os.path.splitext(args.pdf)[0]
            out_path = f"{base}__{name}.xml"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(pretty)
            print(f"  [saved to {out_path}]")


if __name__ == "__main__":
    main()