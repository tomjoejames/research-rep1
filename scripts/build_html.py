#!/usr/bin/env python3
"""
Convert paper_v2.md to a professional, self-contained HTML file.
Uses Python's markdown library + custom CSS for academic styling.
Embeds figures as base64 for a single-file deliverable.
"""

import re
import base64
import os
from pathlib import Path

MANUSCRIPT_DIR = Path('/Users/tom/Documents/Research Paper/manuscript')
MD_FILE = MANUSCRIPT_DIR / 'paper_v2.md'
HTML_FILE = MANUSCRIPT_DIR / 'paper_v2.html'
FIGURES_DIR = MANUSCRIPT_DIR / 'figures'

# Read markdown
md_text = MD_FILE.read_text(encoding='utf-8')

# ─── Simple Markdown → HTML conversion (no external deps) ────────────

def md_to_html(text):
    """Minimal markdown to HTML converter for academic papers."""
    lines = text.split('\n')
    html_parts = []
    in_table = False
    in_code = False
    in_list = False
    table_rows = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Code blocks
        if line.strip().startswith('```'):
            if in_code:
                html_parts.append('</code></pre>')
                in_code = False
            else:
                lang = line.strip().replace('```', '').strip()
                html_parts.append(f'<pre><code class="{lang}">')
                in_code = True
            i += 1
            continue
        
        if in_code:
            html_parts.append(line.replace('<', '&lt;').replace('>', '&gt;'))
            i += 1
            continue
        
        # Horizontal rules
        if line.strip() == '---':
            html_parts.append('<hr>')
            i += 1
            continue
        
        # Headers
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            text_content = line.lstrip('#').strip()
            text_content = inline_format(text_content)
            html_parts.append(f'<h{level}>{text_content}</h{level}>')
            i += 1
            continue
        
        # Images
        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line.strip())
        if img_match:
            alt, src = img_match.groups()
            # Try to embed as base64
            img_path = MANUSCRIPT_DIR / src
            if img_path.exists():
                with open(img_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
                ext = img_path.suffix.lower()
                mime = 'image/png' if ext == '.png' else 'image/jpeg'
                html_parts.append(f'<figure><img src="data:{mime};base64,{b64}" alt="{alt}"><figcaption>{alt}</figcaption></figure>')
            else:
                html_parts.append(f'<figure><img src="{src}" alt="{alt}"><figcaption>{alt}</figcaption></figure>')
            i += 1
            continue
        
        # Tables
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_rows = []
            cells = [c.strip() for c in line.split('|')[1:-1]]
            # Skip separator rows
            if all(re.match(r'^[-:]+$', c) for c in cells):
                i += 1
                continue
            table_rows.append(cells)
            # Check if next line is not a table
            if i + 1 >= len(lines) or '|' not in lines[i + 1] or not lines[i + 1].strip().startswith('|'):
                # Render table
                html_parts.append('<div class="table-wrapper"><table>')
                for j, row in enumerate(table_rows):
                    tag = 'th' if j == 0 else 'td'
                    wrapper = 'thead' if j == 0 else 'tbody' if j == 1 else ''
                    if j == 0:
                        html_parts.append('<thead><tr>')
                    elif j == 1:
                        html_parts.append('</thead><tbody><tr>')
                    else:
                        html_parts.append('<tr>')
                    for cell in row:
                        cell_html = inline_format(cell)
                        html_parts.append(f'<{tag}>{cell_html}</{tag}>')
                    html_parts.append('</tr>')
                html_parts.append('</tbody></table></div>')
                in_table = False
                table_rows = []
            i += 1
            continue
        
        # Blockquotes / notes
        if line.strip().startswith('*') and line.strip().endswith('*') and not line.strip().startswith('**'):
            content = inline_format(line.strip())
            html_parts.append(f'<p class="note">{content}</p>')
            i += 1
            continue
        
        # Paragraphs
        if line.strip():
            para = inline_format(line)
            html_parts.append(f'<p>{para}</p>')
        
        i += 1
    
    return '\n'.join(html_parts)


def inline_format(text):
    """Handle inline markdown formatting."""
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Math (LaTeX)
    text = re.sub(r'\$\$(.+?)\$\$', r'<span class="math">\1</span>', text)
    text = re.sub(r'\$([^$]+?)\$', r'<span class="math">\1</span>', text)
    # Links
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    return text


# ─── CSS ──────────────────────────────────────────────────────────────

CSS = """
:root {
    --primary: #1a365d;
    --accent: #2b6cb0;
    --bg: #ffffff;
    --text: #1a202c;
    --muted: #718096;
    --border: #e2e8f0;
    --code-bg: #f7fafc;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.7;
    color: var(--text);
    background: var(--bg);
    max-width: 800px;
    margin: 0 auto;
    padding: 40px 30px;
}

h1 {
    font-size: 18pt;
    font-weight: 700;
    color: var(--primary);
    margin: 30px 0 10px;
    line-height: 1.3;
    text-align: center;
}

h2 {
    font-size: 14pt;
    font-weight: 700;
    color: var(--primary);
    margin: 28px 0 12px;
    padding-bottom: 4px;
    border-bottom: 2px solid var(--accent);
}

h3 {
    font-size: 12pt;
    font-weight: 700;
    color: var(--accent);
    margin: 22px 0 10px;
}

p {
    margin: 8px 0;
    text-align: justify;
    hyphens: auto;
}

strong { font-weight: 700; }
em { font-style: italic; }

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

code {
    font-family: 'Menlo', 'Consolas', monospace;
    font-size: 9pt;
    background: var(--code-bg);
    padding: 1px 4px;
    border-radius: 3px;
    border: 1px solid var(--border);
}

pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px;
    overflow-x: auto;
    margin: 12px 0;
}

pre code {
    background: none;
    border: none;
    padding: 0;
    font-size: 9pt;
}

hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 30px 0;
}

.table-wrapper {
    overflow-x: auto;
    margin: 14px 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    font-size: 10pt;
    margin: 0;
}

th, td {
    border: 1px solid var(--border);
    padding: 6px 10px;
    text-align: left;
}

th {
    background: var(--primary);
    color: white;
    font-weight: 600;
    font-size: 9.5pt;
}

tr:nth-child(even) { background: #f8fafc; }
tr:hover { background: #edf2f7; }

figure {
    margin: 20px 0;
    text-align: center;
}

figure img {
    max-width: 100%;
    height: auto;
    border: 1px solid var(--border);
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

figcaption {
    font-size: 9.5pt;
    color: var(--muted);
    margin-top: 6px;
    font-style: italic;
}

.note {
    font-size: 9.5pt;
    color: var(--muted);
    font-style: italic;
}

.math {
    font-family: 'Cambria Math', 'STIXGeneral', serif;
    font-style: italic;
}

@media print {
    body { max-width: 100%; padding: 20px; font-size: 10pt; }
    figure { page-break-inside: avoid; }
    table { page-break-inside: avoid; }
    h2, h3 { page-break-after: avoid; }
}

@page {
    size: A4;
    margin: 2cm;
}
"""

# ─── Build HTML ───────────────────────────────────────────────────────

body_html = md_to_html(md_text)

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deployability of LLM-Based Agent Workflows on Resource-Constrained CPU-Only Systems</title>
    <meta name="description" content="Empirical evaluation of deploying LLM-based agent workflows on resource-constrained, CPU-only consumer hardware.">
    <style>{CSS}</style>
</head>
<body>
{body_html}
</body>
</html>"""

HTML_FILE.write_text(full_html, encoding='utf-8')
size_kb = len(full_html.encode('utf-8')) / 1024
print(f'✅ Generated {HTML_FILE.name} ({size_kb:.0f} KB)')
print(f'   Figures embedded: {len(list(FIGURES_DIR.glob("*.png")))}')
