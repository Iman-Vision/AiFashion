from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).parent.parent
GALLERY_DIR = BASE_DIR / "Re-PolyVore" / "Re-PolyVore" / "Re-PolyVore"


def _html_img(src: str, alt: str, cls: str = "") -> str:
    return f'<img class="{cls}" src="{src}" alt="{alt}"/>'


def _resolve_src(image: str, local_top: Optional[str]) -> str:
    if image.startswith("http://") or image.startswith("https://") or image.startswith("data:"):
        return image
    if image.startswith("/gallery/"):
        p = GALLERY_DIR / image[len("/gallery/"):]
        if p.exists():
            return p.resolve().as_uri()
    if local_top and Path(local_top).exists():
        p = Path(local_top).resolve().as_uri()
        return p
    return image


def render_moodboard_html(
    board: Dict[str, object],
    out_path: str,
    uploaded_image: Optional[str] = None,
    uploaded_type: str = "top",
    uploaded_label: Optional[str] = None,
) -> None:
    palette: List[str] = board.get("palette", ["#ffffff", "#eeeeee", "#cccccc"])  # type: ignore
    bottom = board.get("bottom")  # type: ignore
    shoes = board.get("shoes")  # type: ignore
    top = board.get("top")  # type: ignore
    title = board.get("title", "Mood Board")  # type: ignore

    top_src = _resolve_src(top["image"], None) if top else ""
    bottom_src = _resolve_src(bottom["image"], None) if bottom else ""  # type: ignore
    shoes_src = _resolve_src(shoes["image"], None) if shoes else ""  # type: ignore
    if uploaded_image:
        up_src = _resolve_src(uploaded_image, uploaded_image)
        if uploaded_type == "top":
            top_src = up_src
        elif uploaded_type == "bottom":
            bottom_src = up_src
        elif uploaded_type == "shoes":
            shoes_src = up_src
    ulabel = uploaded_label or uploaded_type.title()

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: linear-gradient(135deg, {palette[0]}, {palette[1]});
      color: #111;
    }}
    .wrap {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      grid-gap: 16px;
      padding: 24px;
    }}
    .card {{
      background: #fff8;
      backdrop-filter: blur(6px);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 8px 24px #0002;
    }}
    .card img {{
      width: 100%;
      height: 320px;
      object-fit: cover;
      display: block;
    }}
    .title {{
      padding: 16px 24px;
      font-size: 20px;
      font-weight: 700;
    }}
    .subtitle {{
      padding: 0 24px 16px 24px;
      color: #333;
      font-size: 14px;
    }}
    .palette {{
      display: flex;
      gap: 8px;
      padding: 0 24px 24px 24px;
      align-items: center;
    }}
    .swatch {{
      width: 28px;
      height: 28px;
      border-radius: 50%;
      border: 2px solid #fff;
      box-shadow: 0 2px 8px #0002;
    }}
    .header {{
      padding: 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .header h1 {{
      margin: 0;
      font-size: 28px;
      letter-spacing: 0.5px;
    }}
    .badge {{
      background: {palette[2]};
      color: #fff;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 12px;
    }}
    @media (max-width: 900px) {{
      .wrap {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
  </head>
  <body>
    <div class="header">
      <h1>{title}</h1>
      <div class="badge">Palette</div>
    </div>
    <div class="wrap">
      <div class="card">
        {_html_img(top_src, 'Top')}
        <div class="title">{'Top' if uploaded_type!='top' else ulabel}</div>
        <div class="subtitle">{(top and top.get('name')) or 'Upper piece'}</div>
        <div class="palette">
          {''.join([f'<div class="swatch" style="background:{c}"></div>' for c in palette])}
        </div>
      </div>
      <div class="card">
        {_html_img(bottom_src, 'Bottom')}
        <div class="title">{'Bottom' if uploaded_type!='bottom' else ulabel}</div>
        <div class="subtitle">{(bottom and bottom.get('name')) or 'Lower piece'}</div>
        <div class="palette">
          {''.join([f'<div class="swatch" style="background:{c}"></div>' for c in palette])}
        </div>
      </div>
      <div class="card">
        {_html_img(shoes_src, 'Shoes')}
        <div class="title">{'Shoes' if uploaded_type!='shoes' else ulabel}</div>
        <div class="subtitle">{(shoes and shoes.get('name')) or 'Footwear'}</div>
        <div class="palette">
          {''.join([f'<div class="swatch" style="background:{c}"></div>' for c in palette])}
        </div>
      </div>
    </div>
  </body>
</html>
""".strip()

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html, encoding="utf-8")
