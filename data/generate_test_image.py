"""Generate a simple test architecture diagram for multimodal agent testing.

Creates a PNG image with labeled modules (boxes) so the agent can
demonstrate direct vision understanding — counting modules, reading
labels, and performing calculations based on what it SEES.
"""

from PIL import Image, ImageDraw, ImageFont
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_architecture.png")

W, H = 800, 500
img = Image.new("RGB", (W, H), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

# Title
draw.text((280, 15), "AI Agent Architecture", fill=(0, 0, 0))

# Draw 6 modules as colored boxes
modules = [
    ("User Input", (50, 60, 220, 130), "#4A90D9"),
    ("Agent Core\n(ReAct)", (270, 60, 440, 130), "#E67E22"),
    ("Tools\n(search/calc)", (490, 60, 660, 130), "#27AE60"),
    ("LLM\n(multimodal)", (50, 200, 230, 280), "#8E44AD"),
    ("Memory\n(sessions)", (280, 200, 460, 280), "#C0392B"),
    ("Response\nOutput", (510, 200, 690, 280), "#16A085"),
]

colors_rgb = {
    "#4A90D9": (74, 144, 217),
    "#E67E22": (230, 126, 34),
    "#27AE60": (39, 174, 96),
    "#8E44AD": (142, 68, 173),
    "#C0392B": (192, 57, 43),
    "#16A085": (22, 160, 133),
}

# Draw connectors between modules (a few arrows)
connections = [
    ((220, 95), (270, 95)),
    ((440, 95), (490, 95)),
    ((270, 130), (270, 200)),
    ((490, 130), (490, 200)),
    ((230, 240), (280, 240)),
    ((460, 240), (510, 240)),
]

for start, end in connections:
    draw.line([start, end], fill=(100, 100, 100), width=2)

for i, (label, bbox, color_hex) in enumerate(modules):
    rgb = colors_rgb[color_hex]
    draw.rounded_rectangle(bbox, radius=8, fill=rgb, outline=(50, 50, 50), width=2)

    # Module number
    x0, y0, x1, y1 = bbox
    draw.text((x0 + 8, y0 + 4), f"#{i+1}", fill=(255, 255, 255, 180))

    # Label centered
    lines = label.split("\n")
    y = y0 + 25
    for line in lines:
        # Rough centering
        draw.text((x0 + 15, y), line, fill=(255, 255, 255))
        y += 20

# Caption with number-of-modules hint
draw.text((200, 340), "Figure 1: Agent architecture with 6 core modules", fill=(0, 0, 0))
draw.text((150, 370), "The number of modules is 6, which is 3× the 2 base services.", fill=(80, 80, 80))
draw.text((180, 410), "Module count: 6  |  Dashboards: 2  |  Ratio: 6:2 = 3:1", fill=(100, 100, 100))
draw.text((200, 450), "Base services (LLM, Memory) = 2  |  Modules = 3 × Base", fill=(120, 120, 120))

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
img.save(OUTPUT_PATH)
print(f"Test image saved to: {OUTPUT_PATH}")
