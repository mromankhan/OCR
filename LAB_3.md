# Lab 3: Building Agentic Document Understanding

In this lesson, you will build an agent that combines tools for OCR, layout detection, reading order and Vision-Language Model (VLM).

**Learning Objectives:**
- Use PaddleOCR for text parsing and layout detection
- Use LayoutReader for sorting parsed text into reading order 
- Build VLM tools for chart and table analysis

## Background

LayoutReader is a model for determining reading order. By sorting information on each page, the model captures the logical sequence of text parsed from the document. For documents with multiple columns, floating captions, margin annotations, etc the reading order can be complex. 

- Input: Bounding boxes normalized to 0-1000 range
- Output: Reading order position for each box

LayoutReader uses LayoutLMv3 which was developed by Microsoft on the ReadingBank dataset (500,000+ annotated pages).

## Outline

- [1. Text Extraction with PaddleOCR + LayoutLM Ordering](#1)
  - [1.1. Running OCR on the Document](#1-1)
  - [1.2. Visualizing OCR Bounding Boxes](#1-2)
  - [1.3. Structuring OCR Results with a Dataclass](#1-3)
  - [1.4. LayoutLM Reading Order](#1-4)
  - [1.5. Visualizing the Reading Order](#1-5)
  - [1.6. Creating the Ordered Text Output](#1-6)
- [2. Layout Detection with PaddleOCR](#2)
  - [2.1. Processing Document Layout](#2-1)
  - [2.2. Structuring Layout Results](#2-2)
  - [2.3. Visualizing Layout Detection](#2-3)
  - [2.4. Cropping Regions for Agent Tools](#2-4)
- [3. Agent Tools](#3)
  - [3.1. VLM Helper and Prompts](#3-1)
  - [3.2. Creating the AnalyzeChart Tool](#3-2)
  - [3.3. Creating the AnalyzeTable Tool](#3-3)
  - [3.4. Testing the Tools](#3-4)
- [4. LangChain Agent](#4)
  - [4.1. Formatting Context for the Agent](#4-1)
  - [4.2. Creating the System Prompt](#4-2)
  - [4.3. Assembling the Agent](#4-3)
  - [4.4. Testing the Agent](#4-4)
  
  
  
## Architecture Overview 

<div align="center">
    <img src="architecture.png" width="700">
</div>


## Setup

Load environment variables including the OpenAI API key for VLM tools and agent.


import os
from dotenv import load_dotenv

_ = load_dotenv(override=True)


Import libraries:
- **Pillow** for image loading and manipulation
- **cv2 (OpenCV)** for image processing and bounding box visualization
- **matplotlib** for result visualization
- **numpy** for numerical operations on arrays
- **dataclass** for structured data storage
- **typing** for type hints


from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any



<a id="1"></a>

## 1. Text Extraction with PaddleOCR + LayoutLM Ordering

Extract text and determine reading order using PaddleOCR and LayoutLM.

PaddleOCR returns three components for each detected text region:
- **Recognized text** strings
- **Confidence scores**
- **Bounding box coordinates** (4-point polygons)


from paddleocr import PaddleOCR

# Initialize PaddleOCR 
ocr = PaddleOCR(lang='en')

# Load image
image_path = "report_original.png"
display(Image.open(image_path))



<a id="1-1"></a>

### 1.1. Running OCR on the Document

Run OCR on the document (the same economics report from previous lessons).


# Run OCR 
result = ocr.predict(image_path)
page = result[0]

texts = page['rec_texts']      # recognized text strings
scores = page['rec_scores']    # confidence scores
boxes = page['rec_polys']      # bounding box coordinates

print(f"Extracted {len(texts)} text regions")
print("\nFirst 10 regions:")
for text, score, box in list(zip(texts, scores, boxes))[:10]:
    coords = box.astype(int).tolist()
    print(f"{text:40} | {score:.3f} | {coords}")
    
    
<a id="1-2"></a>

### 1.2. Visualizing OCR Bounding Boxes

Draw bounding boxes to verify OCR detection. Each text line, table cell, and label gets its own box.



processed_img = page['doc_preprocessor_res']['output_img']
img_plot = processed_img.copy()
show_text= False

for text, box in zip(texts, boxes):
    pts = np.array(box, dtype=int)
    cv2.polylines(img_plot, [pts], True, (0, 255, 0), 2)
    x, y = pts[0]
    if show_text:
        cv2.putText(img_plot, text, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

plt.figure(figsize=(8, 10))
plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Aligned Bounding Boxes (Processed Image)")
plt.show()



<a id="1-3"></a>

### 1.3. Structuring OCR Results with a Dataclass

Structure OCR output using an `OCRRegion` dataclass for cleaner code:
- Typed structure for each text region
- `bbox_xyxy` property converts 4-point polygons to `[x1, y1, x2, y2]` format


# Store OCR results in a structured format
@dataclass
class OCRRegion:
    text: str
    bbox: list  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float
    
    @property
    def bbox_xyxy(self):
        """Return bbox as [x1, y1, x2, y2] format."""
        x_coords = [p[0] for p in self.bbox]
        y_coords = [p[1] for p in self.bbox]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

ocr_regions: List[OCRRegion] = []
for text, score, box in zip(texts, scores, boxes):
    ocr_regions.append(OCRRegion(
        text=text, 
        bbox=box.astype(int).tolist(), 
        confidence=score
    ))

print(f"Stored {len(ocr_regions)} OCR regions")


<a id="1-4"></a>

### 1.4. LayoutLM Reading Order

Simple ordering (eg top-to-bottom, left-to-right) does not apply to our complex document. We will use LayoutReader which itself uses LayoutLMv3 model. Hugging Face contains the LayoutLMv3 model. Additionally we use helper functions for LayoutReader available at this [repository](https://github.com/ppaanngggg/layoutreader.git). 



from transformers import LayoutLMv3ForTokenClassification
from layoutreader.v3.helpers import prepare_inputs, boxes2inputs, parse_logits

# Load LayoutReader model
print("Loading LayoutReader model...")
model_slug = "hantian/layoutreader"
layout_model = LayoutLMv3ForTokenClassification.from_pretrained(model_slug)
print("Model loaded successfully!")



Now implement a reading order function called `get_reading_order`: 

1. **Calculate image dimensions** - Estimate size from bounding boxes with 10% padding
2. **Normalize coordinates** - Scale boxes to 0-1000 range for LayoutLM
3. **Prepare inputs** - Convert to transformer format
4. **Run inference** - Get model predictions
5. **Parse results** - Extract reading order from output logits



def get_reading_order(ocr_regions):
    """
    Use LayoutReader to determine reading order of OCR regions.
    Returns list of reading order positions for each region index.
    """
    # 1. Calculate image dimensions from bounding boxes (with padding)
    max_x = max_y = 0
    for region in ocr_regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

    image_width = max_x * 1.1   # Add 10% padding
    image_height = max_y * 1.1

    # 2. Convert bboxes to LayoutReader format (normalized to 0-1000)
    boxes = []
    for region in ocr_regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        # Normalize to 0-1000 range
        left = int((x1 / image_width) * 1000)
        top = int((y1 / image_height) * 1000)
        right = int((x2 / image_width) * 1000)
        bottom = int((y2 / image_height) * 1000)
        boxes.append([left, top, right, bottom])

    # 3. Prepare inputs
    inputs = boxes2inputs(boxes)
    inputs = prepare_inputs(inputs, layout_model)
    
    # 4. Run inference
    logits = layout_model(**inputs).logits.cpu().squeeze(0)
    
    # 5. Parse the model's outputs to get reading order
    reading_order = parse_logits(logits, len(boxes))

    return reading_order

# Get reading order
reading_order = get_reading_order(ocr_regions)

print(f"Reading order determined for {len(reading_order)} regions")
print(f"First 20 positions: {reading_order[:20]}")




<a id="1-5"></a>

### 1.5. Visualizing the Reading Order

Visualize reading order with numbered overlays on each region. Numbers follow the predicted reading sequence.

> **Note:** The model may produce some non-sequential jumps. For complex documents, you may need to fine-tune a custom layout model.

import matplotlib.patches as patches

def visualize_reading_order(ocr_regions, image_array, reading_order, title="Reading Order"):
    """
    Visualize OCR regions with their reading order numbers using matplotlib.
    """
    
    fig, ax = plt.subplots(1, figsize=(10, 14))
    ax.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    
    # Create order mapping: index -> reading order position
    order_map = {i: order for i, order in enumerate(reading_order)}
    
    for i, region in enumerate(ocr_regions):
        bbox = region.bbox
        if bbox and len(bbox) >= 4:
            # Draw polygon
            ax.add_patch(patches.Polygon(bbox, linewidth=2, 
                                         edgecolor='blue',
                                         facecolor='none', alpha=0.7))
            # Add reading order number at center
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            ax.text(sum(xs)/len(xs), sum(ys)/len(ys), 
                    str(order_map.get(i, i)),
                    fontsize=13, color='red', 
                    ha='center', va='center', fontweight='bold')
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

visualize_reading_order(ocr_regions, processed_img, 
                        reading_order, "LayoutLM Reading Order")
                        
                        
                        
<a id="1-6"></a>

### 1.6. Creating the Ordered Text Output

Combine OCR text with reading order:
1. Pair each region with its reading position
2. Sort by position
3. Return structured list with position, text, confidence, and bbox

This ordered text provides agent context for answering text-based questions without VLM calls.


# Create ordered text content
def get_ordered_text(ocr_regions, reading_order):
    """
    Return OCR regions sorted by reading order
    with their text and confidence.
    """
    # 1. Create (reading_position, index, region) tuples and sort
    indexed_regions = [(reading_order[i], 
                        i, 
                        ocr_regions[i]) for i in range(len(ocr_regions))]
    
    # 2. Sort by reading position
    indexed_regions.sort(key=lambda x: x[0])  
    
    # 3. Extract ordered text info
    ordered_text = []
    for position, original_idx, region in indexed_regions:
        ordered_text.append({
            "position": position,
            "text": region.text,
            "confidence": region.confidence,
            "bbox": region.bbox_xyxy
        })
    
    return ordered_text

ordered_text = get_ordered_text(ocr_regions, reading_order)

print("Text in reading order:")
print("=" * 70)
ordered_text[:5]





<a id="2"></a>

## 2. Layout Detection with PaddleOCR

Beyond text extraction, identify **content types** using layout detection.

PaddleOCR's `LayoutDetection` identifies document structure. Each region includes:
- **label**: Content type (text, table, chart, figure, etc.)
- **score**: Confidence score
- **bbox**: Bounding box in XYXY format

from paddleocr import LayoutDetection

# Initialize layout detection 
layout_engine = LayoutDetection()



<a id="2-1"></a>

### 2.1. Processing Document Layout

Run layout detection to identify content types (text blocks, charts, titles, tables, etc.).

# Process document layout 
def process_document(image_path):
    """Get layout regions from document."""
    layout_result = layout_engine.predict(image_path)
    
    regions = []
    for box in layout_result[0]['boxes']:
        regions.append({
            'label': box['label'],
            'score': box['score'],
            'bbox': box['coordinate'],  # [x1, y1, x2, y2]
        })
    
    # Sort by confidence
    regions = sorted(regions, key=lambda x: x['score'], reverse=True)
    return regions

layout_results = process_document(image_path)

print(f"Detected {len(layout_results)} layout regions:")
for r in layout_results:
    print(f"  {r['label']:20} score: {r['score']:.3f}  bbox: {[int(x) for x in r['bbox']]}")
    
    
Top five detected regions:

layout_results[0:5]




<a id="2-2"></a>

### 2.2. Structuring Layout Results

Create a `LayoutRegion` dataclass with unique IDs for tool references.


@dataclass
class LayoutRegion:
    region_id: int
    region_type: str
    bbox: list  # [x1, y1, x2, y2]
    confidence: float
    
# Store layout regions in structured format
layout_regions: List[LayoutRegion] = []
for i, r in enumerate(layout_results):
    layout_regions.append(LayoutRegion(
        region_id=i,
        region_type=r['label'],
        bbox=[int(x) for x in r['bbox']],
        confidence=r['score']
    ))

print(f"Stored {len(layout_regions)} layout regions")




<a id="2-3"></a>

### 2.3. Visualizing Layout Detection

Visualize layout regions with color-coded boxes showing region ID, type, and confidence.

# Visualize layout detection 
from matplotlib import colormaps

def visualize_layout(image_path, layout_regions, min_confidence=0.5, 
                     title="Layout Detection"):
    """
    Visualize layout detection results using cv2 (same pattern as L2).
    """
    img = cv2.imread(image_path)
    img_plot = img.copy()
    
    # Get unique labels and generate colors
    labels = list(set(r.region_type for r in layout_regions))
    cmap = colormaps.get_cmap('tab20')
    color_map = {}
    for i, label in enumerate(labels):
        rgba = cmap(i % 20)
        color_map[label] = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
    
    for region in layout_regions:
        if region.confidence < min_confidence:
            continue
            
        color = color_map[region.region_type]
        x1, y1, x2, y2 = region.bbox
        
        # Draw rectangle
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=int)
        cv2.polylines(img_plot, [pts], True, color, 2)
        
        # Add label
        text = f"{region.region_id}: {region.region_type} ({region.confidence:.2f})"
        cv2.putText(img_plot, text, (x1, y1-8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    plt.figure(figsize=(12, 16))
    plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()
    
    return img_plot

visualize_layout(image_path, layout_regions, 
                 min_confidence=0.5, title="PaddleOCR Layout Detection");
                 
                 
                 

<a id="2-4"></a>

### 2.4. Cropping Regions for Agent Tools

Prepare cropped regions for VLM analysis:
- **Focused analysis** - VLM sees only relevant content
- **Reduced noise** - No surrounding text interference
- **Lower costs** - Smaller images reduce API costs

Images are base64-encoded for vision API compatibility.



import base64
from io import BytesIO

# Crop and save layout regions for agent tools
def crop_region(image, bbox, padding=10):
    """Crop a region from image with optional padding."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)
    return image.crop((x1, y1, x2, y2))

def image_to_base64(img):
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Load image for cropping
pil_image = Image.open(image_path)

# Store cropped regions in dictionary
region_images = {}
for region in layout_regions:
    cropped = crop_region(pil_image, region.bbox)
    region_images[region.region_id] = {
        'image': cropped,
        'base64': image_to_base64(cropped),
        'type': region.region_type,
        'bbox': region.bbox
    }

print(f"Cropped {len(region_images)} regions")

# Also store full image
full_image_base64 = image_to_base64(pil_image)


Display all cropped regions available to the agent:

# Show cropped regions
fig, axes = plt.subplots(5, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (region_id, data) in enumerate(list(region_images.items())[:14]):
    axes[i].imshow(data['image'])
    axes[i].set_title(f"Region {region_id}: {data['type']}")
    axes[i].axis('off')

# Hide unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()



<a id="3"></a>

## 3. Agent Tools

Create two specialized VLM tools with optimized prompts:
- **AnalyzeChart**: Interpret charts and figures
- **AnalyzeTable**: Extract structured table data

Specialized tools enable content-specific prompts and structured JSON outputs.



from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Initialize VLM for tools
vlm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


<a id="3-1"></a>

### 3.1. VLM Helper and Prompts

Prompts define structured VLM output with three components:
1. **Role definition** (e.g., "Chart Analysis specialist")
2. **Extraction fields** (chart type, axes, data points)
3. **JSON template** for consistent formatting



# Tool prompts
CHART_ANALYSIS_PROMPT = """You are a Chart Analysis specialist. 
Analyze this chart/figure image and extract:

1. **Chart Type**: (line, bar, scatter, pie, etc.)
2. **Title**: (if visible)
3. **Axes**: X-axis label, Y-axis label, and tick values
4. **Data Points**: Key values (peaks, troughs, endpoints)
5. **Trends**: Overall pattern description
6. **Legend**: (if present)

Return a JSON object with this structure:
```json
{{
  "chart_type": "...",
  "title": "...",
  "x_axis": {{"label": "...", "ticks": [...]}},
  "y_axis": {{"label": "...", "ticks": [...]}},
  "key_data_points": [...],
  "trends": "...",
  "legend": [...]
}}
```
"""


TABLE_ANALYSIS_PROMPT = """You are a Table Extraction specialist. 
Extract structured data from this table image.

1. **Identify Structure**: 
    - Column headers, row labels, data cells
2. **Extract All Data**: 
    - Preserve exact values and alignment
3. **Handle Special Cases**: 
    - Merged cells, empty cells (mark as null), multi-line headers

Return a JSON object with this structure:
```json
{{
  "table_title": "...",
  "column_headers": ["header1", "header2", ...],
  "rows": [
    {{"row_label": "...", "values": [val1, val2, ...]}},
    ...
  ],
  "notes": "any footnotes or source info"
}}
```
"""



VLM Helper Function

Call VLM with multimodal messages containing prompt and base64-encoded image.

def call_vlm_with_image(image_base64: str, prompt: str) -> str:
    """Call VLM with an image and prompt."""
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            }
        ]
    )
    response = vlm.invoke([message])
    return response.content


<a id="3-2"></a>

### 3.2. Creating the AnalyzeChart Tool

The `@tool` decorator converts functions to agent-usable tools. This tool validates region existence, retrieves cropped images, and calls the VLM with the chart analysis prompt.


@tool
def AnalyzeChart(region_id: int) -> str:
    """Analyze a chart or figure region using VLM. 
    Use this tool when you need to extract data from charts, graphs, or figures.
    
    Args:
        region_id: The ID of the layout region to analyze (must be a chart/figure type)
    
    Returns:
        JSON string with chart type, axes, data points, and trends
    """
    if region_id not in region_images:
        return f"Error: Region {region_id} not found. Available regions: {list(region_images.keys())}"
    
    region_data = region_images[region_id]
    
    if region_data['type'] not in ['chart', 'figure']:
        return f"Warning: Region {region_id} is type '{region_data['type']}', not a chart/figure. Proceeding anyway."
    
    result = call_vlm_with_image(region_data['base64'], CHART_ANALYSIS_PROMPT)
    
    return result

print("AnalyzeChart tool defined")


<a id="3-3"></a>

### 3.3. Creating the AnalyzeTable Tool

Create the table extraction tool using the table-specific prompt for structured data with headers and rows.



@tool
def AnalyzeTable(region_id: int) -> str:
    """
    Extract structured data from a table region using VLM.
    Use this tool when you need to extract tabular data 
    with headers and rows.
    
    Args:
        region_id: The ID of the layout region to analyze (must be a table type)
    
    Returns:
        JSON string with table headers, rows, and any notes
    """
    if region_id not in region_images:
        return f"Error: Region {region_id} not found. Available regions: {list(region_images.keys())}"
    
    region_data = region_images[region_id]
    
    if region_data['type'] != 'table':
        return f"Warning: Region {region_id} is type '{region_data['type']}', not a table. Proceeding anyway."
    
    result = call_vlm_with_image(region_data['base64'], TABLE_ANALYSIS_PROMPT)
    return result

print("AnalyzeTable tool defined")





<a id="3-4"></a>

### 3.4. Testing the Tools

Test tools individually to verify VLM connection and prompt effectiveness.

Analyze the first chart region. Data points are approximate—VLMs have limitations in precise visual localization.


# Test the tools
print("Testing AnalyzeChart...")
chart_regions = [r for r in layout_regions if r.region_type in ['chart', 'figure']]
if chart_regions:
    test_result = AnalyzeChart.invoke({"region_id": chart_regions[0].region_id})
    print(f"Chart analysis result:\n{test_result[:500]}...")
else:
    print("No chart regions found")
    
    
    
    
    
Test the table tool. Works well for simple tables but struggles with complex layouts and may hallucinate.


# Test table tool
print("Testing AnalyzeTable...")
table_regions = [r for r in layout_regions if r.region_type == 'table']
if table_regions:
    test_result = AnalyzeTable.invoke({"region_id": table_regions[0].region_id})
    print(f"Table analysis result:\n{test_result[:500]}...")
else:
    print("No table regions found")
    
    
<a id="4"></a>

## 4. LangChain Agent

Build the agent to orchestrate all components:
1. Receive question about document
2. Read system prompt with OCR text and layout info
3. Decide whether to answer from text or use tools
4. Call appropriate tools for visual content
5. Combine information into coherent response


Verify data structures: ordered text (OCR + LayoutLM) and layout regions (layout detection).


ordered_text[0]

layout_regions[0]


<a id="4-1"></a>

### 4.1. Formatting Context for the Agent

Convert data structures to readable text for the system prompt—the agent's "memory" of the document.


#Prepare context for the agent
def format_ordered_text(ordered_text, max_items=50):
    """Format ordered text for the system prompt."""
    lines = []
    for item in ordered_text[:max_items]:
        lines.append(f"[{item['position']}] {item['text']}")
    
    if len(ordered_text) > max_items:
        lines.append(f"... and {len(ordered_text) - max_items} more text regions")
    
    return "\n".join(lines)

def format_layout_regions(layout_regions):
    """Format layout regions for the system prompt."""
    lines = []
    for region in layout_regions:
        lines.append(f"  - Region {region.region_id}: {region.region_type} (confidence: {region.confidence:.3f})")
    return "\n".join(lines)

# Create the formatted strings
ordered_text_str = format_ordered_text(ordered_text)
layout_regions_str = format_layout_regions(layout_regions)

print("Formatted context for agent:")
print(f"- Ordered text: {len(ordered_text_str)} chars")
print(f"- Layout regions: {len(layout_regions_str)} chars")




<a id="4-2"></a>

### 4.2. Creating the System Prompt

Construct the system prompt with:
- **Role definition**: Document Intelligence Agent
- **Document context**: OCR text in reading order
- **Layout information**: Region types and IDs
- **Tool descriptions**: When to use each tool
- **Instructions**: How to handle different content types



# System prompt for the agent
SYSTEM_PROMPT = f"""You are a Document Intelligence Agent. 
You analyze documents by combining OCR text with visual analysis tools.

## Document Text (in reading order)
The following text was extracted using OCR and ordered using LayoutLM.

{ordered_text_str}

## Document Layout Regions
The following regions were detected in the document:

{layout_regions_str}

## Your Tools
- **AnalyzeChart(region_id)**: 
    - Use for chart/figure regions to extract data points, axes, and trends
- **AnalyzeTable(region_id)**: 
    - Use for table regions to extract structured tabular data

## Instructions
1. For TEXT regions: 
    - Use the OCR text provided above (it's already extracted)
2. For TABLE regions: 
    - Use the AnalyzeTable tool to get structured data
3. For CHART/FIGURE regions: 
    - Use the AnalyzeChart tool to extract visual data

When answering questions about the document, 
use the appropriate tools to get accurate information.
"""

print("System prompt created")
print(f"Total length: {len(SYSTEM_PROMPT)} characters")



<a id="4-3"></a>

### 4.3. Assembling the Agent

Assemble the agent using `create_tool_calling_agent`:
- **Tools**: AnalyzeChart and AnalyzeTable
- **LLM**: GPT-4o-mini for cost efficiency
- **Prompt**: System context + user input + agent scratchpad
- **Verbose mode**: Shows agent reasoning process


# Initialize the agent (using LangChain 0.1.x API)
tools = [AnalyzeChart, AnalyzeTable]

# LLM for the agent 
agent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",SYSTEM_PROMPT),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# 4. Create the tool-calling agent
agent = create_tool_calling_agent(agent_llm, tools, prompt)

# 5. Set up the AgentExecutor to run the tool-enabled loop
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)



<a id="4-4"></a>

### 4.4. Testing the Agent

Test the agent with three question types.


🚨   Different Run Results: The output generated by AI chat models can vary with each execution due to their non-deterministic nature. Don't be surprised if your results differ from those shown in the video.



**Test 1: Document Overview**

Ask a general question answerable from OCR text alone (no tool calls needed).

# Test the agent with a simple question
response = agent_executor.invoke({
    "input": "What types of content are in this document?"
              "List the main sections.",
})
print("\n" + "="*60)
print("Agent Response:")
print("="*60)
print(response["output"])


**Test 2: Table Data Extraction**

Extract table data by calling the `AnalyzeTable` tool. Verbose output shows reasoning.

# Test with table extraction
response = agent_executor.invoke({
    "input": "Extract the data from the table in this document." 
             "Return it in a structured format."})
print("\n" + "="*60)
print("Agent Response:")
print("="*60)
print(response["output"])



**Test 3: Chart Analysis**

Analyze the chart using the `AnalyzeChart` tool to extract visual information unavailable from OCR text.

# Test with chart analysis
response = agent_executor.invoke({
    "input": "Analyze the chart/figure in this document." 
    "What trends does it show?"})
print("\n" + "="*60)
print("Agent Response:")
print("="*60)
print(response["output"])


## Summary

Our hybrid approach breaks documents into separate regions of text, charts or tables. The LangChain agent uses different tools for each region. 

| Component | Purpose | Output |
|-----------|---------|--------|
| **PaddleOCR** | Text Parsing | Text + bounding boxes|
| **LayoutReader** | Reading order prediction | Sorted sequence of regions |
| **PaddleOCR** | Layout Detection | Region types (table, chart, text) |
| **VLM** | Analysis of charts/tables | JSON (title, legend,... / headers, rows,...) |

In the next lesson, you will study the **Agentic Document Extraction (ADE)** framework from LandingAI. It will handle text parsing, layout detection, reading order, multimodal reasoning, and schema-based extraction in unified API's. This will address several limitations of PaddleOCR on real-world documents. 

