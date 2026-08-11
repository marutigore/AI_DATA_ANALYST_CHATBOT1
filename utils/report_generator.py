"""
Executive PDF & HTML Report Generator Utility.
Generates downloadable, executive-grade PDF reports compiling dataset profiling metrics,
key KPI summaries, and chat analysis history.
"""
import io
import logging
from typing import Dict, Any, List
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

logger = logging.getLogger(__name__)

def generate_pdf_report(
    filename: str, 
    metrics: Dict[str, Any], 
    chat_history: List[Dict[str, Any]]
) -> bytes:
    """
    Builds an executive PDF report document in memory.
    
    Args:
        filename (str): Name of the dataset file.
        metrics (Dict[str, Any]): Dataset profiling metrics (rows, cols, memory_mb, missing).
        chat_history (List[Dict[str, Any]]): Conversation turns history.
        
    Returns:
        bytes: PDF binary content ready for download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#4F46E5'),
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'DocHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1F2937'),
        spaceBefore=14,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#374151'),
        leading=14
    )
    
    elements = []
    
    # Title Header
    elements.append(Paragraph("Luminary AI — Executive Data Analysis Report", title_style))
    elements.append(Paragraph(f"<b>Dataset File:</b> {filename}", body_style))
    elements.append(Spacer(1, 12))
    
    # KPI Metrics Table
    elements.append(Paragraph("Dataset Profiling Summary", heading_style))
    kpi_data = [
        ["Total Records / Rows", "Feature Columns", "Memory Usage (MB)", "Missing Values"],
        [
            f"{metrics.get('rows', 0):,}",
            f"{metrics.get('cols', 0)}",
            f"{metrics.get('memory_mb', 0):.2f} MB",
            f"{metrics.get('missing', 0)}"
        ]
    ]
    t = Table(kpi_data, colWidths=[130, 130, 130, 130])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F46E5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#F3F4F6')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 16))
    
    # Analytical Chat Insights
    elements.append(Paragraph("Key Analytical Q&A Insights", heading_style))
    if not chat_history:
        elements.append(Paragraph("<i>No conversation history recorded in this session.</i>", body_style))
    else:
        for item in chat_history:
            role = item.get("role", "User").capitalize()
            content = item.get("content", "")
            if role == "User":
                elements.append(Paragraph(f"<b>User Prompt:</b> {content}", body_style))
            else:
                elements.append(Paragraph(f"<b>AI Insight:</b> {content}", body_style))
                elements.append(Spacer(1, 6))
                
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    logger.info(f"Generated Executive PDF Report ({len(pdf_bytes)} bytes) for {filename}")
    return pdf_bytes
