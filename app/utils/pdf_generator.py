"""
PDF Generator for Loan Eligibility Reports
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import os

def generate_loan_report_pdf(user_inputs, prediction, probability, shap_explanation, output_dir='app/static/reports'):
    """
    Generate a professional PDF report for loan eligibility
    
    Args:
        user_inputs: Dictionary of user input features
        prediction: Prediction (0 or 1)
        probability: Prediction probability
        shap_explanation: SHAP explanation dictionary
        output_dir: Directory to save PDF
    
    Returns:
        Path to generated PDF file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'loan_report_{timestamp}.pdf'
    filepath = os.path.join(output_dir, filename)
    
    doc = SimpleDocTemplate(filepath, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#283593'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    
    # Title
    elements.append(Paragraph("Loan Eligibility Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Report Date
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_RIGHT,
        textColor=colors.grey
    )
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", date_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Eligibility Result
    result_color = colors.HexColor('#2e7d32') if prediction == 1 else colors.HexColor('#c62828')
    result_text = "ELIGIBLE" if prediction == 1 else "NOT ELIGIBLE"
    
    result_style = ParagraphStyle(
        'ResultStyle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=result_color,
        alignment=TA_CENTER,
        spaceAfter=20,
        fontName='Helvetica-Bold'
    )
    
    elements.append(Paragraph(f"Loan Status: {result_text}", result_style))
    elements.append(Paragraph(f"Confidence: {probability:.1%}", 
                              ParagraphStyle('ProbStyle', parent=normal_style, 
                                            alignment=TA_CENTER, fontSize=12)))
    elements.append(Spacer(1, 0.3*inch))
    
    # User Inputs Section
    elements.append(Paragraph("Applicant Information", heading_style))
    
    user_data = [
        ['Field', 'Value'],
        ['Age', f"{user_inputs['person_age']:.0f} years"],
        ['Education', user_inputs['person_education']],
        ['Annual Income', f"${user_inputs['person_income']:,.2f}"],
        ['Employment Experience', f"{user_inputs['person_emp_exp']} years"],
        ['Home Ownership', user_inputs['person_home_ownership']],
        ['Loan Amount', f"${user_inputs['loan_amnt']:,.2f}"],
        ['Loan Intent', user_inputs['loan_intent']],
        ['Loan Percent of Income', f"{user_inputs['loan_percent_income']:.1%}"],
        ['Credit History Length', f"{user_inputs['cb_person_cred_hist_length']:.1f} years"],
        ['Credit Score', f"{user_inputs['credit_score']}"],
        ['Previous Defaults', user_inputs['previous_loan_defaults_on_file']],
    ]
    
    user_table = Table(user_data, colWidths=[3*inch, 4*inch])
    user_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    
    elements.append(user_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # SHAP Explanations Section
    elements.append(Paragraph("Key Factors Influencing Decision", heading_style))
    elements.append(Paragraph(
        "The following factors had the most significant impact on the loan eligibility decision:",
        normal_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    # Top features from SHAP
    top_features = shap_explanation.get('top_features', [])[:10]
    
    shap_data = [['Feature', 'Impact', 'Contribution']]
    if top_features:
        for feat in top_features:
            feature_name = feat.get('feature', 'Unknown').replace('_', ' ').title()
            impact = feat.get('impact', 'Neutral').title()
            contribution = f"{feat.get('shap_value', 0):+.4f}"
            shap_data.append([feature_name, impact, contribution])
    else:
        # If no SHAP features, add a message
        shap_data.append(['N/A', 'SHAP explanation not available', 'N/A'])
    
    shap_table = Table(shap_data, colWidths=[3*inch, 2*inch, 2*inch])
    shap_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    
    elements.append(shap_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Recommendations Section
    elements.append(Paragraph("Recommendations", heading_style))
    
    if prediction == 0:
        recommendations = [
            "Consider improving your credit score before reapplying",
            "Reduce the loan amount relative to your income",
            "Build a longer credit history",
            "Ensure stable employment history",
            "Consider reducing existing debt obligations"
        ]
    else:
        recommendations = [
            "Maintain your current credit score",
            "Continue making timely payments",
            "Keep your debt-to-income ratio low",
            "Build emergency savings",
            "Review loan terms carefully before accepting"
        ]
    
    for i, rec in enumerate(recommendations, 1):
        elements.append(Paragraph(f"{i}. {rec}", normal_style))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Footer
    footer_style = ParagraphStyle(
        'FooterStyle',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.grey
    )
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph(
        "This report is generated by an AI-powered loan eligibility system. "
        "This is a pre-screening tool and does not guarantee loan approval.",
        footer_style
    ))
    
    # Build PDF
    doc.build(elements)
    
    print(f"PDF report generated: {filepath}")
    return filepath



