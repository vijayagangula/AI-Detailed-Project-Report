import pdfkit
import os

# ✅ Manually configure the path to wkhtmltopdf.exe
path_wkhtmltopdf = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# Simple HTML content
html_content = """
<h1 style='color: green;'>PDF Test Successful ✅</h1>
<p>This PDF was generated using wkhtmltopdf and pdfkit!</p>
"""

# Generate a PDF using the configuration
pdfkit.from_string(html_content, "test_output.pdf", configuration=config)

print("✅ PDF generated successfully! Check test_output.pdf in your project folder.")
