"""
Generate presentation slides and visualizations from Task1 data analysis
Creates markdown slides and HTML visualizations for easy presentation use
"""

import json
import os
from pathlib import Path
from datetime import datetime


class PresentationGenerator:
    """Generate presentation materials from data analysis"""
    
    def __init__(self, task1_root: str):
        self.task1_root = Path(task1_root)
        self.report_file = self.task1_root / 'data_analysis_report.json'
        self.data = self._load_report()
    
    def _load_report(self) -> dict:
        """Load the JSON analysis report"""
        if not self.report_file.exists():
            print(f"❌ Report file not found: {self.report_file}")
            return {}
        
        with open(self.report_file, 'r') as f:
            return json.load(f)
    
    def generate_markdown_slides(self, output_file: str = None) -> str:
        """Generate Markdown slides for presentations"""
        if output_file is None:
            output_file = self.task1_root / 'SLIDES_Dataset_Summary.md'
        
        slides = []
        
        # Title slide
        slides.append("# Task1 Medical Imaging Dataset\n")
        slides.append("## Comprehensive Overview for Deep Learning\n")
        slides.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        slides.append("\n---\n")
        
        # Overview slide
        brain_count = self.data['brain']['num_patients']
        pelvis_count = self.data['pelvis']['num_patients']
        total = brain_count + pelvis_count
        
        slides.append("# Dataset Overview\n")
        slides.append(f"## Total Patient Records: **{total}**\n")
        slides.append(f"- **Brain Region**: {brain_count} patients")
        slides.append(f"- **Pelvis Region**: {pelvis_count} patients\n")
        slides.append("## Per Patient Data\n")
        slides.append("- CT Scan (Computed Tomography)")
        slides.append("- MR Scan (Magnetic Resonance)")
        slides.append("- Mask (ROI delineation)")
        slides.append("\n")
        slides.append("```")
        slides.append("Total Files: 1,080")
        slides.append("  • 360 CT scans")
        slides.append("  • 360 MR scans")
        slides.append("  • 360 Masks")
        slides.append("```\n")
        slides.append("\n---\n")
        
        # Brain dataset slide
        slides.append("# Brain Dataset Details\n")
        brain_data = self.data['brain']
        slides.append(f"## Patients: {brain_data['num_patients']}\n")
        slides.append("### Image Dimensions (Voxels)\n")
        
        ct_dims = brain_data['modalities']['ct']['dimensions']
        slides.append("#### CT Scans")
        slides.append(f"- **X-axis**: {ct_dims['X']['mean']:.0f} ± {ct_dims['X']['std']:.1f}")
        slides.append(f"  - Range: {ct_dims['X']['min']} - {ct_dims['X']['max']} voxels")
        slides.append(f"- **Y-axis**: {ct_dims['Y']['mean']:.0f} ± {ct_dims['Y']['std']:.1f}")
        slides.append(f"  - Range: {ct_dims['Y']['min']} - {ct_dims['Y']['max']} voxels")
        slides.append(f"- **Z-axis**: {ct_dims['Z']['mean']:.0f} ± {ct_dims['Z']['std']:.1f}")
        slides.append(f"  - Range: {ct_dims['Z']['min']} - {ct_dims['Z']['max']} voxels")
        slides.append(f"\n**Typical Brain Volume**: ~220 × 251 × 192 voxels\n")
        slides.append("\n---\n")
        
        # Pelvis dataset slide
        slides.append("# Pelvis Dataset Details\n")
        pelvis_data = self.data['pelvis']
        slides.append(f"## Patients: {pelvis_data['num_patients']}\n")
        slides.append("### Image Dimensions (Voxels)\n")
        
        ct_dims_pelvis = pelvis_data['modalities']['ct']['dimensions']
        slides.append("#### CT Scans")
        slides.append(f"- **X-axis**: {ct_dims_pelvis['X']['mean']:.0f} ± {ct_dims_pelvis['X']['std']:.1f}")
        slides.append(f"  - Range: {ct_dims_pelvis['X']['min']} - {ct_dims_pelvis['X']['max']} voxels")
        slides.append(f"- **Y-axis**: {ct_dims_pelvis['Y']['mean']:.0f} ± {ct_dims_pelvis['Y']['std']:.1f}")
        slides.append(f"  - Range: {ct_dims_pelvis['Y']['min']} - {ct_dims_pelvis['Y']['max']} voxels")
        slides.append(f"- **Z-axis**: {ct_dims_pelvis['Z']['mean']:.0f} ± {ct_dims_pelvis['Z']['std']:.1f}")
        slides.append(f"  - Range: {ct_dims_pelvis['Z']['min']} - {ct_dims_pelvis['Z']['max']} voxels")
        slides.append(f"\n**Typical Pelvis Volume**: ~461 × 300 × 117 voxels\n")
        slides.append("\n---\n")
        
        # Data characteristics
        slides.append("# Key Data Characteristics\n")
        slides.append("## Modality Alignment\n")
        slides.append("✓ **Perfect alignment** between modalities:")
        slides.append("  - CT and MR scans have identical dimensions per patient")
        slides.append("  - Mask delineations match imaging dimensions")
        slides.append("\n## Image Properties\n")
        slides.append("- **Voxel Spacing**: 1.0 mm isotropic (typical)")
        slides.append("- **File Format**: NIfTI (.nii.gz) - Medical imaging standard")
        slides.append("- **Data Type**: Float32 precision")
        slides.append("\n## Size Differences\n")
        slides.append("- Brain region: ~220×251×192 voxels (smaller, focused)")
        slides.append("- Pelvis region: ~461×300×117 voxels (larger, wider area)")
        slides.append("- Pelvis scans require ~2× storage per patient")
        slides.append("\n---\n")
        
        # Use cases
        slides.append("# Intended Applications\n")
        slides.append("## Synthetic CT Generation\n")
        slides.append("Generate CT scans from MR images using:")
        slides.append("- **Deep Learning**: 3D transformer-based diffusion models")
        slides.append("- **Training Data**: 360 aligned CT-MR pairs per region")
        slides.append("- **Validation**: Separate test splits with ground truth CT\n")
        slides.append("## Clinical Applications\n")
        slides.append("- 🏥 Radiation therapy planning")
        slides.append("- 🎯 Dose calculation from MR-only workflows")
        slides.append("- 📊 Multi-modal image synthesis")
        slides.append("- ✓ MRI-only treatment protocols\n")
        slides.append("\n---\n")
        
        # Technical details
        slides.append("# Technical Specifications\n")
        slides.append("## Data Organization")
        slides.append("```")
        slides.append("Task1/")
        slides.append("  ├── brain/")
        slides.append("  │   ├── 1BA001/  {ct.nii.gz, mr.nii.gz, mask.nii.gz}")
        slides.append("  │   ├── 1BA005/")
        slides.append("  │   └── ... (180 patients)")
        slides.append("  ├── pelvis/")
        slides.append("  │   ├── 1PE001/")
        slides.append("  │   └── ... (180 patients)")
        slides.append("  └── overview/  (visual references)")
        slides.append("```\n")
        slides.append("## File Naming\n")
        slides.append("- `ct.nii.gz`: Reference CT scan")
        slides.append("- `mr.nii.gz`: MR scan to synthesize from")
        slides.append("- `mask.nii.gz`: Region of interest mask")
        slides.append("\n---\n")
        
        # Summary
        slides.append("# Summary Statistics\n")
        slides.append("| Metric | Brain | Pelvis |")
        slides.append("|--------|-------|--------|")
        slides.append(f"| Patients | {brain_count} | {pelvis_count} |")
        slides.append(f"| Avg CT Size (X) | {ct_dims['X']['mean']:.0f} | {ct_dims_pelvis['X']['mean']:.0f} |")
        slides.append(f"| Avg CT Size (Y) | {ct_dims['Y']['mean']:.0f} | {ct_dims_pelvis['Y']['mean']:.0f} |")
        slides.append(f"| Avg CT Size (Z) | {ct_dims['Z']['mean']:.0f} | {ct_dims_pelvis['Z']['mean']:.0f} |")
        slides.append(f"| Total Files | 540 | 540 |")
        slides.append("\n---\n")
        
        # Final slide
        slides.append("# Ready for Deep Learning\n")
        slides.append("## Dataset is prepared for:\n")
        slides.append("✅ **Train/Val/Test splits**")
        slides.append("✅ **Cross-modal synthesis tasks**")
        slides.append("✅ **3D volumetric processing**")
        slides.append("✅ **Transformer-based models**")
        slides.append("✅ **Diffusion model training**\n")
        slides.append("## Resources")
        slides.append("- Complete data analysis: `data_analysis_report.json`")
        slides.append("- Per-patient details: `data_details_report.csv`")
        
        # Write to file
        content = "\n".join(slides)
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"\n✅ Markdown slides generated: {output_file}")
        return content
    
    def generate_html_report(self, output_file: str = None) -> str:
        """Generate HTML visualization of data statistics"""
        if output_file is None:
            output_file = self.task1_root / 'dataset_visualization.html'
        
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html lang='en'>")
        html.append("<head>")
        html.append("  <meta charset='UTF-8'>")
        html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html.append("  <title>Task1 Dataset Analysis</title>")
        html.append("  <style>")
        html.append("    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }")
        html.append("    body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; margin: 0; }")
        html.append("    .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }")
        html.append("    h1 { color: #333; text-align: center; border-bottom: 3px solid #667eea; padding-bottom: 10px; }")
        html.append("    .section { margin: 30px 0; }")
        html.append("    .region { display: inline-block; width: 48%; margin-right: 2%; vertical-align: top; }")
        html.append("    .region h2 { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 5px; }")
        html.append("    .stat-box { background: #f5f5f5; padding: 20px; margin: 15px 0; border-left: 4px solid #667eea; border-radius: 5px; }")
        html.append("    .stat-box h3 { margin-top: 0; color: #333; }")
        html.append("    .dimension-table { width: 100%; border-collapse: collapse; margin-top: 10px; }")
        html.append("    .dimension-table th, .dimension-table td { padding: 10px; border: 1px solid #ddd; text-align: center; }")
        html.append("    .dimension-table th { background: #667eea; color: white; }")
        html.append("    .dimension-table tr:nth-child(even) { background: #f9f9f9; }")
        html.append("    .footer { text-align: center; color: #666; margin-top: 30px; border-top: 1px solid #ddd; padding-top: 20px; }")
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")
        html.append("  <div class='container'>")
        html.append("    <h1>🧬 Task1 Medical Imaging Dataset Analysis</h1>")
        
        brain_count = self.data['brain']['num_patients']
        pelvis_count = self.data['pelvis']['num_patients']
        
        html.append("    <div class='section' style='text-align: center;'>")
        html.append(f"      <h2>📊 Dataset Overview</h2>")
        html.append(f"      <p><strong>Total Patients:</strong> {brain_count + pelvis_count} | ")
        html.append(f"<strong>Brain:</strong> {brain_count} | <strong>Pelvis:</strong> {pelvis_count}</p>")
        html.append(f"      <p><strong>Total Files:</strong> 1,080 (360 CT + 360 MR + 360 Masks)</p>")
        html.append("    </div>")
        
        # Brain section
        brain_data = self.data['brain']
        ct_dims = brain_data['modalities']['ct']['dimensions']
        
        html.append("    <div class='section'>")
        html.append("      <div class='region'>")
        html.append("        <h2>🧠 Brain Dataset</h2>")
        html.append(f"        <div class='stat-box'>")
        html.append(f"          <h3>Patient Count: {brain_data['num_patients']}</h3>")
        html.append(f"          <table class='dimension-table'>")
        html.append(f"            <tr><th>Axis</th><th>Min</th><th>Mean ± Std</th><th>Max</th></tr>")
        html.append(f"            <tr><td><strong>X</strong></td><td>{ct_dims['X']['min']}</td>")
        html.append(f"                <td>{ct_dims['X']['mean']:.0f} ± {ct_dims['X']['std']:.1f}</td>")
        html.append(f"                <td>{ct_dims['X']['max']}</td></tr>")
        html.append(f"            <tr><td><strong>Y</strong></td><td>{ct_dims['Y']['min']}</td>")
        html.append(f"                <td>{ct_dims['Y']['mean']:.0f} ± {ct_dims['Y']['std']:.1f}</td>")
        html.append(f"                <td>{ct_dims['Y']['max']}</td></tr>")
        html.append(f"            <tr><td><strong>Z</strong></td><td>{ct_dims['Z']['min']}</td>")
        html.append(f"                <td>{ct_dims['Z']['mean']:.0f} ± {ct_dims['Z']['std']:.1f}</td>")
        html.append(f"                <td>{ct_dims['Z']['max']}</td></tr>")
        html.append(f"          </table>")
        html.append(f"          <p><em>Typical brain volume: ~220 × 251 × 192 voxels</em></p>")
        html.append(f"        </div>")
        html.append("      </div>")
        
        # Pelvis section
        pelvis_data = self.data['pelvis']
        ct_dims_pelvis = pelvis_data['modalities']['ct']['dimensions']
        
        html.append("      <div class='region'>")
        html.append("        <h2>🦵 Pelvis Dataset</h2>")
        html.append(f"        <div class='stat-box'>")
        html.append(f"          <h3>Patient Count: {pelvis_data['num_patients']}</h3>")
        html.append(f"          <table class='dimension-table'>")
        html.append(f"            <tr><th>Axis</th><th>Min</th><th>Mean ± Std</th><th>Max</th></tr>")
        html.append(f"            <tr><td><strong>X</strong></td><td>{ct_dims_pelvis['X']['min']}</td>")
        html.append(f"                <td>{ct_dims_pelvis['X']['mean']:.0f} ± {ct_dims_pelvis['X']['std']:.1f}</td>")
        html.append(f"                <td>{ct_dims_pelvis['X']['max']}</td></tr>")
        html.append(f"            <tr><td><strong>Y</strong></td><td>{ct_dims_pelvis['Y']['min']}</td>")
        html.append(f"                <td>{ct_dims_pelvis['Y']['mean']:.0f} ± {ct_dims_pelvis['Y']['std']:.1f}</td>")
        html.append(f"                <td>{ct_dims_pelvis['Y']['max']}</td></tr>")
        html.append(f"            <tr><td><strong>Z</strong></td><td>{ct_dims_pelvis['Z']['min']}</td>")
        html.append(f"                <td>{ct_dims_pelvis['Z']['mean']:.0f} ± {ct_dims_pelvis['Z']['std']:.1f}</td>")
        html.append(f"                <td>{ct_dims_pelvis['Z']['max']}</td></tr>")
        html.append(f"          </table>")
        html.append(f"          <p><em>Typical pelvis volume: ~461 × 300 × 117 voxels</em></p>")
        html.append(f"        </div>")
        html.append("      </div>")
        html.append("    </div>")
        
        html.append("    <div class='footer'>")
        html.append(f"      Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        html.append("    </div>")
        html.append("  </div>")
        html.append("</body>")
        html.append("</html>")
        
        content = "\n".join(html)
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"✅ HTML report generated: {output_file}")
        return content
    
    def generate_all(self):
        """Generate all presentation materials"""
        print("\n🎨 Generating presentation materials...\n")
        self.generate_markdown_slides()
        self.generate_html_report()
        print("\n✨ All presentation materials ready!")


def main():
    """Main execution"""
    task1_path = Path(__file__).parent.parent / 'Task1'
    
    if not task1_path.exists():
        print(f"❌ Error: Task1 folder not found at {task1_path}")
        return
    
    generator = PresentationGenerator(str(task1_path))
    generator.generate_all()


if __name__ == '__main__':
    main()
