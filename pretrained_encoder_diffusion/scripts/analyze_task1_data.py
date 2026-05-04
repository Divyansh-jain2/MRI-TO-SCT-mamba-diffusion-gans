"""
Task1 Data Analysis Script
Analyzes brain and pelvis imaging data (CT, MRI, Mask)
Extracts dimensions, statistics, and generates report for presentations
"""

import os
import json
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas as pd


class DataAnalyzer:
    """Analyze medical imaging data structure and extract metadata"""
    
    def __init__(self, task1_root: str):
        self.task1_root = Path(task1_root)
        self.results = {
            'brain': {},
            'pelvis': {}
        }
        self.stats = {
            'brain': {'ct': [], 'mri': [], 'mask': []},
            'pelvis': {'ct': [], 'mri': [], 'mask': []}
        }
    
    def get_nifti_info(self, filepath: str) -> Dict:
        """Extract metadata from NIfTI file (header only, memory efficient)"""
        try:
            nib_file = nib.load(filepath, mmap=False)
            header = nib_file.header
            shape = header.get_data_shape()
            
            return {
                'shape': shape,
                'dtype': str(header.get_data_dtype()),
                'size_gb': os.path.getsize(filepath) / (1024**3),
                'voxel_dims': tuple(header.get_zooms()[:3]),
                'affine': nib_file.affine.tolist() if hasattr(nib_file, 'affine') else None,
                'description': 'NIfTI file (full data analysis skipped for efficiency)'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_modality(self, data_path: str, modality: str) -> Dict:
        """Analyze a specific modality (CT, MRI, or Mask)"""
        # Map modality names to actual filenames
        filename_map = {
            'ct': 'ct.nii.gz',
            'mri': 'mr.nii.gz',  # Note: file is named 'mr' not 'mri'
            'mask': 'mask.nii.gz'
        }
        
        filepath = os.path.join(data_path, filename_map[modality])
        
        if not os.path.exists(filepath):
            return {'error': f'{modality.upper()} file not found'}
        
        info = self.get_nifti_info(filepath)
        info['filepath'] = filepath
        return info
    
    def analyze_folder_type(self, folder_type: str):
        """Analyze all data in brain or pelvis folder"""
        folder_path = self.task1_root / folder_type
        
        if not folder_path.exists():
            print(f"❌ {folder_type.upper()} folder not found at {folder_path}")
            return
        
        print(f"\n📊 Analyzing {folder_type.upper()} data...")
        
        # Get all data folders (excluding overview)
        data_folders = [f for f in os.listdir(folder_path) 
                       if f != 'overview' and 
                       os.path.isdir(os.path.join(folder_path, f))]
        data_folders.sort()
        
        print(f"   Found {len(data_folders)} data folders")
        
        # Analyze each data folder
        for patient_id in tqdm(data_folders, desc=f"Processing {folder_type}"):
            patient_path = folder_path / patient_id
            
            self.results[folder_type][patient_id] = {
                'ct': self.analyze_modality(str(patient_path), 'ct'),
                'mri': self.analyze_modality(str(patient_path), 'mri'),
                'mask': self.analyze_modality(str(patient_path), 'mask')
            }
            
            # Collect statistics
            if 'error' not in self.results[folder_type][patient_id]['ct']:
                self.stats[folder_type]['ct'].append(
                    self.results[folder_type][patient_id]['ct']['shape']
                )
            if 'error' not in self.results[folder_type][patient_id]['mri']:
                self.stats[folder_type]['mri'].append(
                    self.results[folder_type][patient_id]['mri']['shape']
                )
            if 'error' not in self.results[folder_type][patient_id]['mask']:
                self.stats[folder_type]['mask'].append(
                    self.results[folder_type][patient_id]['mask']['shape']
                )
    
    def compute_statistics(self, shapes: List[Tuple]) -> Dict:
        """Compute statistics from list of shapes"""
        if not shapes:
            return {}
        
        shapes_array = np.array(shapes)
        stats = {}
        for i, dim_name in enumerate(['X', 'Y', 'Z']):
            if i < shapes_array.shape[1]:
                dims = shapes_array[:, i]
                stats[dim_name] = {
                    'min': int(np.min(dims)),
                    'max': int(np.max(dims)),
                    'mean': float(np.mean(dims)),
                    'std': float(np.std(dims))
                }
        return stats
    
    def generate_summary(self) -> Dict:
        """Generate comprehensive summary statistics"""
        summary = {}
        
        for folder_type in ['brain', 'pelvis']:
            summary[folder_type] = {
                'num_patients': len(self.results[folder_type]),
                'modalities': {
                    'ct': {
                        'count': len(self.stats[folder_type]['ct']),
                        'dimensions': self.compute_statistics(self.stats[folder_type]['ct'])
                    },
                    'mri': {
                        'count': len(self.stats[folder_type]['mri']),
                        'dimensions': self.compute_statistics(self.stats[folder_type]['mri'])
                    },
                    'mask': {
                        'count': len(self.stats[folder_type]['mask']),
                        'dimensions': self.compute_statistics(self.stats[folder_type]['mask'])
                    }
                }
            }
        
        return summary
    
    def print_report(self):
        """Print formatted report"""
        summary = self.generate_summary()
        
        print("\n" + "="*80)
        print("📋 DATA ANALYSIS REPORT - TASK1 IMAGING DATASET".center(80))
        print("="*80)
        
        for folder_type in ['brain', 'pelvis']:
            print(f"\n{'█' * 80}")
            print(f"  {folder_type.upper()} DATASET")
            print(f"{'█' * 80}")
            
            info = summary[folder_type]
            print(f"\n  📁 Total Patients: {info['num_patients']}")
            
            for modality, mod_info in info['modalities'].items():
                print(f"\n  🔹 {modality.upper()} Scans:")
                print(f"     Count: {mod_info['count']}")
                
                if mod_info['dimensions']:
                    dims = mod_info['dimensions']
                    print(f"     Dimensions (voxels):")
                    for axis, stats in dims.items():
                        print(f"       • {axis}: {stats['min']:4d} - {stats['max']:4d} "
                              f"(mean: {stats['mean']:.1f} ± {stats['std']:.1f})")
        
        print("\n" + "="*80)
    
    def save_json_report(self, output_file: str = None):
        """Save detailed report as JSON"""
        if output_file is None:
            output_file = self.task1_root / 'data_analysis_report.json'
        
        summary = self.generate_summary()
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ JSON report saved to: {output_file}")
    
    def save_csv_report(self, output_file: str = None):
        """Save sample-by-sample details as CSV"""
        if output_file is None:
            output_file = self.task1_root / 'data_details_report.csv'
        
        rows = []
        
        for folder_type in ['brain', 'pelvis']:
            for patient_id, modalities in self.results[folder_type].items():
                row = {'Folder': folder_type, 'PatientID': patient_id}
                
                for modality, info in modalities.items():
                    if 'shape' in info:
                        row[f'{modality}_shape'] = str(info['shape'])
                        row[f'{modality}_size_gb'] = f"{info['size_gb']:.3f}"
                        row[f'{modality}_voxel_dims'] = str(info.get('voxel_dims', 'N/A'))
                    else:
                        row[f'{modality}_shape'] = 'ERROR'
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"✅ CSV report saved to: {output_file}")
    
    def generate_slide_summary(self) -> str:
        """Generate text summary suitable for presentation slides"""
        summary = self.generate_summary()
        
        slide_text = []
        slide_text.append("TASK1 DATASET SUMMARY")
        slide_text.append("=" * 50)
        
        total_patients = sum(summary[ft]['num_patients'] for ft in ['brain', 'pelvis'])
        slide_text.append(f"\n📊 Total Dataset Size:")
        slide_text.append(f"   • Brain: {summary['brain']['num_patients']} patients")
        slide_text.append(f"   • Pelvis: {summary['pelvis']['num_patients']} patients")
        slide_text.append(f"   • TOTAL: {total_patients} patients")
        
        slide_text.append(f"\n🧠 BRAIN DATA:")
        brain_ct_dims = summary['brain']['modalities']['ct']['dimensions']
        if brain_ct_dims:
            slide_text.append(f"   CT Scan - Image Dimensions:")
            for axis in ['X', 'Y', 'Z']:
                d = brain_ct_dims[axis]
                slide_text.append(f"      {axis}-axis: {d['mean']:.0f} ± {d['std']:.0f} voxels")
        
        slide_text.append(f"\n🦵 PELVIS DATA:")
        pelvis_ct_dims = summary['pelvis']['modalities']['ct']['dimensions']
        if pelvis_ct_dims:
            slide_text.append(f"   CT Scan - Image Dimensions:")
            for axis in ['X', 'Y', 'Z']:
                d = pelvis_ct_dims[axis]
                slide_text.append(f"      {axis}-axis: {d['mean']:.0f} ± {d['std']:.0f} voxels")
        
        slide_text.append(f"\n📝 Per Patient:")
        slide_text.append(f"   • 3 Modalities: CT, MR, Mask")
        slide_text.append(f"   • Total files: 540 per folder type")
        
        return "\n".join(slide_text)
    
    def run_analysis(self):
        """Run complete analysis"""
        print("\n🚀 Starting Task1 Data Analysis...\n")
        
        self.analyze_folder_type('brain')
        self.analyze_folder_type('pelvis')
        
        self.print_report()
        self.save_json_report()
        self.save_csv_report()
        
        print("\n" + "="*80)
        print("PRESENTATION SLIDE SUMMARY:")
        print("="*80)
        print(self.generate_slide_summary())
        print("="*80)


def main():
    """Main execution"""
    # Task1 location (same directory as root)
    task1_path = Path(__file__).parent.parent / 'Task1'
    
    if not task1_path.exists():
        print(f"❌ Error: Task1 folder not found at {task1_path}")
        print(f"   Please adjust task1_path in the script")
        return
    
    # Run analysis
    analyzer = DataAnalyzer(str(task1_path))
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
