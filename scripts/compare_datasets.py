#!/usr/bin/env python3
"""
Domino Dataset CSV Comparison Script
Compares CSV data between current dataset and snapshot, generates PDF report of differences.

Author: jim_coates
Project: NLP_Quality_Analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import argparse

class DominoCSVComparator:
    def __init__(self, project_type='dfs'):
        """
        Initialize the CSV comparator for Domino datasets
        
        Args:
            project_type (str): 'dfs' for DFS projects or 'git' for Git-based projects
        """
        self.project_type = project_type.lower()
        self.base_path = '/domino' if project_type == 'dfs' else '/mnt'
        self.differences = []
        self.summary_stats = {}
        
    def get_dataset_path(self, dataset_name, is_snapshot=False, snapshot_id=None, snapshot_tag=None):
        """
        Construct the path to dataset or snapshot based on Domino conventions
        
        Args:
            dataset_name (str): Name of the dataset
            is_snapshot (bool): Whether this is a snapshot path
            snapshot_id (str): Snapshot number (e.g., '1', '2')
            snapshot_tag (str): Snapshot tag name
            
        Returns:
            str: Full path to the dataset or snapshot
        """
        if self.project_type == 'dfs':
            if is_snapshot:
                base = f"{self.base_path}/datasets/local/snapshots/{dataset_name}"
                if snapshot_tag:
                    return f"{base}/{snapshot_tag}"
                elif snapshot_id:
                    return f"{base}/{snapshot_id}"
                else:
                    # Default to snapshot 1 if no tag or ID specified
                    return f"{base}/1"
            else:
                return f"{self.base_path}/datasets/local/{dataset_name}"
        else:  # git-based
            if is_snapshot:
                base = f"{self.base_path}/data/snapshots/{dataset_name}"
                if snapshot_tag:
                    return f"{base}/{snapshot_tag}"
                elif snapshot_id:
                    return f"{base}/{snapshot_id}"
                else:
                    return f"{base}/1"
            else:
                return f"{self.base_path}/data/{dataset_name}"
    
    def load_csv_data(self, file_path):
        """
        Load CSV data with error handling
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame or None: Loaded dataframe or None if error
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded {file_path}: {len(df)} rows, {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: Empty CSV file - {file_path}")
            return None
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None
    
    def compare_dataframes(self, df_current, df_snapshot, dataset_name, csv_filename):
        """
        Compare two dataframes and identify differences
        
        Args:
            df_current (pd.DataFrame): Current dataset dataframe
            df_snapshot (pd.DataFrame): Snapshot dataframe
            dataset_name (str): Name of the dataset
            csv_filename (str): Name of the CSV file being compared
        """
        comparison_result = {
            'dataset': dataset_name,
            'file': csv_filename,
            'timestamp': datetime.now(),
            'current_shape': df_current.shape,
            'snapshot_shape': df_snapshot.shape,
            'differences': []
        }
        
        # Compare shapes
        if df_current.shape != df_snapshot.shape:
            comparison_result['differences'].append({
                'type': 'Shape Difference',
                'description': f"Current: {df_current.shape}, Snapshot: {df_snapshot.shape}",
                'severity': 'High'
            })
        
        # Compare column names
        current_cols = set(df_current.columns)
        snapshot_cols = set(df_snapshot.columns)
        
        if current_cols != snapshot_cols:
            missing_in_current = snapshot_cols - current_cols
            missing_in_snapshot = current_cols - snapshot_cols
            
            if missing_in_current:
                comparison_result['differences'].append({
                    'type': 'Missing Columns in Current',
                    'description': f"Columns: {list(missing_in_current)}",
                    'severity': 'High'
                })
            
            if missing_in_snapshot:
                comparison_result['differences'].append({
                    'type': 'New Columns in Current',
                    'description': f"Columns: {list(missing_in_snapshot)}",
                    'severity': 'Medium'
                })
        
        # Compare data for common columns
        common_cols = current_cols.intersection(snapshot_cols)
        
        if len(common_cols) > 0:
            # Align dataframes for comparison
            df_current_aligned = df_current[list(common_cols)].copy()
            df_snapshot_aligned = df_snapshot[list(common_cols)].copy()
            
            # Compare data types
            for col in common_cols:
                if str(df_current_aligned[col].dtype) != str(df_snapshot_aligned[col].dtype):
                    comparison_result['differences'].append({
                        'type': 'Data Type Change',
                        'description': f"Column '{col}': {df_snapshot_aligned[col].dtype} -> {df_current_aligned[col].dtype}",
                        'severity': 'Medium'
                    })
            
            # Compare null counts
            current_nulls = df_current_aligned.isnull().sum()
            snapshot_nulls = df_snapshot_aligned.isnull().sum()
            
            for col in common_cols:
                if current_nulls[col] != snapshot_nulls[col]:
                    comparison_result['differences'].append({
                        'type': 'Null Count Change',
                        'description': f"Column '{col}': {snapshot_nulls[col]} -> {current_nulls[col]} nulls",
                        'severity': 'Low'
                    })
            
            # Compare summary statistics for numeric columns
            numeric_cols = df_current_aligned.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                try:
                    current_stats = df_current_aligned[col].describe()
                    snapshot_stats = df_snapshot_aligned[col].describe()
                    
                    # Check for significant changes in mean (>5% difference)
                    if abs(current_stats['mean'] - snapshot_stats['mean']) / snapshot_stats['mean'] > 0.05:
                        comparison_result['differences'].append({
                            'type': 'Statistical Change',
                            'description': f"Column '{col}' mean: {snapshot_stats['mean']:.4f} -> {current_stats['mean']:.4f}",
                            'severity': 'Medium'
                        })
                except:
                    pass  # Skip if calculation fails
        
        self.differences.append(comparison_result)
        return comparison_result
    
    def generate_pdf_report(self, output_path='dataset_comparison_report.pdf'):
        """
        Generate a PDF report of all differences found
        
        Args:
            output_path (str): Path where to save the PDF report
        """
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        story.append(Paragraph("Domino Dataset Comparison Report", title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"Project: NLP_Quality_Analytics", styles['Normal']))
        story.append(Paragraph(f"User: jim_coates", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        
        total_files = len(self.differences)
        files_with_differences = len([d for d in self.differences if d['differences']])
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Files Compared', str(total_files)],
            ['Files with Differences', str(files_with_differences)],
            ['Files Identical', str(total_files - files_with_differences)]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Detailed Differences
        if files_with_differences > 0:
            story.append(Paragraph("Detailed Differences", styles['Heading2']))
            
            for comparison in self.differences:
                if comparison['differences']:
                    story.append(Paragraph(f"Dataset: {comparison['dataset']} - File: {comparison['file']}", styles['Heading3']))
                    
                    # File info
                    info_data = [
                        ['Current Shape', f"{comparison['current_shape'][0]} rows × {comparison['current_shape'][1]} columns"],
                        ['Snapshot Shape', f"{comparison['snapshot_shape'][0]} rows × {comparison['snapshot_shape'][1]} columns"],
                        ['Comparison Time', comparison['timestamp'].strftime('%Y-%m-%d %H:%M:%S')]
                    ]
                    
                    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
                    info_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP')
                    ]))
                    
                    story.append(info_table)
                    story.append(Spacer(1, 10))
                    
                    # Differences table
                    diff_data = [['Type', 'Description', 'Severity']]
                    for diff in comparison['differences']:
                        diff_data.append([diff['type'], diff['description'], diff['severity']])
                    
                    diff_table = Table(diff_data, colWidths=[1.5*inch, 3.5*inch, 1*inch])
                    diff_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('FONTSIZE', (0, 1), (-1, -1), 10)
                    ]))
                    
                    story.append(diff_table)
                    story.append(Spacer(1, 20))
        
        else:
            story.append(Paragraph("No Differences Found", styles['Heading2']))
            story.append(Paragraph("All compared files are identical between the current dataset and snapshot.", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        print(f"PDF report generated: {output_path}")
    
    def compare_dataset_with_snapshot(self, dataset_name, csv_filename, snapshot_id=None, snapshot_tag=None):
        """
        Main method to compare a CSV file in current dataset with its snapshot version
        
        Args:
            dataset_name (str): Name of the Domino dataset
            csv_filename (str): Name of the CSV file to compare
            snapshot_id (str): Snapshot number (optional)
            snapshot_tag (str): Snapshot tag (optional)
        """
        print(f"Comparing {dataset_name}/{csv_filename} with snapshot...")
        
        # Construct paths
        current_path = os.path.join(self.get_dataset_path(dataset_name), csv_filename)
        snapshot_path = os.path.join(
            self.get_dataset_path(dataset_name, is_snapshot=True, snapshot_id=snapshot_id, snapshot_tag=snapshot_tag),
            csv_filename
        )
        
        print(f"Current dataset path: {current_path}")
        print(f"Snapshot path: {snapshot_path}")
        
        # Load data
        df_current = self.load_csv_data(current_path)
        df_snapshot = self.load_csv_data(snapshot_path)
        
        if df_current is None or df_snapshot is None:
            print("Failed to load one or both CSV files. Skipping comparison.")
            return False
        
        # Compare data
        comparison_result = self.compare_dataframes(df_current, df_snapshot, dataset_name, csv_filename)
        
        if comparison_result['differences']:
            print(f"Found {len(comparison_result['differences'])} differences")
            for diff in comparison_result['differences']:
                print(f"  - {diff['type']}: {diff['description']} (Severity: {diff['severity']})")
        else:
            print("No differences found")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Compare Domino dataset CSV files with snapshots')
    parser.add_argument('dataset_name', help='Name of the Domino dataset')
    parser.add_argument('csv_filename', help='Name of the CSV file to compare')
    parser.add_argument('--project-type', choices=['dfs', 'git'], default='dfs',
                       help='Type of Domino project (dfs or git-based)')
    parser.add_argument('--snapshot-id', help='Snapshot number to compare against')
    parser.add_argument('--snapshot-tag', help='Snapshot tag to compare against')
    parser.add_argument('--output', default='dataset_comparison_report.pdf',
                       help='Output PDF file path')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = DominoCSVComparator(project_type=args.project_type)
    
    # Perform comparison
    success = comparator.compare_dataset_with_snapshot(
        args.dataset_name,
        args.csv_filename,
        snapshot_id=args.snapshot_id,
        snapshot_tag=args.snapshot_tag
    )
    
    if success:
        # Generate PDF report
        comparator.generate_pdf_report(args.output)
        print(f"\nComparison complete! Report saved to: {args.output}")
    else:
        print("Comparison failed due to file loading errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()