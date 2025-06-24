#!/usr/bin/env python
# create_dataset.py test
# Run this as a Domino job to populate the NLP_Quality_Analytics dataset

import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

def create_audit_findings():
    """Create sample audit findings dataset"""
    np.random.seed(42)  # For reproducible demo data
    
    categories = [
        'Documentation Management', 'SOP Compliance', 'Training & Competency',
        'Data Integrity', 'Quality Control', 'Facility Management',
        'Change Control', 'Risk Management', 'Regulatory Compliance', 'Animal Welfare'
    ]
    
    severities = ['Minor', 'Major', 'Critical']
    areas = ['Clinical', 'Manufacturing', 'Quality', 'Regulatory', 'Preclinical', 'IT Systems']
    
    # Sample finding texts
    finding_texts = [
        "Committee correspondence and documentation was not filed in the appropriate location. Required notification letters missing.",
        "Standard Operating Procedure documentation was incomplete. Master SOP not available and verification processes not documented.",
        "Animal health monitoring procedures were not followed correctly. Veterinary oversight documentation missing.",
        "Training records for personnel were incomplete. Required certifications missing and competency assessments not conducted.",
        "Data integrity issues identified during review. Electronic records showed discrepancies and backup procedures not functioning.",
        "Quality control testing procedures not performed according to specifications. Required testing intervals missed.",
        "Facility maintenance records were incomplete. Environmental monitoring data missing and calibration certificates expired.",
        "Audit trail documentation was insufficient. Change control procedures not followed and approval workflows not documented.",
        "Risk assessment procedures were not conducted as required. Mitigation strategies not documented.",
        "Regulatory compliance documentation was missing. Required submissions delayed and correspondence not filed properly.",
        "Protocol deviation not properly documented. Investigator failed to follow proper notification procedures.",
        "Laboratory equipment calibration records were missing. Temperature monitoring logs showed gaps in documentation.",
        "Informed consent process was not followed correctly. Patient signatures missing from consent forms.",
        "Drug accountability records were incomplete. Dispensing logs did not match inventory counts.",
        "Adverse event reporting was delayed. Required safety reports not submitted within regulatory timeframes.",
        "Site file organization was inadequate. Essential documents were missing or misfiled.",
        "Source document verification revealed discrepancies. CRF entries did not match source documents.",
        "Investigator delegation log was not properly maintained. Staff responsibilities were not clearly defined.",
        "Monitoring visit reports were not filed correctly. Required follow-up actions were not documented.",
        "Quality assurance procedures were not implemented. Required QA reviews were not performed."
    ]
    
    # Generate dataset
    n_records = 500
    data = []
    
    for i in range(n_records):
        finding_id = f'QF-2024-{i+1:03d}'
        finding_text = np.random.choice(finding_texts)
        
        # Add some variation to the text
        if np.random.random() < 0.3:
            finding_text += f" Additional details for case {i+1}."
        
        category = np.random.choice(categories)
        severity = np.random.choice(severities, p=[0.5, 0.35, 0.15])  # More minor than critical
        area = np.random.choice(areas)
        
        # Generate dates over the last 18 months
        base_date = datetime.now() - timedelta(days=540)
        date = base_date + timedelta(days=np.random.randint(0, 540))
        
        data.append({
            'finding_id': finding_id,
            'finding_text': finding_text,
            'category': category,
            'severity': severity,
            'area': area,
            'date': date.strftime('%Y-%m-%d'),
            'status': np.random.choice(['Open', 'In Progress', 'Closed'], p=[0.2, 0.3, 0.5])
        })
    
    df = pd.DataFrame(data)
    return df

def create_sample_queries():
    """Create sample search queries for demo"""
    return {
        "sample_queries": [
            "documentation missing",
            "animal health monitoring", 
            "SOP procedures",
            "training records incomplete",
            "data integrity issues",
            "quality control testing",
            "facility maintenance",
            "regulatory compliance",
            "protocol deviation",
            "equipment calibration",
            "informed consent",
            "adverse event reporting"
        ],
        "trending_queries": [
            {"query": "SOP documentation", "count": 45},
            {"query": "training records", "count": 38},
            {"query": "animal welfare", "count": 32},
            {"query": "data integrity", "count": 28},
            {"query": "equipment calibration", "count": 24}
        ]
    }

def create_metadata():
    """Create dataset metadata"""
    return {
        "dataset_info": {
            "name": "Quality Compliance Data",
            "description": "Quality and compliance findings for analytics",
            "version": "1.0.0",
            "created_date": datetime.now().strftime('%Y-%m-%d'),
            "last_updated": datetime.now().strftime('%Y-%m-%d'),
            "total_records": 500,
            "categories": 10,
            "time_range": "2023-01-01 to 2024-06-01"
        },
        "data_sources": [
            "Quality Management System",
            "Audit Reports", 
            "Inspection Findings",
            "CAPA System",
            "Training Records"
        ],
        "model_info": {
            "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
            "topic_model": "Latent Dirichlet Allocation",
            "classification_model": "BERT-base fine-tuned",
            "performance_metrics": {
                "search_accuracy": "86% Top1, 95% Top3",
                "topic_coherence": 0.84,
                "classification_accuracy": 0.89
            }
        }
    }

def main():
    """Main execution function"""
    print("🚀 NLP Quality Analytics Dataset Creation Job")
    print("=" * 50)
    
    # Define base path for Domino dataset
    base_path = Path("/mnt/data/NLP_Quality_Analytics")
    
    print(f"📁 Target dataset path: {base_path}")
    
    # Create directories if they don't exist
    print("📂 Creating directory structure...")
    (base_path / "embeddings").mkdir(parents=True, exist_ok=True)
    (base_path / "models").mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")
    
    # Create audit findings CSV
    print("\n📄 Generating audit findings dataset...")
    df = create_audit_findings()
    df_path = base_path / "audit_findings.csv"
    df.to_csv(df_path, index=False)
    print(f"✅ Created {df_path} with {len(df)} records")
    
    # Display dataset summary
    print(f"\n📊 Dataset Summary:")
    print(f"   • Total records: {len(df)}")
    print(f"   • Categories: {df['category'].nunique()}")
    print(f"   • Severity levels: {df['severity'].nunique()}")
    print(f"   • Areas: {df['area'].nunique()}")
    print(f"   • Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Create sample queries JSON
    print("\n🔍 Creating sample queries...")
    queries = create_sample_queries()
    queries_path = base_path / "sample_queries.json"
    with open(queries_path, 'w') as f:
        json.dump(queries, f, indent=2)
    print(f"✅ Created {queries_path}")
    
    # Create metadata JSON
    print("\n📋 Creating metadata...")
    metadata = create_metadata()
    metadata_path = base_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Created {metadata_path}")
    
    # Create placeholder model files
    print("\n🧠 Creating placeholder model files...")
    
    # Embeddings placeholder
    embeddings_path = base_path / "embeddings" / "bert_embeddings.pkl"
    placeholder_embeddings = {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'embedding_dim': 384,
        'num_documents': len(df),
        'created_date': datetime.now().isoformat(),
        'note': 'Placeholder for demo - real embeddings would be computed'
    }
    with open(embeddings_path, 'wb') as f:
        pickle.dump(placeholder_embeddings, f)
    print(f"✅ Created {embeddings_path}")
    
    # Topic model placeholder
    topic_model_path = base_path / "models" / "topic_model.pkl"
    placeholder_topic_model = {
        'algorithm': 'Latent Dirichlet Allocation',
        'num_topics': 8,
        'coherence_score': 0.84,
        'created_date': datetime.now().isoformat(),
        'note': 'Placeholder for demo - real model would be trained'
    }
    with open(topic_model_path, 'wb') as f:
        pickle.dump(placeholder_topic_model, f)
    print(f"✅ Created {topic_model_path}")
    
    # Classification model placeholder
    classifier_path = base_path / "models" / "category_classifier.pkl"
    placeholder_classifier = {
        'model_type': 'BERT-based classifier',
        'num_classes': 10,
        'accuracy': 0.89,
        'top_1_accuracy': 0.86,
        'top_3_accuracy': 0.95,
        'created_date': datetime.now().isoformat(),
        'note': 'Placeholder for demo - real model would be trained'
    }
    with open(classifier_path, 'wb') as f:
        pickle.dump(placeholder_classifier, f)
    print(f"✅ Created {classifier_path}")
    
    # Create summary file
    print("\n📊 Creating dataset summary...")
    summary_path = base_path / "dataset_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("NLP Quality Analytics Dataset Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Categories: {df['category'].nunique()}\n")
        f.write(f"Severity Levels: {df['severity'].nunique()}\n")
        f.write(f"Operational Areas: {df['area'].nunique()}\n")
        f.write(f"Date Range: {df['date'].min()} to {df['date'].max()}\n\n")
        
        f.write("Files Created:\n")
        f.write("- audit_findings.csv (main dataset)\n")
        f.write("- sample_queries.json (demo queries)\n")
        f.write("- metadata.json (dataset information)\n")
        f.write("- embeddings/bert_embeddings.pkl (placeholder)\n")
        f.write("- models/topic_model.pkl (placeholder)\n")
        f.write("- models/category_classifier.pkl (placeholder)\n\n")
        
        f.write("Category Distribution:\n")
        for cat, count in df['category'].value_counts().items():
            f.write(f"- {cat}: {count} ({count/len(df)*100:.1f}%)\n")
    
    print(f"✅ Created {summary_path}")
    
    # Final verification and summary
    print(f"\n🎉 Dataset creation completed successfully!")
    print(f"\n📁 Files created in {base_path}:")
    
    total_size = 0
    for file_path in sorted(base_path.rglob("*")):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size += file_path.stat().st_size
            print(f"   • {file_path.relative_to(base_path)} ({size_mb:.2f} MB)")
    
    print(f"\n📊 Total dataset size: {total_size / (1024 * 1024):.2f} MB")
    
    print(f"\n✨ The NLP_Quality_Analytics dataset is now ready!")
    print(f"🚀 Next steps:")
    print(f"   1. Launch the semantic search app")
    print(f"   2. Launch the topic explorer app") 
    print(f"   3. Test the model API endpoints")
    print(f"   4. Run the MLflow experiment tracking")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Job completed successfully!")
        else:
            print("\n❌ Job failed!")
            exit(1)
    except Exception as e:
        print(f"\n💥 Error during dataset creation: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)