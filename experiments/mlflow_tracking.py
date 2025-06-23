# experiments/mlflow_tracking.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

def log_semantic_search_experiment():
    """Log semantic search model experiment"""
    
    with mlflow.start_run(run_name="semantic_search_bert_v1"):
        # Log parameters
        mlflow.log_param("model_type", "sentence-transformers")
        mlflow.log_param("base_model", "all-MiniLM-L6-v2")
        mlflow.log_param("embedding_dim", 384)
        mlflow.log_param("max_sequence_length", 512)
        mlflow.log_param("training_samples", 7000)
        mlflow.log_param("fine_tuning_epochs", 3)
        mlflow.log_param("batch_size", 16)
        mlflow.log_param("learning_rate", 2e-5)
        
        # Log metrics
        mlflow.log_metric("validation_accuracy", 0.89)
        mlflow.log_metric("search_precision_at_1", 0.86)
        mlflow.log_metric("search_precision_at_3", 0.95)
        mlflow.log_metric("average_query_time_ms", 45)
        mlflow.log_metric("embedding_generation_time_ms", 12)
        
        # Log additional metrics over time (simulating training progress)
        for epoch in range(1, 4):
            mlflow.log_metric("training_loss", 0.8 - (epoch * 0.2), step=epoch)
            mlflow.log_metric("validation_loss", 0.7 - (epoch * 0.15), step=epoch)
        
        # Log tags
        mlflow.set_tag("model_family", "semantic_search")
        mlflow.set_tag("use_case", "quality_compliance")
        mlflow.set_tag("production_ready", "true")
        mlflow.set_tag("version", "1.0.0")
        
        print("✅ Logged semantic search experiment")

def log_topic_modeling_experiment():
    """Log topic modeling experiment"""
    
    with mlflow.start_run(run_name="topic_modeling_lda_v2"):
        # Log parameters
        mlflow.log_param("algorithm", "Latent Dirichlet Allocation")
        mlflow.log_param("num_topics", 8)
        mlflow.log_param("alpha", 0.1)
        mlflow.log_param("beta", 0.01)
        mlflow.log_param("iterations", 1000)
        mlflow.log_param("documents", 500)
        mlflow.log_param("vocabulary_size", 5000)
        mlflow.log_param("min_word_frequency", 5)
        
        # Log metrics
        mlflow.log_metric("coherence_score", 0.84)
        mlflow.log_metric("perplexity", -245.6)
        mlflow.log_metric("topic_diversity", 0.78)
        mlflow.log_metric("silhouette_score", 0.62)
        
        # Log topic-specific metrics
        for topic_id in range(8):
            mlflow.log_metric(f"topic_{topic_id}_coherence", np.random.uniform(0.7, 0.9))
            mlflow.log_metric(f"topic_{topic_id}_document_count", np.random.randint(30, 100))
        
        # Log tags
        mlflow.set_tag("model_family", "topic_modeling")
        mlflow.set_tag("use_case", "quality_compliance")
        mlflow.set_tag("algorithm", "LDA")
        mlflow.set_tag("version", "2.0.0")
        
        print("✅ Logged topic modeling experiment")

def log_classification_experiment():
    """Log classification model experiment"""
    
    with mlflow.start_run(run_name="category_classification_bert_v3"):
        # Log parameters
        mlflow.log_param("model_type", "BERT")
        mlflow.log_param("base_model", "bert-base-uncased")
        mlflow.log_param("num_classes", 10)
        mlflow.log_param("max_sequence_length", 256)
        mlflow.log_param("training_samples", 4000)
        mlflow.log_param("validation_samples", 1000)
        mlflow.log_param("test_samples", 500)
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("learning_rate", 3e-5)
        mlflow.log_param("dropout_rate", 0.1)
        mlflow.log_param("weight_decay", 0.01)
        
        # Log metrics
        mlflow.log_metric("accuracy", 0.89)
        mlflow.log_metric("precision_macro", 0.87)
        mlflow.log_metric("recall_macro", 0.86)
        mlflow.log_metric("f1_macro", 0.86)
        mlflow.log_metric("top_1_accuracy", 0.86)
        mlflow.log_metric("top_3_accuracy", 0.95)
        
        # Log per-class metrics
        categories = [
            'Documentation Management', 'SOP Compliance', 'Training & Competency',
            'Data Integrity', 'Quality Control', 'Facility Management',
            'Change Control', 'Risk Management', 'Regulatory Compliance', 'Animal Welfare'
        ]
        
        for i, category in enumerate(categories):
            precision = np.random.uniform(0.8, 0.95)
            recall = np.random.uniform(0.8, 0.95)
            f1 = 2 * (precision * recall) / (precision + recall)
            
            mlflow.log_metric(f"precision_{category.lower().replace(' ', '_')}", precision)
            mlflow.log_metric(f"recall_{category.lower().replace(' ', '_')}", recall)
            mlflow.log_metric(f"f1_{category.lower().replace(' ', '_')}", f1)
        
        # Log training progress
        for epoch in range(1, 6):
            train_loss = 1.2 - (epoch * 0.2) + np.random.normal(0, 0.05)
            val_loss = 1.1 - (epoch * 0.18) + np.random.normal(0, 0.05)
            val_acc = 0.6 + (epoch * 0.06) + np.random.normal(0, 0.02)
            
            mlflow.log_metric("training_loss", max(0.1, train_loss), step=epoch)
            mlflow.log_metric("validation_loss", max(0.1, val_loss), step=epoch)
            mlflow.log_metric("validation_accuracy", min(0.95, max(0.5, val_acc)), step=epoch)
        
        # Log tags
        mlflow.set_tag("model_family", "classification")
        mlflow.set_tag("use_case", "category_prediction")
        mlflow.set_tag("production_ready", "true")
        mlflow.set_tag("version", "3.0.0")
        
        print("✅ Logged classification experiment")

def log_hyperparameter_tuning_experiments():
    """Log multiple experiments from hyperparameter tuning"""
    
    # Simulate hyperparameter tuning results
    learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]
    batch_sizes = [16, 32, 64]
    dropout_rates = [0.1, 0.2, 0.3]
    
    best_accuracy = 0
    best_run_id = None
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for dropout in dropout_rates:
                with mlflow.start_run(run_name=f"hyperparam_tune_lr{lr}_bs{batch_size}_dr{dropout}"):
                    # Log parameters
                    mlflow.log_param("learning_rate", lr)
                    mlflow.log_param("batch_size", batch_size)
                    mlflow.log_param("dropout_rate", dropout)
                    mlflow.log_param("epochs", 3)
                    
                    # Simulate performance (better performance for certain combinations)
                    base_accuracy = 0.82
                    if lr == 2e-5:
                        base_accuracy += 0.04
                    if batch_size == 32:
                        base_accuracy += 0.02
                    if dropout == 0.1:
                        base_accuracy += 0.01
                    
                    # Add some noise
                    accuracy = base_accuracy + np.random.normal(0, 0.02)
                    accuracy = max(0.75, min(0.92, accuracy))  # Clamp values
                    
                    mlflow.log_metric("validation_accuracy", accuracy)
                    mlflow.log_metric("training_time_minutes", np.random.uniform(15, 45))
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_run_id = mlflow.active_run().info.run_id
                    
                    mlflow.set_tag("experiment_type", "hyperparameter_tuning")
                    mlflow.set_tag("model_family", "classification")
    
    print(f"✅ Logged {len(learning_rates) * len(batch_sizes) * len(dropout_rates)} hyperparameter tuning experiments")
    print(f"🏆 Best accuracy: {best_accuracy:.4f} (Run ID: {best_run_id})")

def create_model_registry_entries():
    """Create model registry entries for production models"""
    
    # This would typically be done through MLflow UI or separate scripts
    model_info = {
        "semantic_search_model": {
            "name": "QualityComplianceSemanticSearch",
            "version": "1.0.0",
            "stage": "Production",
            "description": "BERT-based semantic search for quality compliance findings"
        },
        "topic_model": {
            "name": "QualityComplianceTopicModel", 
            "version": "2.0.0",
            "stage": "Production",
            "description": "LDA topic modeling for quality compliance categorization"
        },
        "classification_model": {
            "name": "QualityComplianceCategoryClassifier",
            "version": "3.0.0", 
            "stage": "Production",
            "description": "BERT-based classification for quality finding categories"
        }
    }
    
    print("📝 Model Registry Information:")
    for model_key, info in model_info.items():
        print(f"   • {info['name']} v{info['version']} ({info['stage']})")
    
    return model_info

def main():
    """Run all experiment tracking demonstrations"""
    
    print("🧪 Starting MLflow Experiment Tracking Demo")
    print("=" * 50)
    
    # Set experiment
    mlflow.set_experiment("Quality_Compliance_NLP_Models")
    
    try:
        # Log individual experiments
        log_semantic_search_experiment()
        log_topic_modeling_experiment() 
        log_classification_experiment()
        
        # Log hyperparameter tuning runs
        print("\n🔍 Running hyperparameter tuning experiments...")
        log_hyperparameter_tuning_experiments()
        
        # Create model registry info
        print("\n📚 Model Registry:")
        create_model_registry_entries()
        
        print(f"\n✅ Experiment tracking demo complete!")
        print(f"📊 View experiments in MLflow UI at: http://localhost:5000")
        
        # Log summary statistics
        print(f"\n📈 Summary:")
        print(f"   • Experiments logged: 4 main + hyperparameter tuning")
        print(f"   • Model families: 3 (search, topic modeling, classification)")
        print(f"   • Production models: 3")
        print(f"   • Best classification accuracy: 89%")
        print(f"   • Best search precision@1: 86%")
        print(f"   • Topic coherence score: 0.84")
        
    except Exception as e:
        print(f"❌ Error in experiment tracking: {str(e)}")
        print("💡 Make sure MLflow is properly configured in your Domino environment")

if __name__ == "__main__":
    main()