# experiments/mlflow_tracking.py
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import tempfile
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def create_dummy_model():
    """Create a dummy model for demonstration purposes"""
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions for metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test, y_pred

def log_semantic_search_experiment():
    """Log semantic search model experiment with model artifacts"""
    
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
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("warmup_steps", 100)
        mlflow.log_param("weight_decay", 0.01)
        mlflow.log_param("gradient_clipping", 1.0)
        mlflow.log_param("scheduler", "linear")
        
        # Log metrics
        mlflow.log_metric("validation_accuracy", 0.89)
        mlflow.log_metric("search_precision_at_1", 0.86)
        mlflow.log_metric("search_precision_at_3", 0.95)
        mlflow.log_metric("search_precision_at_5", 0.98)
        mlflow.log_metric("search_recall_at_1", 0.86)
        mlflow.log_metric("search_recall_at_3", 0.92)
        mlflow.log_metric("search_recall_at_5", 0.96)
        mlflow.log_metric("average_query_time_ms", 45)
        mlflow.log_metric("embedding_generation_time_ms", 12)
        mlflow.log_metric("model_size_mb", 90.5)
        mlflow.log_metric("inference_throughput_qps", 150)
        mlflow.log_metric("mean_reciprocal_rank", 0.91)
        mlflow.log_metric("normalized_dcg", 0.94)
        
        # Log additional metrics over time (simulating training progress)
        for epoch in range(1, 4):
            mlflow.log_metric("training_loss", 0.8 - (epoch * 0.2), step=epoch)
            mlflow.log_metric("validation_loss", 0.7 - (epoch * 0.15), step=epoch)
            mlflow.log_metric("learning_rate", 2e-5 * (0.9 ** epoch), step=epoch)
            mlflow.log_metric("gradient_norm", 0.5 + np.random.normal(0, 0.1), step=epoch)
        
        # Log hyperparameters as artifacts
        hyperparams = {
            "model_architecture": "BERT-based encoder",
            "pooling_strategy": "mean_pooling",
            "similarity_metric": "cosine",
            "temperature": 0.07,
            "negative_samples": 5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(hyperparams, f, indent=2)
            mlflow.log_artifact(f.name, "config")
        
        # Create and log a dummy model for demonstration
        model, accuracy, _, _, _ = create_dummy_model()
        mlflow.sklearn.log_model(model, "semantic_search_model", 
                                registered_model_name="QualityComplianceSemanticSearch")
        
        # Log tags
        mlflow.set_tag("model_family", "semantic_search")
        mlflow.set_tag("use_case", "quality_compliance")
        mlflow.set_tag("production_ready", "true")
        mlflow.set_tag("version", "1.0.0")
        mlflow.set_tag("framework", "sentence-transformers")
        mlflow.set_tag("domain", "quality_assurance")
        mlflow.set_tag("data_source", "compliance_documents")
        mlflow.set_tag("model_stage", "production")
        
        print("✅ Logged semantic search experiment with model")

def log_topic_modeling_experiment():
    """Log topic modeling experiment with model artifacts"""
    
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
        mlflow.log_param("max_word_frequency", 0.95)
        mlflow.log_param("ngram_range", "1-2")
        mlflow.log_param("remove_stopwords", True)
        mlflow.log_param("lemmatization", True)
        mlflow.log_param("random_state", 42)
        
        # Log metrics
        mlflow.log_metric("coherence_score", 0.84)
        mlflow.log_metric("perplexity", -245.6)
        mlflow.log_metric("topic_diversity", 0.78)
        mlflow.log_metric("silhouette_score", 0.62)
        mlflow.log_metric("umass_coherence", 0.79)
        mlflow.log_metric("cv_coherence", 0.86)
        mlflow.log_metric("model_convergence", 0.95)
        mlflow.log_metric("training_time_minutes", 23.5)
        
        # Log topic-specific metrics
        topic_names = [
            "documentation_mgmt", "sop_compliance", "training_competency",
            "data_integrity", "quality_control", "facility_mgmt",
            "change_control", "risk_mgmt"
        ]
        
        for i, topic_name in enumerate(topic_names):
            mlflow.log_metric(f"topic_{i}_coherence", np.random.uniform(0.7, 0.9))
            mlflow.log_metric(f"topic_{i}_document_count", np.random.randint(30, 100))
            mlflow.log_metric(f"topic_{i}_avg_confidence", np.random.uniform(0.6, 0.9))
        
        # Create and log model
        model, accuracy, _, _, _ = create_dummy_model()
        mlflow.sklearn.log_model(model, "topic_model", 
                                registered_model_name="QualityComplianceTopicModel")
        
        # Log topic words as artifact
        topic_words = {f"topic_{i}": [f"word_{j}" for j in range(10)] for i in range(8)}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(topic_words, f, indent=2)
            mlflow.log_artifact(f.name, "topics")
        
        # Log tags
        mlflow.set_tag("model_family", "topic_modeling")
        mlflow.set_tag("use_case", "quality_compliance")
        mlflow.set_tag("algorithm", "LDA")
        mlflow.set_tag("version", "2.0.0")
        mlflow.set_tag("framework", "gensim")
        mlflow.set_tag("preprocessing", "nltk")
        mlflow.set_tag("model_stage", "production")
        
        print("✅ Logged topic modeling experiment with model")

def log_classification_experiment():
    """Log classification model experiment with fixed metric names"""
    
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
        mlflow.log_param("warmup_ratio", 0.1)
        mlflow.log_param("gradient_accumulation_steps", 1)
        mlflow.log_param("label_smoothing", 0.1)
        mlflow.log_param("early_stopping_patience", 3)
        
        # Log metrics
        mlflow.log_metric("accuracy", 0.89)
        mlflow.log_metric("precision_macro", 0.87)
        mlflow.log_metric("recall_macro", 0.86)
        mlflow.log_metric("f1_macro", 0.86)
        mlflow.log_metric("f1_weighted", 0.88)
        mlflow.log_metric("top_1_accuracy", 0.86)
        mlflow.log_metric("top_3_accuracy", 0.95)
        mlflow.log_metric("top_5_accuracy", 0.98)
        mlflow.log_metric("roc_auc_macro", 0.93)
        mlflow.log_metric("roc_auc_weighted", 0.94)
        mlflow.log_metric("log_loss", 0.34)
        mlflow.log_metric("inference_time_ms", 23.4)
        mlflow.log_metric("model_parameters", 110000000)
        
        # Fixed category names (removed invalid characters)
        categories = [
            'Documentation_Management', 'SOP_Compliance', 'Training_Competency',
            'Data_Integrity', 'Quality_Control', 'Facility_Management',
            'Change_Control', 'Risk_Management', 'Regulatory_Compliance', 'Animal_Welfare'
        ]
        
        # Log per-class metrics with valid names
        for i, category in enumerate(categories):
            precision = np.random.uniform(0.8, 0.95)
            recall = np.random.uniform(0.8, 0.95)
            f1 = 2 * (precision * recall) / (precision + recall)
            support = np.random.randint(30, 120)
            
            # Use valid metric names (no special characters except underscore, dash, period)
            safe_category = category.lower().replace(' ', '_').replace('&', 'and')
            mlflow.log_metric(f"precision_{safe_category}", precision)
            mlflow.log_metric(f"recall_{safe_category}", recall)
            mlflow.log_metric(f"f1_{safe_category}", f1)
            mlflow.log_metric(f"support_{safe_category}", support)
        
        # Log training progress with more detailed metrics
        for epoch in range(1, 6):
            train_loss = 1.2 - (epoch * 0.2) + np.random.normal(0, 0.05)
            val_loss = 1.1 - (epoch * 0.18) + np.random.normal(0, 0.05)
            val_acc = 0.6 + (epoch * 0.06) + np.random.normal(0, 0.02)
            lr = 3e-5 * (0.95 ** epoch)
            
            mlflow.log_metric("training_loss", max(0.1, train_loss), step=epoch)
            mlflow.log_metric("validation_loss", max(0.1, val_loss), step=epoch)
            mlflow.log_metric("validation_accuracy", min(0.95, max(0.5, val_acc)), step=epoch)
            mlflow.log_metric("learning_rate_epoch", lr, step=epoch)
            mlflow.log_metric("gradient_norm", np.random.uniform(0.1, 2.0), step=epoch)
            mlflow.log_metric("training_time_minutes", epoch * 15.2, step=epoch)
        
        # Create and log model
        model, model_accuracy, X_test, y_test, y_pred = create_dummy_model()
        mlflow.sklearn.log_model(model, "classification_model", 
                                registered_model_name="QualityComplianceCategoryClassifier")
        
        # Log confusion matrix as artifact
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            cm_df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "metrics")
        
        # Log feature importance
        feature_importance = {f"feature_{i}": float(importance) 
                            for i, importance in enumerate(model.feature_importances_[:10])}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(feature_importance, f, indent=2)
            mlflow.log_artifact(f.name, "model_analysis")
        
        # Log tags
        mlflow.set_tag("model_family", "classification")
        mlflow.set_tag("use_case", "category_prediction")
        mlflow.set_tag("production_ready", "true")
        mlflow.set_tag("version", "3.0.0")
        mlflow.set_tag("framework", "transformers")
        mlflow.set_tag("model_architecture", "BERT")
        mlflow.set_tag("fine_tuned", "true")
        mlflow.set_tag("data_augmentation", "false")
        mlflow.set_tag("model_stage", "production")
        
        print("✅ Logged classification experiment with model")

def log_hyperparameter_tuning_experiments():
    """Log multiple experiments from hyperparameter tuning with models"""
    
    # Simulate hyperparameter tuning results
    learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]
    batch_sizes = [16, 32, 64]
    dropout_rates = [0.1, 0.2, 0.3]
    
    best_accuracy = 0
    best_run_id = None
    experiment_count = 0
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for dropout in dropout_rates:
                experiment_count += 1
                run_name = f"hyperparam_tune_lr{lr}_bs{batch_size}_dr{dropout}"
                
                with mlflow.start_run(run_name=run_name):
                    # Log parameters
                    mlflow.log_param("learning_rate", lr)
                    mlflow.log_param("batch_size", batch_size)
                    mlflow.log_param("dropout_rate", dropout)
                    mlflow.log_param("epochs", 3)
                    mlflow.log_param("optimizer", "AdamW")
                    mlflow.log_param("weight_decay", 0.01)
                    mlflow.log_param("warmup_steps", 100)
                    
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
                    accuracy = max(0.75, min(0.92, accuracy))
                    
                    # Log metrics
                    mlflow.log_metric("validation_accuracy", accuracy)
                    mlflow.log_metric("validation_loss", 1.2 - accuracy + np.random.normal(0, 0.1))
                    mlflow.log_metric("training_time_minutes", np.random.uniform(15, 45))
                    mlflow.log_metric("memory_usage_gb", np.random.uniform(4, 12))
                    mlflow.log_metric("gpu_utilization", np.random.uniform(0.7, 0.95))
                    
                    # Log training curves
                    for epoch in range(1, 4):
                        train_acc = accuracy - 0.05 + (epoch * 0.02)
                        mlflow.log_metric("training_accuracy", train_acc, step=epoch)
                        mlflow.log_metric("epoch_time_minutes", np.random.uniform(5, 15), step=epoch)
                    
                    # Create and log model for best runs
                    if accuracy > 0.85:
                        model, _, _, _, _ = create_dummy_model()
                        model_name = f"hyperparameter_tuning_model_{experiment_count}"
                        mlflow.sklearn.log_model(model, "tuned_model")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_run_id = mlflow.active_run().info.run_id
                    
                    # Log tags
                    mlflow.set_tag("experiment_type", "hyperparameter_tuning")
                    mlflow.set_tag("model_family", "classification")
                    mlflow.set_tag("tuning_method", "grid_search")
                    mlflow.set_tag("optimization_metric", "validation_accuracy")
                    mlflow.set_tag("version", f"tune_v{experiment_count}")
    
    print(f"✅ Logged {len(learning_rates) * len(batch_sizes) * len(dropout_rates)} hyperparameter tuning experiments")
    print(f"🏆 Best accuracy: {best_accuracy:.4f} (Run ID: {best_run_id})")

def create_model_registry_entries():
    """Create model registry entries for production models"""
    
    model_info = {
        "semantic_search_model": {
            "name": "QualityComplianceSemanticSearch",
            "version": "1.0.0",
            "stage": "Production",
            "description": "BERT-based semantic search for quality compliance findings",
            "use_case": "Document similarity and search",
            "performance": "86% precision@1, 95% precision@3"
        },
        "topic_model": {
            "name": "QualityComplianceTopicModel", 
            "version": "2.0.0",
            "stage": "Production",
            "description": "LDA topic modeling for quality compliance categorization",
            "use_case": "Document topic discovery and clustering",
            "performance": "0.84 coherence score, 8 topics identified"
        },
        "classification_model": {
            "name": "QualityComplianceCategoryClassifier",
            "version": "3.0.0", 
            "stage": "Production",
            "description": "BERT-based classification for quality finding categories",
            "use_case": "Automated category assignment",
            "performance": "89% accuracy, 86% F1-macro"
        }
    }
    
    print("📝 Model Registry Information:")
    for model_key, info in model_info.items():
        print(f"   • {info['name']} v{info['version']} ({info['stage']})")
        print(f"     - {info['description']}")
        print(f"     - Performance: {info['performance']}")
    
    return model_info

def main():
    """Run all experiment tracking demonstrations"""
    
    print("🧪 Starting MLflow Experiment Tracking Demo")
    print("=" * 50)
    
    # Set experiment
    experiment_name = "Quality_Compliance_NLP_Models"
    mlflow.set_experiment(experiment_name)
    
    try:
        # Log individual experiments
        print("\n🔍 Logging semantic search experiment...")
        log_semantic_search_experiment()
        
        print("\n📊 Logging topic modeling experiment...")
        log_topic_modeling_experiment() 
        
        print("\n🎯 Logging classification experiment...")
        log_classification_experiment()
        
        # Log hyperparameter tuning runs
        print("\n⚙️ Running hyperparameter tuning experiments...")
        log_hyperparameter_tuning_experiments()
        
        # Create model registry info
        print("\n📚 Model Registry Summary:")
        model_info = create_model_registry_entries()
        
        print(f"\n✅ Experiment tracking demo complete!")
        print(f"📊 View experiments in MLflow UI")
        
        # Log summary statistics
        print(f"\n📈 Summary:")
        print(f"   • Experiment: {experiment_name}")
        print(f"   • Main experiments logged: 3")
        print(f"   • Hyperparameter tuning runs: {4*3*3}")
        print(f"   • Model families: 3 (search, topic modeling, classification)")
        print(f"   • Models registered: 3")
        print(f"   • Best classification accuracy: 89%")
        print(f"   • Best search precision@1: 86%")
        print(f"   • Topic coherence score: 0.84")
        print(f"   • All models logged with artifacts and ready for registration")
        
    except Exception as e:
        print(f"❌ Error in experiment tracking: {str(e)}")
        print("💡 Check MLflow configuration and parameter names")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()