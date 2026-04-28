import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import argparse
from torch.utils.data import DataLoader

print(f"🔍 PYTORCH SKIN ANALYSIS TOOL")
print("=" * 50)

# Skin conditions
CONDITIONS = ['acne', 'blackheads', 'dark spots', 'dryness', 'normal', 'oily', 'pores', 'wrinkles']

class SkinAnalyzer:
    def __init__(self, model_path='models/fixed_skin_model.pth'):
        """
        Initialize the skin analyzer with a trained model
        """
        self.conditions = CONDITIONS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Load the model
        self.model = self.load_model(model_path)
        print("✅ Model loaded successfully!")
    
    def load_model(self, model_path):
        """Load the trained PyTorch model"""
        try:
            # Import the model architecture
            from FixedSkinModel import FixedSkinModel
            
            # Create model instance
            model = FixedSkinModel(num_classes=len(self.conditions))
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image):
        """
        Preprocess image for model inference
        Supports file path, numpy array, or PIL Image
        """
        try:
            # Handle different input types
            if isinstance(image, str):  # File path
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image not found: {image}")
                image = cv2.imread(image)
                if image is None:
                    raise ValueError(f"Could not load image: {image}")
            
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize and normalize
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            return image_tensor.to(self.device)
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            raise
    
    def analyze_single_image(self, image_path, threshold=0.5):
        """
        Analyze a single skin image
        """
        print(f"\n📷 Analyzing image: {os.path.basename(image_path)}")
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = outputs.cpu().numpy()[0]
            
            # Generate results
            results = {}
            detected_conditions = []
            
            print("\n🔬 Analysis Results:")
            print("-" * 30)
            
            for condition, prob in zip(self.conditions, probabilities):
                results[condition] = {
                    'probability': float(prob),
                    'detected': prob > threshold,
                    'confidence': f"{prob*100:.1f}%"
                }
                
                if prob > threshold:
                    detected_conditions.append(condition)
                    print(f"✅ {condition}: {prob*100:.1f}%")
                else:
                    print(f"❌ {condition}: {prob*100:.1f}%")
            
            # Generate recommendations
            recommendations = self.generate_recommendations(results)
            
            return {
                'image_path': image_path,
                'results': results,
                'detected_conditions': detected_conditions,
                'recommendations': recommendations,
                'skin_type': self.determine_skin_type(results)
            }
            
        except Exception as e:
            print(f"❌ Error analyzing image: {e}")
            return None
    
    def analyze_batch(self, dataset_csv, output_dir='analysis_results'):
        """
        Analyze a batch of images from CSV dataset
        """
        print(f"\n📊 Batch Analysis Started...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        df = pd.read_csv(dataset_csv)
        print(f"📁 Loaded dataset with {len(df)} images")
        
        all_predictions = []
        all_true_labels = []
        analysis_results = []
        
        # Analyze each image
        for idx, row in df.iterrows():
            try:
                image_path = row['image_path']
                
                if not os.path.exists(image_path):
                    print(f"⚠️  Image not found: {image_path}")
                    continue
                
                # Get true labels
                true_labels = [row[condition] for condition in self.conditions]
                
                # Analyze image
                result = self.analyze_single_image(image_path)
                if result:
                    # Get predictions
                    pred_probs = [result['results'][cond]['probability'] for cond in self.conditions]
                    pred_labels = [1 if prob > 0.5 else 0 for prob in pred_probs]
                    
                    all_predictions.append(pred_labels)
                    all_true_labels.append(true_labels)
                    analysis_results.append(result)
                    
                    if (idx + 1) % 10 == 0:
                        print(f"📈 Processed {idx + 1}/{len(df)} images...")
                        
            except Exception as e:
                print(f"❌ Error processing image {idx}: {e}")
                continue
        
        # Generate comprehensive report
        self.generate_comprehensive_report(
            all_true_labels, 
            all_predictions, 
            analysis_results, 
            output_dir
        )
        
        return analysis_results
    
    def generate_recommendations(self, results):
        """
        Generate skincare recommendations based on analysis results
        """
        recommendations = []
        
        # Acne recommendations
        if results['acne']['probability'] > 0.5:
            recommendations.append({
                'condition': 'acne',
                'priority': 'high' if results['acne']['probability'] > 0.7 else 'medium',
                'advice': 'Use salicylic acid or benzoyl peroxide products. Avoid oily cosmetics and maintain proper hygiene.',
                'products': ['Salicylic acid cleanser', 'Benzoyl peroxide spot treatment', 'Oil-free moisturizer']
            })
        
        # Dark spots recommendations
        if results['dark spots']['probability'] > 0.5:
            recommendations.append({
                'condition': 'dark spots',
                'priority': 'high' if results['dark spots']['probability'] > 0.7 else 'medium',
                'advice': 'Use vitamin C serum and daily sunscreen. Consider products with niacinamide.',
                'products': ['Vitamin C serum', 'Broad-spectrum sunscreen SPF 30+', 'Niacinamide treatment']
            })
        
        # Wrinkles recommendations
        if results['wrinkles']['probability'] > 0.5:
            recommendations.append({
                'condition': 'wrinkles',
                'priority': 'high' if results['wrinkles']['probability'] > 0.7 else 'medium',
                'advice': 'Incorporate retinol into your routine. Use hydrating products and always wear sunscreen.',
                'products': ['Retinol serum', 'Peptide moisturizer', 'Hyaluronic acid serum']
            })
        
        # Dryness recommendations
        if results['dryness']['probability'] > 0.5:
            recommendations.append({
                'condition': 'dryness',
                'priority': 'high' if results['dryness']['probability'] > 0.7 else 'medium',
                'advice': 'Focus on hydration. Use ceramide-based products and avoid harsh cleansers.',
                'products': ['Ceramide moisturizer', 'Hyaluronic acid serum', 'Gentle cream cleanser']
            })
        
        # General skincare advice
        recommendations.append({
            'condition': 'general',
            'priority': 'low',
            'advice': 'Maintain a consistent skincare routine, stay hydrated, and protect your skin from sun exposure.',
            'products': ['Gentle cleanser', 'Moisturizer', 'Sunscreen']
        })
        
        return recommendations
    
    def determine_skin_type(self, results):
        """
        Determine overall skin type based on analysis
        """
        skin_scores = {
            'normal': results['normal']['probability'],
            'oily': results['oily']['probability'],
            'dry': results['dryness']['probability'],
            'combination': (results['oily']['probability'] + results['dryness']['probability']) / 2
        }
        
        return max(skin_scores, key=skin_scores.get)
    
    def generate_comprehensive_report(self, true_labels, predictions, analysis_results, output_dir):
        """
        Generate comprehensive analysis report with visualizations
        """
        print(f"\n📈 Generating Comprehensive Report...")
        
        # Convert to numpy arrays
        true_labels = np.array(true_labels)
        predictions = np.array(predictions)
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        # Per-class metrics
        precision = precision_score(true_labels, predictions, average=None, zero_division=0)
        recall = recall_score(true_labels, predictions, average=None, zero_division=0)
        f1 = f1_score(true_labels, predictions, average=None, zero_division=0)
        
        # Overall metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Create visualizations
        self.create_confusion_matrices(true_labels, predictions, output_dir)
        self.create_metrics_chart(precision, recall, f1, output_dir)
        
        # Save detailed report
        self.save_detailed_report(analysis_results, output_dir, accuracy, precision, recall, f1)
        
        print(f"✅ Report saved to: {output_dir}/")
    
    def create_confusion_matrices(self, true_labels, predictions, output_dir):
        """
        Create confusion matrices for each condition
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for i, condition in enumerate(self.conditions):
            cm = confusion_matrix(true_labels[:, i], predictions[:, i])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            axes[i].set_title(f'{condition.title()}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_metrics_chart(self, precision, recall, f1, output_dir):
        """
        Create metrics comparison chart
        """
        x = np.arange(len(self.conditions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Skin Conditions')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance by Skin Condition')
        ax.set_xticks(x)
        ax.set_xticklabels([cond.title() for cond in self.conditions], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_detailed_report(self, analysis_results, output_dir, accuracy, precision, recall, f1):
        """
        Save detailed analysis report to CSV
        """
        report_data = []
        
        for result in analysis_results:
            row = {
                'image_path': result['image_path'],
                'skin_type': result['skin_type'],
                'detected_conditions': ', '.join(result['detected_conditions'])
            }
            
            # Add probabilities for each condition
            for condition in self.conditions:
                row[f'{condition}_prob'] = result['results'][condition]['probability']
                row[f'{condition}_detected'] = result['results'][condition]['detected']
            
            report_data.append(row)
        
        # Create DataFrame and save
        df_report = pd.DataFrame(report_data)
        df_report.to_csv(f'{output_dir}/detailed_analysis_report.csv', index=False)
        
        # Save summary statistics
        with open(f'{output_dir}/summary_statistics.txt', 'w') as f:
            f.write("SKIN ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Images Analyzed: {len(analysis_results)}\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            
            f.write("Per-Condition Performance:\n")
            f.write("-" * 30 + "\n")
            for i, condition in enumerate(self.conditions):
                f.write(f"{condition.title()}:\n")
                f.write(f"  Precision: {precision[i]:.4f}\n")
                f.write(f"  Recall:    {recall[i]:.4f}\n")
                f.write(f"  F1-Score:  {f1[i]:.4f}\n\n")
            
            # Most common conditions
            all_detected = [cond for result in analysis_results for cond in result['detected_conditions']]
            from collections import Counter
            condition_counts = Counter(all_detected)
            
            f.write("Most Common Conditions:\n")
            f.write("-" * 30 + "\n")
            for condition, count in condition_counts.most_common():
                f.write(f"{condition.title()}: {count} occurrences\n")

def main():
    """
    Main function for command line usage
    """
    parser = argparse.ArgumentParser(description='Skin Condition Analyzer')
    parser.add_argument('--image', type=str, help='Path to single image for analysis')
    parser.add_argument('--dataset', type=str, help='Path to dataset CSV for batch analysis')
    parser.add_argument('--model', type=str, default='models/fixed_skin_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='analysis_results',
                       help='Output directory for batch analysis results')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SkinAnalyzer(model_path=args.model)
    
    # Perform analysis based on arguments
    if args.image:
        # Single image analysis
        result = analyzer.analyze_single_image(args.image)
        if result:
            print(f"\n🎯 ANALYSIS COMPLETE!")
            print(f"Skin Type: {result['skin_type'].title()}")
            print(f"Detected Conditions: {', '.join(result['detected_conditions']) if result['detected_conditions'] else 'None'}")
            
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in result['recommendations']:
                print(f"• {rec['advice']}")
    
    elif args.dataset:
        # Batch analysis
        analyzer.analyze_batch(args.dataset, args.output)
        print(f"\n✅ BATCH ANALYSIS COMPLETED!")
        print(f"Results saved to: {args.output}/")
    
    else:
        print("❌ Please provide either --image for single analysis or --dataset for batch analysis")
        print("Usage examples:")
        print("  python pytorch_analyzer.py --image path/to/image.jpg")
        print("  python pytorch_analyzer.py --dataset data/skin_dataset.csv --output my_analysis")

if __name__ == "__main__":
    main()