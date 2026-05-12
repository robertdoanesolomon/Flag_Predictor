#!/usr/bin/env python3
"""
Display training diagnostics from the last ISIS model training run.
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_training_diagnostics(location='isis', models_dir='models'):
    """Load and display training diagnostics."""
    models_path = Path(models_dir)
    config_path = models_path / f'config_{location}_latest.pkl'
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return None
    
    # Load config
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    return config

def display_diagnostics(config):
    """Display training diagnostics in a formatted way."""
    print("=" * 80)
    print("ISIS MODEL TRAINING DIAGNOSTICS")
    print("=" * 80)
    
    # Basic model info
    print("\n📊 MODEL CONFIGURATION:")
    print(f"  Location: {config.get('location', 'N/A')}")
    print(f"  Sequence length: {config.get('sequence_length', 'N/A')}")
    print(f"  Input size (features): {config.get('input_size', 'N/A')}")
    print(f"  Hidden sizes: {config.get('hidden_sizes', 'N/A')}")
    print(f"  Dropout rate: {config.get('dropout_rate', 'N/A')}")
    print(f"  Number of horizons: {len(config.get('horizons', []))}")
    print(f"  Horizons: {config.get('horizons', [])}")
    
    # Training history
    history = config.get('training_history', {})
    if history:
        print("\n📈 TRAINING HISTORY:")
        print(f"  Total epochs: {len(history.get('train_loss', []))}")
        
        if history.get('train_loss'):
            train_losses = history['train_loss']
            val_losses = history['val_loss']
            train_maes = history['train_mae']
            val_maes = history['val_mae']
            
            print(f"\n  Final Training Loss: {train_losses[-1]:.6f}")
            print(f"  Final Validation Loss: {val_losses[-1]:.6f}")
            print(f"  Final Training MAE: {train_maes[-1]:.6f}")
            print(f"  Final Validation MAE: {val_maes[-1]:.6f}")
            
            print(f"\n  Best Validation Loss: {min(val_losses):.6f} (epoch {val_losses.index(min(val_losses)) + 1})")
            print(f"  Best Validation MAE: {min(val_maes):.6f} (epoch {val_maes.index(min(val_maes)) + 1})")
            
            print(f"\n  Initial Training Loss: {train_losses[0]:.6f}")
            print(f"  Initial Validation Loss: {val_losses[0]:.6f}")
            
            if len(train_losses) > 1:
                loss_improvement = ((train_losses[0] - train_losses[-1]) / train_losses[0]) * 100
                val_loss_improvement = ((val_losses[0] - val_losses[-1]) / val_losses[0]) * 100
                print(f"\n  Training Loss Improvement: {loss_improvement:.2f}%")
                print(f"  Validation Loss Improvement: {val_loss_improvement:.2f}%")
            
            # Create visualization
            print("\n📊 Creating training curves plot...")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            epochs = range(1, len(train_losses) + 1)
            
            # Loss curves
            axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
            axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # MAE curves
            axes[0, 1].plot(epochs, train_maes, 'b-', label='Training MAE', linewidth=2)
            axes[0, 1].plot(epochs, val_maes, 'r-', label='Validation MAE', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE (meters)')
            axes[0, 1].set_title('Training and Validation MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Loss improvement over time
            train_loss_diff = [train_losses[0] - loss for loss in train_losses]
            val_loss_diff = [val_losses[0] - loss for loss in val_losses]
            axes[1, 0].plot(epochs, train_loss_diff, 'b-', label='Training Loss Reduction', linewidth=2)
            axes[1, 0].plot(epochs, val_loss_diff, 'r-', label='Validation Loss Reduction', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Reduction')
            axes[1, 0].set_title('Loss Improvement Over Training')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Loss ratio (val/train) - overfitting indicator
            if train_losses[-1] > 0:
                loss_ratio = [v/t if t > 0 else 0 for v, t in zip(val_losses, train_losses)]
                axes[1, 1].plot(epochs, loss_ratio, 'g-', linewidth=2)
                axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal (val=train)')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Validation / Training Loss')
                axes[1, 1].set_title('Overfitting Indicator (val/train ratio)')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = Path('training_diagnostics_isis.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved plot to: {output_path}")
            plt.close()
            
            # Print epoch-by-epoch summary (last 10 epochs)
            print("\n📋 LAST 10 EPOCHS SUMMARY:")
            print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Train MAE':<15} {'Val MAE':<15}")
            print("-" * 80)
            for i in range(max(0, len(epochs) - 10), len(epochs)):
                print(f"{epochs[i]:<8} {train_losses[i]:<15.6f} {val_losses[i]:<15.6f} "
                      f"{train_maes[i]:<15.6f} {val_maes[i]:<15.6f}")
    else:
        print("\n⚠️  No training history found in config file")
    
    # Feature information
    feature_columns = config.get('feature_columns', [])
    if feature_columns:
        print(f"\n🔧 FEATURES ({len(feature_columns)} total):")
        print(f"  First 10: {feature_columns[:10]}")
        if len(feature_columns) > 10:
            print(f"  ... and {len(feature_columns) - 10} more")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    config = load_training_diagnostics(location='isis')
    if config:
        display_diagnostics(config)
    else:
        print("Failed to load diagnostics")
