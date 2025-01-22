"""
Command-line interface for Flag Predictor.

Usage:
    python -m flag_predictor train --location isis
    python -m flag_predictor forecast --location isis --ensemble
    python -m flag_predictor forecast --location godstow
"""

import argparse
import sys
from pathlib import Path

from .pipeline import train_location_model, run_forecast
from .config import LOCATIONS, TRAINING_CONFIG


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Flag Predictor - River Differential Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Train a model for ISIS lock:
        python -m flag_predictor train --location isis

    Run ensemble forecast for Godstow:
        python -m flag_predictor forecast --location godstow --ensemble

    Run single forecast:
        python -m flag_predictor forecast --location isis
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument(
        '--location', '-l',
        choices=list(LOCATIONS.keys()),
        default='isis',
        help='Location to train for (default: isis)'
    )
    train_parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=TRAINING_CONFIG['epochs'],
        help=f'Number of training epochs (default: {TRAINING_CONFIG["epochs"]})'
    )
    train_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=64,
        help='Training batch size (default: 64)'
    )
    train_parser.add_argument(
        '--output-dir', '-o',
        default='models',
        help='Directory to save model (default: models)'
    )
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Run a forecast')
    forecast_parser.add_argument(
        '--location', '-l',
        choices=list(LOCATIONS.keys()),
        default='isis',
        help='Location to forecast (default: isis)'
    )
    forecast_parser.add_argument(
        '--ensemble', '-e',
        action='store_true',
        help='Use ensemble prediction (recommended)'
    )
    forecast_parser.add_argument(
        '--members', '-m',
        type=int,
        default=20,
        help='Number of ensemble members (default: 20)'
    )
    forecast_parser.add_argument(
        '--model-dir',
        default='models',
        help='Directory containing trained model (default: models)'
    )
    forecast_parser.add_argument(
        '--output', '-o',
        help='Output file for predictions (CSV)'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'train':
        print(f"\n🚀 Training model for {args.location.upper()}...")
        model, config = train_location_model(
            location=args.location,
            save_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=True
        )
        print(f"\n✓ Model saved to {args.output_dir}/")
    
    elif args.command == 'forecast':
        print(f"\n🔮 Running forecast for {args.location.upper()}...")
        results = run_forecast(
            location=args.location,
            model_path=args.model_dir,
            ensemble=args.ensemble,
            n_members=args.members,
            verbose=True
        )
        
        if args.ensemble:
            print(f"\n📊 Forecast Summary:")
            stats = results['statistics']
            print(f"  Mean prediction range: {stats['mean'].min():.3f}m to {stats['mean'].max():.3f}m")
            print(f"  Uncertainty (std): {stats['std'].mean():.3f}m average")
            
            flag_probs = results['flag_probabilities']
            print(f"\n🚩 Flag Probability at end of forecast:")
            if 'p_green' in flag_probs.columns:
                print(f"  Green:      {flag_probs['p_green'].iloc[-1]*100:.1f}%")
                print(f"  Light Blue: {flag_probs['p_light_blue'].iloc[-1]*100:.1f}%")
                print(f"  Dark Blue:  {flag_probs['p_dark_blue'].iloc[-1]*100:.1f}%")
                print(f"  Amber:      {flag_probs['p_amber'].iloc[-1]*100:.1f}%")
                print(f"  Red:        {flag_probs['p_red'].iloc[-1]*100:.1f}%")
            else:
                # Fallback for old format (shouldn't happen with new code)
                print(f"  Blue:   {flag_probs['p_blue'].iloc[-1]*100:.1f}%")
                print(f"  Yellow: {flag_probs['p_yellow'].iloc[-1]*100:.1f}%")
                print(f"  Orange: {flag_probs['p_orange'].iloc[-1]*100:.1f}%")
                print(f"  Red:    {flag_probs['p_red'].iloc[-1]*100:.1f}%")
            
            if args.output:
                stats.to_csv(args.output)
                print(f"\n✓ Statistics saved to {args.output}")
        else:
            prediction = results['prediction']
            print(f"\n📊 Forecast Summary:")
            print(f"  Prediction range: {prediction.min():.3f}m to {prediction.max():.3f}m")
            
            if args.output:
                prediction.to_csv(args.output, header=['differential'])
                print(f"\n✓ Prediction saved to {args.output}")


if __name__ == '__main__':
    main()
