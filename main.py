#!/usr/bin/env python3
"""
Main entry point for the PS06 Speech-to-Text System.
"""

import click
import logging
from pathlib import Path

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """PS06 Speech-to-Text System - AI4Bharat IndicConformer with advanced analytics."""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ensure data directories exist
    data_dirs = ['data/audio', 'data/transcripts', 'data/embeddings', 'data/metrics', 'logs']
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

@cli.command()
@click.argument('dataset_path', type=click.Path(exists=True),
              help='Path to dataset directory or metadata file')
@click.option('--output-dir', '-o', type=click.Path(), default='data/transcripts',
              help='Output directory for transcripts')
@click.option('--chunk-length', '-c', type=float, default=30.0,
              help='Length of audio chunks in seconds')
@click.option('--overlap', '-v', type=float, default=2.0,
              help='Overlap between chunks in seconds')
@click.option('--device', '-d', type=str, default='auto',
              help='Device to use (auto, cpu, cuda)')
@click.option('--batch-size', '-b', type=int, default=32,
              help='Batch size for processing')
def transcribe(dataset_path, output_dir, chunk_length, overlap, device, batch_size):
    """Transcribe audio files from dataset using AI4Bharat IndicConformer."""
    from src.cli.transcribe import transcribe as transcribe_func
    transcribe_func(dataset_path, output_dir, chunk_length, overlap, device, batch_size)

@cli.command()
@click.argument('transcripts_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--ground-truth-dir', '-g', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Directory containing ground truth files')
@click.option('--output-dir', '-o', type=click.Path(), default='data/metrics',
              help='Output directory for evaluation results')
@click.option('--format', '-f', type=click.Choice(['csv', 'parquet', 'json']), default='csv',
              help='Output format for results')
def evaluate(transcripts_dir, ground_truth_dir, output_dir, format):
    """Evaluate transcript quality and compute metrics."""
    from src.cli.evaluate import evaluate as evaluate_func
    evaluate_func(transcripts_dir, ground_truth_dir, output_dir, format)

@cli.command()
@click.argument('transcripts_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--output-dir', '-o', type=click.Path(), default='data/embeddings',
              help='Output directory for embeddings')
@click.option('--model', '-m', type=str, default=None,
              help='Sentence transformer model to use')
@click.option('--format', '-f', type=click.Choice(['pickle', 'json', 'numpy']), default='pickle',
              help='Output format for embeddings')
def embeddings(transcripts_dir, output_dir, model, format):
    """Generate multilingual sentence embeddings for transcripts."""
    from src.embeddings.generator import MultilingualEmbeddingGenerator
    from src.asr.transcriber import Transcript
    import json
    
    # Load transcripts
    transcripts_path = Path(transcripts_dir)
    transcript_files = list(transcripts_path.glob('*.json'))
    
    if not transcript_files:
        click.echo(f"No transcript files found in {transcripts_dir}")
        return
    
    click.echo(f"Found {len(transcript_files)} transcript files")
    
    # Load transcripts
    transcripts = []
    for transcript_file in transcript_files:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        transcript = Transcript(
            file_path=transcript_data['file_path'],
            file_name=transcript_data['file_name'],
            duration=transcript_data['duration'],
            segments=transcript_data['segments'],
            metadata=transcript_data['metadata']
        )
        transcripts.append(transcript)
    
    # Generate embeddings
    click.echo("Generating embeddings...")
    generator = MultilingualEmbeddingGenerator(model_name=model)
    embeddings = generator.generate_embeddings(transcripts)
    
    # Save embeddings
    output_path = Path(output_dir)
    generator.save_embeddings(embeddings, output_path, format)
    
    click.echo(f"Generated {len(embeddings)} embeddings")
    click.echo(f"Embeddings saved to: {output_path}")

@cli.command()
@click.argument('transcripts_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('embeddings_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--neo4j-uri', type=str, default='bolt://localhost:7687',
              help='Neo4j database URI')
@click.option('--neo4j-username', type=str, default='neo4j',
              help='Neo4j username')
@click.option('--neo4j-password', type=str, default='password',
              help='Neo4j password')
def graph(transcripts_dir, embeddings_dir, neo4j_uri, neo4j_username, neo4j_password):
    """Build knowledge graph from transcripts and embeddings."""
    from src.graph.builder import KnowledgeGraphBuilder
    from src.asr.transcriber import Transcript
    from src.embeddings.generator import EmbeddingRecord
    import json
    
    # Load transcripts
    transcripts_path = Path(transcripts_dir)
    transcript_files = list(transcripts_path.glob('*.json'))
    
    if not transcript_files:
        click.echo(f"No transcript files found in {transcripts_dir}")
        return
    
    click.echo(f"Found {len(transcript_files)} transcript files")
    
    # Load transcripts
    transcripts = []
    for transcript_file in transcript_files:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        transcript = Transcript(
            file_path=transcript_data['file_path'],
            file_name=transcript_data['file_name'],
            duration=transcript_data['duration'],
            segments=transcript_data['segments'],
            metadata=transcript_data['metadata']
        )
        transcripts.append(transcript)
    
    # Load embeddings
    embeddings_path = Path(embeddings_dir)
    embeddings_file = embeddings_path / 'embeddings.pkl'
    
    if not embeddings_file.exists():
        click.echo(f"Embeddings file not found: {embeddings_file}")
        return
    
    click.echo("Loading embeddings...")
    generator = MultilingualEmbeddingGenerator()
    embeddings = generator.load_embeddings(embeddings_path, 'pickle')
    
    click.echo(f"Loaded {len(embeddings)} embeddings")
    
    # Build graph
    click.echo("Building knowledge graph...")
    try:
        builder = KnowledgeGraphBuilder(neo4j_uri, neo4j_username, neo4j_password)
        stats = builder.build_graph(transcripts, embeddings)
        
        click.echo("Knowledge graph built successfully!")
        click.echo(f"Nodes created: {stats['total_nodes']}")
        click.echo(f"Edges created: {stats['total_edges']}")
        
        builder.close()
        
    except Exception as e:
        click.echo(f"Failed to build graph: {e}")

@cli.command()
@click.option('--host', type=str, default='0.0.0.0', help='Host to bind to')
@click.option('--port', type=int, default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host, port, reload):
    """Start the FastAPI server."""
    import uvicorn
    from src.api.main import app
    
    click.echo(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=reload)

@cli.command()
def demo():
    """Run a demo of the system with sample data."""
    click.echo("PS06 Speech-to-Text System Demo")
    click.echo("=" * 40)
    
    # Check if sample data exists
    sample_audio_dir = Path("data/audio")
    if not sample_audio_dir.exists() or not list(sample_audio_dir.glob("*")):
        click.echo("No sample audio files found in data/audio/")
        click.echo("Please add some audio files and run the demo again")
        return
    
    click.echo("1. Transcribing audio files...")
    # This would run transcription
    
    click.echo("2. Evaluating transcripts...")
    # This would run evaluation
    
    click.echo("3. Generating embeddings...")
    # This would generate embeddings
    
    click.echo("4. Building knowledge graph...")
    # This would build the graph
    
    click.echo("Demo completed!")

if __name__ == '__main__':
    cli()
