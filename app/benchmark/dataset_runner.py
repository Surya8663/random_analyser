"""
Benchmark runner for evaluating the pipeline on document datasets
"""
import asyncio
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics
from tqdm import tqdm
import numpy as np

from app.core.models import MultiModalDocument
from app.services.document_processor import DocumentProcessor
from app.agents.orchestrator import Phase3Orchestrator
from app.eval.evaluator import DocumentEvaluator
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class DatasetBenchmarkRunner:
    """Run benchmark evaluation on document datasets"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.document_processor = DocumentProcessor()
        self.orchestrator = Phase3Orchestrator()
        self.evaluator = DocumentEvaluator()
        
        self.results = {}
        self.summary_stats = {}
    
    async def run_benchmark(self, 
                          dataset_dir: str,
                          ground_truth_dir: Optional[str] = None,
                          max_documents: Optional[int] = None) -> Dict[str, Any]:
        """
        Run benchmark on a dataset of documents
        
        Args:
            dataset_dir: Directory containing document files
            ground_truth_dir: Optional directory with ground truth JSON files
            max_documents: Maximum number of documents to process
        """
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        
        # Get all document files
        document_files = []
        for ext in ['.pdf', '.png', '.jpg', '.jpeg', '.txt']:
            document_files.extend(list(dataset_path.glob(f"*{ext}")))
        
        if max_documents:
            document_files = document_files[:max_documents]
        
        logger.info(f"📊 Starting benchmark on {len(document_files)} documents")
        
        # Process each document
        results = {}
        for i, doc_file in enumerate(tqdm(document_files, desc="Processing documents")):
            try:
                doc_id = f"benchmark_{doc_file.stem}_{datetime.now().strftime('%H%M%S')}"
                
                # Load ground truth if available
                ground_truth_file = None
                if ground_truth_dir:
                    gt_path = Path(ground_truth_dir) / f"{doc_file.stem}.json"
                    if gt_path.exists():
                        ground_truth_file = str(gt_path)
                
                # Run full pipeline
                result = await self._process_single_document(
                    str(doc_file), 
                    doc_id,
                    ground_truth_file
                )
                
                results[doc_id] = {
                    "document_file": str(doc_file),
                    "ground_truth_file": ground_truth_file,
                    "result": result,
                    "success": result.get("success", False),
                    "processing_time": result.get("processing_time", 0),
                    "risk_score": result.get("risk_score", 0),
                    "extracted_fields": len(result.get("extracted_fields", {})),
                    "contradictions": len(result.get("contradictions", [])),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Save intermediate result
                if (i + 1) % 5 == 0:
                    self._save_intermediate_results(results)
                
            except Exception as e:
                logger.error(f"Failed to process {doc_file}: {e}")
                results[f"error_{doc_file.stem}"] = {
                    "document_file": str(doc_file),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        # Generate summary statistics
        self.results = results
        self.summary_stats = self._generate_summary_statistics(results)
        
        # Save all results
        self._save_results(results, self.summary_stats)
        
        return {
            "total_documents": len(document_files),
            "successful_documents": sum(1 for r in results.values() if r.get("success")),
            "failed_documents": sum(1 for r in results.values() if "error" in r),
            "summary_statistics": self.summary_stats,
            "results": results
        }
    
    async def _process_single_document(self, 
                                     file_path: str, 
                                     document_id: str,
                                     ground_truth_file: Optional[str] = None) -> Dict[str, Any]:
        """Process a single document through the full pipeline"""
        start_time = datetime.now()
        
        try:
            # Step 1: Process with DocumentProcessor
            base_document = await self.document_processor.process_document(file_path, document_id)
            
            # Step 2: Run through Phase 3 orchestrator
            result = await self.orchestrator.process_document(base_document)
            
            # Step 3: Evaluate if ground truth exists
            evaluation_report = None
            if ground_truth_file and Path(ground_truth_file).exists():
                evaluation_report = await self.evaluator.evaluate_document(
                    base_document, 
                    ground_truth_file
                )
            
            # Extract agent performance data
            agent_performance = {}
            if hasattr(base_document, 'processing_metadata') and base_document.processing_metadata:
                if "reasoning" in base_document.processing_metadata:
                    reasoning_data = base_document.processing_metadata["reasoning"]
                    agent_performance = {
                        "validation_score": reasoning_data.get("validation_results", {}).get("score", 0),
                        "confidence_scores": reasoning_data.get("confidence_scores", {}),
                        "risk_assessment": reasoning_data.get("risk_assessment", {})
                    }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": result.get("success", False),
                "processing_time": processing_time,
                "document_type": result.get("document_type", "unknown"),
                "risk_score": result.get("risk_score", 0),
                "risk_level": result.get("risk_level", "LOW"),
                "contradictions": result.get("contradictions_count", 0),
                "extracted_fields": result.get("extracted_fields_count", 0),
                "agent_performance": agent_performance,
                "evaluation_report": evaluation_report.dict() if evaluation_report else None,
                "errors": result.get("errors", []),
                "recommendations": result.get("recommendations", [])
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed for {file_path}: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": False,
                "processing_time": processing_time,
                "error": str(e),
                "errors": [str(e)],
                "risk_score": 0.5
            }
    
    def _generate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        successful_results = [r for r in results.values() if r.get("success")]
        
        if not successful_results:
            return {"error": "No successful document processing"}
        
        # Basic statistics
        processing_times = [r.get("processing_time", 0) for r in successful_results]
        risk_scores = [r.get("risk_score", 0) for r in successful_results]
        extracted_fields_counts = [r.get("extracted_fields", 0) for r in successful_results]
        contradictions_counts = [r.get("contradictions", 0) for r in successful_results]
        
        # Agent performance statistics
        validation_scores = []
        confidence_scores = []
        
        for result in successful_results:
            perf = result.get("agent_performance", {})
            if perf:
                validation_scores.append(perf.get("validation_score", 0))
                if "confidence_scores" in perf:
                    conf = perf["confidence_scores"]
                    confidence_scores.append(conf.get("overall", 0))
        
        stats = {
            "basic": {
                "total_documents": len(results),
                "successful_documents": len(successful_results),
                "success_rate": len(successful_results) / len(results) if results else 0,
                "average_processing_time": statistics.mean(processing_times) if processing_times else 0,
                "std_processing_time": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                "min_processing_time": min(processing_times) if processing_times else 0,
                "max_processing_time": max(processing_times) if processing_times else 0,
                "average_risk_score": statistics.mean(risk_scores) if risk_scores else 0,
                "average_extracted_fields": statistics.mean(extracted_fields_counts) if extracted_fields_counts else 0,
                "average_contradictions": statistics.mean(contradictions_counts) if contradictions_counts else 0,
            },
            "agent_performance": {
                "average_validation_score": statistics.mean(validation_scores) if validation_scores else 0,
                "average_confidence": statistics.mean(confidence_scores) if confidence_scores else 0,
                "successful_agents_ratio": self._calculate_agent_success_ratio(successful_results)
            },
            "risk_distribution": self._calculate_risk_distribution(successful_results),
            "document_type_distribution": self._calculate_document_type_distribution(successful_results)
        }
        
        # Add evaluation metrics if available
        evaluation_metrics = self._aggregate_evaluation_metrics(successful_results)
        if evaluation_metrics:
            stats["evaluation_metrics"] = evaluation_metrics
        
        return stats
    
    def _calculate_agent_success_ratio(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate success ratios for different agents"""
        agent_counts = {
            "vision": 0,
            "text": 0,
            "fusion": 0,
            "reasoning": 0,
            "explainability": 0
        }
        
        for result in results:
            # Check for errors in each agent phase
            errors = result.get("errors", [])
            if not any("Vision agent error" in err for err in errors):
                agent_counts["vision"] += 1
            if not any("Text agent error" in err for err in errors):
                agent_counts["text"] += 1
            if not any("Fusion agent error" in err for err in errors):
                agent_counts["fusion"] += 1
            if not any("Reasoning agent error" in err for err in errors):
                agent_counts["reasoning"] += 1
            if not any("Explainability" in err for err in errors):
                agent_counts["explainability"] += 1
        
        total = len(results)
        return {agent: count / total for agent, count in agent_counts.items()}
    
    def _calculate_risk_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """Calculate risk score distribution"""
        distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        
        for result in results:
            risk_score = result.get("risk_score", 0)
            if risk_score > 0.7:
                distribution["CRITICAL"] += 1
            elif risk_score > 0.5:
                distribution["HIGH"] += 1
            elif risk_score > 0.3:
                distribution["MEDIUM"] += 1
            else:
                distribution["LOW"] += 1
        
        return distribution
    
    def _calculate_document_type_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """Calculate document type distribution"""
        distribution = {}
        
        for result in results:
            doc_type = result.get("document_type", "unknown")
            distribution[doc_type] = distribution.get(doc_type, 0) + 1
        
        return distribution
    
    def _aggregate_evaluation_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate evaluation metrics from ground truth comparisons"""
        all_metrics = []
        
        for result in results:
            eval_report = result.get("evaluation_report")
            if eval_report and "metrics" in eval_report:
                all_metrics.append(eval_report["metrics"])
        
        if not all_metrics:
            return {}
        
        # Aggregate metrics
        aggregated = {}
        metric_fields = [
            "field_precision", "field_recall", "field_f1", "field_coverage",
            "entity_precision", "entity_recall", "entity_f1",
            "alignment_accuracy", "cross_modal_consistency",
            "risk_detection_precision", "risk_detection_recall",
            "overall_accuracy", "processing_success_rate", "average_confidence"
        ]
        
        for field in metric_fields:
            values = [m.get(field, 0) for m in all_metrics if field in m]
            if values:
                aggregated[f"average_{field}"] = statistics.mean(values)
                aggregated[f"std_{field}"] = statistics.stdev(values) if len(values) > 1 else 0
                aggregated[f"min_{field}"] = min(values)
                aggregated[f"max_{field}"] = max(values)
        
        return aggregated
    
    def _save_intermediate_results(self, results: Dict[str, Any]):
        """Save intermediate results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"intermediate_results_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _save_results(self, results: Dict[str, Any], summary_stats: Dict[str, Any]):
        """Save all benchmark results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "total_documents": len(results),
                "summary_statistics": summary_stats,
                "detailed_results": results
            }, f, indent=2, default=str)
        
        # Save summary as CSV
        csv_file = self.output_dir / f"benchmark_summary_{timestamp}.csv"
        self._save_summary_csv(summary_stats, csv_file)
        
        logger.info(f"💾 Saved benchmark results to {results_file}")
        logger.info(f"📊 Saved summary to {csv_file}")
    
    def _save_summary_csv(self, summary_stats: Dict[str, Any], csv_path: Path):
        """Save summary statistics as CSV"""
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Basic stats
                writer.writerow(["Metric", "Value"])
                writer.writerow(["total_documents", summary_stats["basic"]["total_documents"]])
                writer.writerow(["successful_documents", summary_stats["basic"]["successful_documents"]])
                writer.writerow(["success_rate", f"{summary_stats['basic']['success_rate']:.2%}"])
                writer.writerow(["average_processing_time", f"{summary_stats['basic']['average_processing_time']:.2f}s"])
                writer.writerow(["average_risk_score", f"{summary_stats['basic']['average_risk_score']:.3f}"])
                writer.writerow(["average_extracted_fields", f"{summary_stats['basic']['average_extracted_fields']:.1f}"])
                writer.writerow(["average_contradictions", f"{summary_stats['basic']['average_contradictions']:.1f}"])
                
                # Agent success rates
                writer.writerow([])
                writer.writerow(["Agent Success Rates"])
                writer.writerow(["Agent", "Success Rate"])
                for agent, rate in summary_stats["agent_performance"]["successful_agents_ratio"].items():
                    writer.writerow([agent, f"{rate:.2%}"])
                
                # Risk distribution
                writer.writerow([])
                writer.writerow(["Risk Distribution"])
                writer.writerow(["Risk Level", "Count"])
                for level, count in summary_stats["risk_distribution"].items():
                    writer.writerow([level, count])
                
                # Document type distribution
                writer.writerow([])
                writer.writerow(["Document Type Distribution"])
                writer.writerow(["Document Type", "Count"])
                for doc_type, count in summary_stats["document_type_distribution"].items():
                    writer.writerow([doc_type, count])
                
                # Evaluation metrics if available
                if "evaluation_metrics" in summary_stats:
                    writer.writerow([])
                    writer.writerow(["Evaluation Metrics"])
                    writer.writerow(["Metric", "Average", "Std Dev", "Min", "Max"])
                    
                    for key, value in summary_stats["evaluation_metrics"].items():
                        if key.startswith("average_"):
                            metric_name = key.replace("average_", "")
                            avg = value
                            std = summary_stats["evaluation_metrics"].get(f"std_{metric_name}", 0)
                            min_val = summary_stats["evaluation_metrics"].get(f"min_{metric_name}", 0)
                            max_val = summary_stats["evaluation_metrics"].get(f"max_{metric_name}", 0)
                            
                            writer.writerow([
                                metric_name,
                                f"{avg:.3f}",
                                f"{std:.3f}",
                                f"{min_val:.3f}",
                                f"{max_val:.3f}"
                            ])
        
        except Exception as e:
            logger.error(f"Failed to save CSV summary: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive benchmark report"""
        return {
            "benchmark_summary": self.summary_stats,
            "total_results": len(self.results),
            "generated_at": datetime.now().isoformat(),
            "output_directory": str(self.output_dir.absolute())
        }


async def run_benchmark_cli():
    """CLI entry point for benchmark runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run benchmark evaluation on document dataset")
    parser.add_argument("dataset_dir", help="Directory containing documents")
    parser.add_argument("--ground-truth-dir", help="Directory with ground truth JSON files")
    parser.add_argument("--max-documents", type=int, help="Maximum number of documents to process")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    
    args = parser.parse_args()
    
    runner = DatasetBenchmarkRunner(args.output_dir)
    results = await runner.run_benchmark(
        args.dataset_dir,
        args.ground_truth_dir,
        args.max_documents
    )
    
    print(f"✅ Benchmark completed:")
    print(f"   Total documents: {results['total_documents']}")
    print(f"   Successful: {results['successful_documents']}")
    print(f"   Failed: {results['failed_documents']}")
    print(f"   Results saved to: {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(run_benchmark_cli())