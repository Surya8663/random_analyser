"""
Advanced metrics aggregation and statistical analysis
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import scipy.stats as stats

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class MetricsAggregator:
    """Advanced metrics aggregation and statistical analysis"""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_data = {}
        self.aggregated_metrics = {}
        self.statistical_tests = {}
    
    def load_results(self, results_file: Optional[str] = None):
        """Load benchmark results from file"""
        if results_file:
            filepath = Path(results_file)
        else:
            # Find latest results file
            result_files = list(self.results_dir.glob("benchmark_results_*.json"))
            if not result_files:
                raise ValueError(f"No benchmark results found in {self.results_dir}")
            
            filepath = max(result_files, key=lambda p: p.stat().st_mtime)
        
        with open(filepath, 'r') as f:
            self.results_data = json.load(f)
        
        logger.info(f"📂 Loaded benchmark results from {filepath}")
        logger.info(f"   Documents: {len(self.results_data.get('detailed_results', {}))}")
    
    def aggregate_metrics(self) -> Dict[str, Any]:
        """Perform comprehensive metrics aggregation"""
        if not self.results_data:
            self.load_results()
        
        detailed_results = self.results_data.get("detailed_results", {})
        summary_stats = self.results_data.get("summary_statistics", {})
        
        # Basic aggregation
        aggregated = {
            "dataset_summary": {
                "total_documents": len(detailed_results),
                "successful_documents": sum(1 for r in detailed_results.values() if r.get("success")),
                "failed_documents": sum(1 for r in detailed_results.values() if "error" in r),
                "success_rate": summary_stats.get("basic", {}).get("success_rate", 0)
            },
            "performance_metrics": self._aggregate_performance_metrics(detailed_results),
            "agent_performance": self._aggregate_agent_performance(detailed_results),
            "risk_analysis": self._analyze_risk_patterns(detailed_results),
            "modality_contribution": self._analyze_modality_contribution(detailed_results),
            "statistical_significance": self._perform_statistical_tests(detailed_results)
        }
        
        self.aggregated_metrics = aggregated
        return aggregated
    
    def _aggregate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate processing performance metrics"""
        successful_results = [r for r in results.values() if r.get("success")]
        
        if not successful_results:
            return {}
        
        # Extract performance metrics
        processing_times = []
        risk_scores = []
        extracted_counts = []
        contradiction_counts = []
        
        for result in successful_results:
            processing_times.append(result.get("processing_time", 0))
            risk_scores.append(result.get("risk_score", 0))
            extracted_counts.append(result.get("extracted_fields", 0))
            contradiction_counts.append(result.get("contradictions", 0))
        
        # Statistical analysis
        def analyze_distribution(data: List[float], name: str) -> Dict[str, Any]:
            if not data:
                return {}
            
            return {
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "std": float(np.std(data)),
                "min": float(min(data)),
                "max": float(max(data)),
                "percentiles": {
                    "25th": float(np.percentile(data, 25)),
                    "75th": float(np.percentile(data, 75)),
                    "95th": float(np.percentile(data, 95))
                },
                "distribution_type": self._identify_distribution_type(data, name)
            }
        
        return {
            "processing_time": analyze_distribution(processing_times, "processing_time"),
            "risk_scores": analyze_distribution(risk_scores, "risk_scores"),
            "extracted_fields": analyze_distribution(extracted_counts, "extracted_fields"),
            "contradictions": analyze_distribution(contradiction_counts, "contradictions")
        }
    
    def _aggregate_agent_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate agent-level performance metrics"""
        successful_results = [r for r in results.values() if r.get("success")]
        
        agent_metrics = {
            "vision": {"validation_scores": [], "confidences": []},
            "text": {"validation_scores": [], "confidences": []},
            "fusion": {"validation_scores": [], "confidences": []},
            "reasoning": {"validation_scores": [], "confidences": []}
        }
        
        for result in successful_results:
            agent_perf = result.get("agent_performance", {})
            if agent_perf:
                # Extract validation scores
                val_score = agent_perf.get("validation_score", 0)
                if val_score > 0:
                    for agent in agent_metrics:
                        agent_metrics[agent]["validation_scores"].append(val_score)
                
                # Extract confidence scores
                conf_scores = agent_perf.get("confidence_scores", {})
                for agent, conf_key in [("vision", "visual_analysis"), 
                                      ("text", "text_extraction"),
                                      ("fusion", "fusion"),
                                      ("reasoning", "validation")]:
                    if conf_key in conf_scores:
                        agent_metrics[agent]["confidences"].append(conf_scores[conf_key])
        
        # Calculate agent performance metrics
        agent_performance = {}
        for agent, metrics in agent_metrics.items():
            if metrics["validation_scores"]:
                agent_performance[agent] = {
                    "average_validation_score": float(np.mean(metrics["validation_scores"])),
                    "average_confidence": float(np.mean(metrics["confidences"])) if metrics["confidences"] else 0,
                    "validation_score_std": float(np.std(metrics["validation_scores"])),
                    "score_distribution": self._calculate_score_distribution(metrics["validation_scores"])
                }
        
        return agent_performance
    
    def _analyze_risk_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk patterns and correlations"""
        successful_results = [r for r in results.values() if r.get("success")]
        
        if len(successful_results) < 2:
            return {}
        
        # Extract risk-related data
        risk_scores = []
        processing_times = []
        extracted_counts = []
        contradiction_counts = []
        
        for result in successful_results:
            risk_scores.append(result.get("risk_score", 0))
            processing_times.append(result.get("processing_time", 0))
            extracted_counts.append(result.get("extracted_fields", 0))
            contradiction_counts.append(result.get("contradictions", 0))
        
        # Calculate correlations
        correlations = {}
        try:
            correlations["risk_processing_time"] = float(np.corrcoef(risk_scores, processing_times)[0, 1])
            correlations["risk_extracted_fields"] = float(np.corrcoef(risk_scores, extracted_counts)[0, 1])
            correlations["risk_contradictions"] = float(np.corrcoef(risk_scores, contradiction_counts)[0, 1])
            correlations["processing_time_extracted_fields"] = float(np.corrcoef(processing_times, extracted_counts)[0, 1])
        except:
            # Handle cases where correlation can't be computed
            pass
        
        # Risk clustering analysis
        risk_clusters = self._cluster_risk_scores(risk_scores)
        
        return {
            "correlations": correlations,
            "risk_clusters": risk_clusters,
            "high_risk_patterns": self._identify_high_risk_patterns(successful_results)
        }
    
    def _analyze_modality_contribution(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contribution of different modalities"""
        successful_results = [r for r in results.values() if r.get("success")]
        
        # Extract evaluation metrics for modality analysis
        text_scores = []
        visual_scores = []
        fusion_scores = []
        
        for result in successful_results:
            eval_report = result.get("evaluation_report")
            if eval_report and "metrics" in eval_report:
                metrics = eval_report["metrics"]
                text_scores.append(metrics.get("text_extraction_score", metrics.get("entity_precision", 0)))
                visual_scores.append(metrics.get("visual_analysis_score", 0.5))
                fusion_scores.append(metrics.get("fusion_score", metrics.get("alignment_accuracy", 0)))
        
        modality_analysis = {}
        if text_scores:
            modality_analysis["text"] = {
                "average_score": float(np.mean(text_scores)),
                "contribution_percentage": 0.4  # Placeholder - real calculation would need more data
            }
        
        if visual_scores:
            modality_analysis["visual"] = {
                "average_score": float(np.mean(visual_scores)),
                "contribution_percentage": 0.3
            }
        
        if fusion_scores:
            modality_analysis["fusion"] = {
                "average_score": float(np.mean(fusion_scores)),
                "contribution_percentage": 0.3
            }
        
        # Calculate modality importance
        if len(modality_analysis) >= 2:
            total_score = sum(mod["average_score"] for mod in modality_analysis.values())
            for modality in modality_analysis:
                modality_analysis[modality]["relative_importance"] = (
                    modality_analysis[modality]["average_score"] / total_score
                    if total_score > 0 else 0
                )
        
        return modality_analysis
    
    def _perform_statistical_tests(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        successful_results = [r for r in results.values() if r.get("success")]
        
        if len(successful_results) < 10:  # Need sufficient samples
            return {"insufficient_data": True}
        
        # Group by document type if available
        doc_type_groups = {}
        for result in successful_results:
            doc_type = result.get("document_type", "unknown")
            if doc_type not in doc_type_groups:
                doc_type_groups[doc_type] = []
            doc_type_groups[doc_type].append(result.get("risk_score", 0))
        
        statistical_tests = {}
        
        # Compare risk scores across document types (if enough types)
        if len(doc_type_groups) >= 2:
            doc_types = list(doc_type_groups.keys())
            group1 = doc_type_groups[doc_types[0]]
            group2 = doc_type_groups[doc_types[1]]
            
            if len(group1) >= 3 and len(group2) >= 3:
                try:
                    # T-test for difference in means
                    t_stat, p_value = stats.ttest_ind(group1, group2)
                    statistical_tests["document_type_risk_difference"] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                        "groups": [doc_types[0], doc_types[1]],
                        "group1_mean": float(np.mean(group1)),
                        "group2_mean": float(np.mean(group2))
                    }
                except:
                    pass
        
        # Test for processing time vs extracted fields correlation
        processing_times = [r.get("processing_time", 0) for r in successful_results]
        extracted_fields = [r.get("extracted_fields", 0) for r in successful_results]
        
        if len(processing_times) >= 3 and len(extracted_fields) >= 3:
            try:
                correlation, p_value = stats.pearsonr(processing_times, extracted_fields)
                statistical_tests["processing_extracted_correlation"] = {
                    "correlation": float(correlation),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05
                }
            except:
                pass
        
        return statistical_tests
    
    def _identify_distribution_type(self, data: List[float], name: str) -> str:
        """Identify the type of distribution"""
        if not data or len(data) < 10:
            return "insufficient_data"
        
        try:
            # Test for normality
            stat, p_value = stats.shapiro(data)
            if p_value > 0.05:
                return "normal"
            
            # Test for log-normal
            log_data = [np.log(x + 1e-10) for x in data if x > 0]
            if len(log_data) >= 10:
                stat, p_value = stats.shapiro(log_data)
                if p_value > 0.05:
                    return "log_normal"
            
            # Check for exponential
            if all(x >= 0 for x in data):
                stat, p_value = stats.kstest(data, 'expon', args=(0, np.mean(data)))
                if p_value > 0.05:
                    return "exponential"
            
            return "non_normal"
        except:
            return "unknown"
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, Any]:
        """Calculate score distribution statistics"""
        if not scores:
            return {}
        
        bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
        hist, _ = np.histogram(scores, bins=bins)
        
        return {
            "histogram": hist.tolist(),
            "bin_edges": bins,
            "percentage_high_confidence": sum(1 for s in scores if s >= 0.7) / len(scores),
            "percentage_low_confidence": sum(1 for s in scores if s < 0.5) / len(scores)
        }
    
    def _cluster_risk_scores(self, risk_scores: List[float]) -> Dict[str, Any]:
        """Cluster risk scores into categories"""
        if not risk_scores:
            return {}
        
        # Simple threshold-based clustering
        low_risk = [s for s in risk_scores if s < 0.3]
        medium_risk = [s for s in risk_scores if 0.3 <= s < 0.5]
        high_risk = [s for s in risk_scores if 0.5 <= s < 0.7]
        critical_risk = [s for s in risk_scores if s >= 0.7]
        
        return {
            "low_risk": {
                "count": len(low_risk),
                "percentage": len(low_risk) / len(risk_scores),
                "average_score": float(np.mean(low_risk)) if low_risk else 0
            },
            "medium_risk": {
                "count": len(medium_risk),
                "percentage": len(medium_risk) / len(risk_scores),
                "average_score": float(np.mean(medium_risk)) if medium_risk else 0
            },
            "high_risk": {
                "count": len(high_risk),
                "percentage": len(high_risk) / len(risk_scores),
                "average_score": float(np.mean(high_risk)) if high_risk else 0
            },
            "critical_risk": {
                "count": len(critical_risk),
                "percentage": len(critical_risk) / len(risk_scores),
                "average_score": float(np.mean(critical_risk)) if critical_risk else 0
            }
        }
    
    def _identify_high_risk_patterns(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """Identify patterns in high-risk documents"""
        high_risk_docs = [r for r in results if r.get("risk_score", 0) >= 0.5]
        
        patterns = []
        
        # Pattern 1: High contradictions
        high_contradiction_docs = [r for r in high_risk_docs if r.get("contradictions", 0) >= 3]
        if high_contradiction_docs:
            patterns.append({
                "pattern": "high_contradictions",
                "count": len(high_contradiction_docs),
                "percentage_of_high_risk": len(high_contradiction_docs) / len(high_risk_docs) if high_risk_docs else 0,
                "average_contradictions": float(np.mean([r.get("contradictions", 0) for r in high_contradiction_docs])),
                "description": "Documents with 3+ contradictions tend to be high risk"
            })
        
        # Pattern 2: Low extracted fields but high processing time
        inefficient_docs = [
            r for r in high_risk_docs 
            if r.get("extracted_fields", 0) < 3 and r.get("processing_time", 0) > 10
        ]
        if inefficient_docs:
            patterns.append({
                "pattern": "low_efficiency",
                "count": len(inefficient_docs),
                "percentage_of_high_risk": len(inefficient_docs) / len(high_risk_docs) if high_risk_docs else 0,
                "average_extracted_fields": float(np.mean([r.get("extracted_fields", 0) for r in inefficient_docs])),
                "average_processing_time": float(np.mean([r.get("processing_time", 0) for r in inefficient_docs])),
                "description": "Documents with few extracted fields but high processing time"
            })
        
        return patterns
    
    def generate_visualizations(self, output_dir: Optional[str] = None):
        """Generate comprehensive visualizations"""
        if not self.aggregated_metrics:
            self.aggregate_metrics()
        
        viz_dir = Path(output_dir) if output_dir else self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Generate various plots
        self._generate_performance_plots(viz_dir)
        self._generate_agent_comparison_plots(viz_dir)
        self._generate_risk_analysis_plots(viz_dir)
        self._generate_modality_contribution_plots(viz_dir)
        
        logger.info(f"📈 Generated visualizations in {viz_dir}")
    
    def _generate_performance_plots(self, output_dir: Path):
        """Generate performance metric plots"""
        perf_metrics = self.aggregated_metrics.get("performance_metrics", {})
        
        # Processing time distribution
        if "processing_time" in perf_metrics:
            times = self._get_sample_data("processing_time")
            fig = px.histogram(
                x=times,
                nbins=20,
                title="Processing Time Distribution",
                labels={"x": "Processing Time (seconds)", "y": "Count"},
                template="plotly_dark"
            )
            fig.write_html(output_dir / "processing_time_distribution.html")
        
        # Risk score distribution
        if "risk_scores" in perf_metrics:
            risk_scores = self._get_sample_data("risk_score")
            fig = px.box(
                y=risk_scores,
                title="Risk Score Distribution",
                labels={"y": "Risk Score"},
                template="plotly_dark"
            )
            fig.write_html(output_dir / "risk_score_distribution.html")
    
    def _generate_agent_comparison_plots(self, output_dir: Path):
        """Generate agent comparison plots"""
        agent_perf = self.aggregated_metrics.get("agent_performance", {})
        
        if agent_perf:
            agents = list(agent_perf.keys())
            validation_scores = [agent_perf[agent]["average_validation_score"] for agent in agents]
            confidences = [agent_perf[agent]["average_confidence"] for agent in agents]
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Agent Validation Scores", "Agent Confidence Scores")
            )
            
            fig.add_trace(
                go.Bar(x=agents, y=validation_scores, name="Validation Score"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=agents, y=confidences, name="Confidence"),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Agent Performance Comparison",
                template="plotly_dark",
                showlegend=True
            )
            
            fig.write_html(output_dir / "agent_performance_comparison.html")
    
    def _generate_risk_analysis_plots(self, output_dir: Path):
        """Generate risk analysis plots"""
        risk_analysis = self.aggregated_metrics.get("risk_analysis", {})
        
        if "risk_clusters" in risk_analysis:
            clusters = risk_analysis["risk_clusters"]
            
            labels = ["Low", "Medium", "High", "Critical"]
            counts = [clusters.get(f"{level.lower()}_risk", {}).get("count", 0) for level in labels]
            
            fig = px.pie(
                values=counts,
                names=labels,
                title="Risk Level Distribution",
                template="plotly_dark"
            )
            
            fig.write_html(output_dir / "risk_level_distribution.html")
    
    def _generate_modality_contribution_plots(self, output_dir: Path):
        """Generate modality contribution plots"""
        modality_analysis = self.aggregated_metrics.get("modality_contribution", {})
        
        if modality_analysis:
            modalities = list(modality_analysis.keys())
            scores = [modality_analysis[mod]["average_score"] for mod in modalities]
            importance = [modality_analysis[mod].get("relative_importance", 0) for mod in modalities]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Modality Performance Scores", "Relative Modality Importance")
            )
            
            fig.add_trace(
                go.Bar(x=modalities, y=scores, name="Average Score"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Pie(labels=modalities, values=importance, name="Importance"),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Modality Contribution Analysis",
                template="plotly_dark",
                showlegend=True
            )
            
            fig.write_html(output_dir / "modality_contribution_analysis.html")
    
    def _get_sample_data(self, metric_name: str) -> List[float]:
        """Extract sample data for visualization"""
        # This would extract from detailed results
        # Simplified for now
        return []
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
        if not self.aggregated_metrics:
            self.aggregate_metrics()
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "summary": {
                "total_metrics_calculated": len(self.aggregated_metrics),
                "statistical_significance_tests": len(self.statistical_tests),
                "key_findings": self._extract_key_findings()
            },
            "detailed_metrics": self.aggregated_metrics,
            "statistical_tests": self.statistical_tests,
            "recommendations": self._generate_recommendations()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"📋 Generated metrics report: {output_file}")
        
        return report
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from aggregated metrics"""
        findings = []
        
        # Performance findings
        perf_metrics = self.aggregated_metrics.get("performance_metrics", {})
        if "processing_time" in perf_metrics:
            avg_time = perf_metrics["processing_time"]["mean"]
            findings.append(f"Average processing time: {avg_time:.2f} seconds")
        
        # Agent performance findings
        agent_perf = self.aggregated_metrics.get("agent_performance", {})
        if agent_perf:
            best_agent = max(agent_perf.items(), key=lambda x: x[1]["average_validation_score"])[0]
            findings.append(f"Highest performing agent: {best_agent}")
        
        # Risk findings
        risk_analysis = self.aggregated_metrics.get("risk_analysis", {})
        if "risk_clusters" in risk_analysis:
            critical_pct = risk_analysis["risk_clusters"].get("critical_risk", {}).get("percentage", 0)
            findings.append(f"Critical risk documents: {critical_pct:.1%}")
        
        return findings
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate system improvement recommendations"""
        recommendations = []
        
        agent_perf = self.aggregated_metrics.get("agent_performance", {})
        if agent_perf:
            # Find weakest agent
            weakest_agent = min(agent_perf.items(), key=lambda x: x[1]["average_validation_score"])[0]
            recommendations.append({
                "area": "agent_performance",
                "recommendation": f"Improve {weakest_agent} agent validation logic",
                "priority": "medium",
                "impact": f"Current validation score: {agent_perf[weakest_agent]['average_validation_score']:.2f}"
            })
        
        perf_metrics = self.aggregated_metrics.get("performance_metrics", {})
        if "processing_time" in perf_metrics:
            if perf_metrics["processing_time"]["std"] > perf_metrics["processing_time"]["mean"] * 0.5:
                recommendations.append({
                    "area": "performance",
                    "recommendation": "Optimize processing pipeline for consistent performance",
                    "priority": "high",
                    "impact": "High variance in processing times detected"
                })
        
        return recommendations